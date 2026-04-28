"""
LiveKit Agent Worker — AI Voice Agent (doctor appointment booking)

Architecture:
    Vobiz SIP trunk  →  LiveKit SIP bridge  →  This worker
    Per call:
        Silero VAD  →  Sarvam Saaras v2 STT  →  ConversationManager  →  Sarvam bulbul:v3 TTS

Run locally:
    python agent_livekit.py start

Required env vars (add to Railway + .env):
    LIVEKIT_URL            wss://<your-project>.livekit.cloud
    LIVEKIT_API_KEY        your livekit api key
    LIVEKIT_API_SECRET     your livekit api secret
    SARVAM_API_KEY         sarvam subscription key
    GROQ_API_KEY           groq api key
    GOOGLE_SERVICE_ACCOUNT_JSON  path to service account JSON
    (optional) SARVAM_TTS_SPEAKER   default: priya
    (optional) SARVAM_LANGUAGE_CODE  default: hi-IN
    (optional) DOCTOR_NAME / CLINIC_NAME / CLINIC_HOURS
"""

import asyncio
import audioop  # built-in on Python ≤ 3.12; Railway uses 3.11
import base64
import io
import logging
import os
import sys
import threading
from uuid import uuid4
import wave
from http.server import BaseHTTPRequestHandler, HTTPServer

import httpx
from dotenv import load_dotenv

from livekit import rtc
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    WorkerOptions,
    cli,
    llm,
    stt,
    tts,
)
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.agents.types import APIConnectOptions, DEFAULT_API_CONNECT_OPTIONS
from livekit.plugins import silero

# livekit-agents 0.12.x removed stt.AudioBuffer — define locally
AudioBuffer = list[rtc.AudioFrame]

from agent.conversation import ConversationManager, State
from services.calendar_service import CalendarService
from services.cancellation_monitor import run_monitor

load_dotenv()

# Decode Google credentials from base64 env var (Railway deployment)
_b64_creds = os.getenv("GOOGLE_SERVICE_ACCOUNT_B64")
if _b64_creds:
    import base64 as _b64mod
    _creds_path = "/tmp/google-service-account.json"
    with open(_creds_path, "wb") as _f:
        _f.write(_b64mod.b64decode(_b64_creds))
    os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"] = _creds_path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("agent")

_SARVAM_KEY   = os.environ["SARVAM_API_KEY"]
_TTS_SPEAKER  = os.getenv("SARVAM_TTS_SPEAKER", "priya")
_TTS_LANG     = os.getenv("SARVAM_LANGUAGE_CODE", "hi-IN")

# Azure TTS config
_AZURE_SPEECH_KEY    = os.getenv("AZURE_SPEECH_KEY", "")
_AZURE_SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION", "eastus")
_AZURE_TTS_VOICE     = os.getenv("AZURE_TTS_VOICE", "hi-IN-SwaraNeural")

# Deepgram STT config
_DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")

# Shared HTTP client — keeps TLS connections alive across STT + TTS calls
_http = httpx.AsyncClient(
    timeout=15,
    limits=httpx.Limits(max_keepalive_connections=5, max_connections=10, keepalive_expiry=30.0),
)


# ---------------------------------------------------------------------------
# Health-check HTTP server (Railway needs a live port)
# ---------------------------------------------------------------------------

class _HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"status":"ok"}')

    def log_message(self, *_):  # silence access logs
        pass


def _start_health_server():
    port = int(os.getenv("PORT", 8000))
    server = HTTPServer(("0.0.0.0", port), _HealthHandler)
    logger.info(f"[Health] Listening on :{port}")
    server.serve_forever()


# ---------------------------------------------------------------------------
# Deepgram nova-2 — STT plugin (REST, no native SDK)
# ---------------------------------------------------------------------------

class DeepgramSTT(stt.STT):
    """
    Wraps Deepgram nova-2 REST API as a LiveKit STT plugin.
    Better noise handling than Sarvam for phone audio.
    Supports Hindi + English codemix via language=hi.
    """

    def __init__(self):
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=False,
                interim_results=False,
            )
        )
        self._http = _http

    async def _recognize_impl(
        self,
        buffer,
        *,
        language=None,
        conn_options=None,
    ) -> stt.SpeechEvent:
        if isinstance(buffer, rtc.AudioFrame):
            frames: list = [buffer]
        else:
            frames = list(buffer) if buffer else []

        if not frames:
            return _empty_speech_event()

        sample_rate = frames[0].sample_rate
        num_channels = frames[0].num_channels
        raw_pcm = b"".join(bytes(f.data) for f in frames)

        if num_channels > 1:
            raw_pcm = audioop.tomono(raw_pcm, 2, 0.5, 0.5)

        # Deepgram works best at 16kHz
        if sample_rate != 16000:
            raw_pcm, _ = audioop.ratecv(raw_pcm, 2, 1, sample_rate, 16000, None)

        wav_buf = io.BytesIO()
        with wave.open(wav_buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(raw_pcm)
        wav_buf.seek(0)

        try:
            resp = await self._http.post(
                "https://api.deepgram.com/v1/listen",
                params={
                    "model":        "nova-3",
                    "language":     "multi",
                    "smart_format": "true",
                    "punctuate":    "false",
                },
                headers={
                    "Authorization": f"Token {_DEEPGRAM_API_KEY}",
                    "Content-Type":  "audio/wav",
                },
                content=wav_buf.read(),
            )
            resp.raise_for_status()
            data = resp.json()
            alts = data.get("results", {}).get("channels", [{}])[0].get("alternatives", [{}])
            alt = alts[0] if alts else {}
            confidence = alt.get("confidence", 1.0)
            text = alt.get("transcript", "").strip() if confidence >= 0.6 else ""
        except Exception as e:
            logger.error(f"[DeepgramSTT Error] {e}")
            text = ""
            confidence = 0.0

        logger.info(f"[STT] {text!r} (conf={confidence:.2f})")
        if not text:
            return _empty_speech_event()

        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[stt.SpeechData(language="hi", text=text)],
        )

    def stream(self, *, language=None, conn_options=None):
        raise NotImplementedError("DeepgramSTT batch mode does not support streaming")


# ---------------------------------------------------------------------------
# Sarvam Saaras v2 codemix — STT plugin
# ---------------------------------------------------------------------------

class SarvamSTT(stt.STT):
    """
    Wraps Sarvam Saaras v2/v3 codemix REST API as a LiveKit STT plugin.
    """

    def __init__(self):
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=False,
                interim_results=False,
            )
        )
        self._http = _http  # shared module-level client

    async def _recognize_impl(
        self,
        buffer,
        *,
        language=None,
        conn_options=None,
    ) -> stt.SpeechEvent:
        # Handle both single AudioFrame (new API) and AudioBuffer/list (old API)
        if isinstance(buffer, rtc.AudioFrame):
            frames: list = [buffer]
        else:
            frames = list(buffer) if buffer else []

        if not frames:
            return _empty_speech_event()

        # Merge all frames into one raw PCM blob
        sample_rate = frames[0].sample_rate
        num_channels = frames[0].num_channels
        raw_pcm = b"".join(bytes(f.data) for f in frames)

        # Flatten to mono
        if num_channels > 1:
            raw_pcm = audioop.tomono(raw_pcm, 2, 0.5, 0.5)

        # Sarvam expects 16 kHz input
        if sample_rate != 16000:
            raw_pcm, _ = audioop.ratecv(raw_pcm, 2, 1, sample_rate, 16000, None)

        wav_buf = io.BytesIO()
        with wave.open(wav_buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(raw_pcm)
        wav_buf.seek(0)

        try:
            resp = await self._http.post(
                "https://api.sarvam.ai/speech-to-text",
                headers={"api-subscription-key": _SARVAM_KEY},
                files={"file": ("audio.wav", wav_buf.read(), "audio/wav")},
                data={
                    "model": "saaras:v3",
                    "mode": "codemix",
                    "language_code": "unknown",
                },
            )
            resp.raise_for_status()
            text = resp.json().get("transcript", "").strip()
        except Exception as e:
            logger.error(f"[STT Error] {e}")
            text = ""

        logger.info(f"[STT] {text!r}")

        # Drop transcripts that are too short or pure filler noise
        if len(text) < 1:
            return _empty_speech_event()

        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[stt.SpeechData(language="hi-IN", text=text)],
        )

    def stream(self, *, language=None, conn_options=None):
        raise NotImplementedError("Sarvam STT does not support streaming")


def _empty_speech_event() -> stt.SpeechEvent:
    return stt.SpeechEvent(
        type=stt.SpeechEventType.FINAL_TRANSCRIPT,
        alternatives=[stt.SpeechData(language="hi-IN", text="")],
    )


# ---------------------------------------------------------------------------
# Azure Cognitive Services — TTS plugin (REST, no native SDK)
# ---------------------------------------------------------------------------

class AzureTTS(tts.TTS):
    """
    Azure TTS via REST API — no native SDK, no libuuid dependency.
    Requests riff-8khz-16bit-mono-pcm directly for phone calls.
    """

    def __init__(self):
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=8000,
            num_channels=1,
        )
        self._http = _http

    def synthesize(self, text: str, **kwargs) -> "AzureTTSStream":
        return AzureTTSStream(tts=self, input_text=text, http=self._http, **kwargs)


class AzureTTSStream(tts.ChunkedStream):
    def __init__(self, *, tts: AzureTTS, input_text: str, http: httpx.AsyncClient, **kwargs):
        super().__init__(tts=tts, input_text=input_text, **kwargs)
        self._http = http

    async def _run(self, output_emitter=None) -> None:
        ssml = (
            f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="hi-IN">'
            f'<voice name="{_AZURE_TTS_VOICE}"><prosody rate="+15%">{self._input_text}</prosody></voice>'
            f'</speak>'
        )
        endpoint = f"https://{_AZURE_SPEECH_REGION}.tts.speech.microsoft.com/cognitiveservices/v1"
        headers = {
            "Ocp-Apim-Subscription-Key": _AZURE_SPEECH_KEY,
            "Content-Type": "application/ssml+xml",
            "X-Microsoft-OutputFormat": "riff-8khz-16bit-mono-pcm",
            "User-Agent": "ai-voice-agent",
        }

        try:
            resp = await self._http.post(endpoint, headers=headers, content=ssml.encode("utf-8"))
            resp.raise_for_status()
            pcm = _wav_to_pcm(resp.content)
            frame = rtc.AudioFrame(
                data=bytearray(pcm),
                sample_rate=8000,
                num_channels=1,
                samples_per_channel=len(pcm) // 2,
            )

            if output_emitter is not None:
                request_id = uuid4().hex
                output_emitter.initialize(
                    request_id=request_id,
                    sample_rate=8000,
                    num_channels=1,
                    mime_type="audio/pcm",
                )
                output_emitter.push(frame)
                output_emitter.flush()
            else:
                self._event_ch.send_nowait(
                    tts.SynthesizedAudio(
                        request_id=getattr(self, "_request_id", uuid4().hex),
                        frame=frame,
                    )
                )
        except Exception as e:
            logger.error(f"[AzureTTS Error] {e}")


# ---------------------------------------------------------------------------
# Sarvam bulbul:v3 — TTS plugin
# ---------------------------------------------------------------------------

class SarvamTTS(tts.TTS):
    """
    Wraps Sarvam bulbul:v3 REST API as a LiveKit TTS plugin.
    Requests 8 kHz output directly — no resampling needed.
    """

    def __init__(self):
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=8000,
            num_channels=1,
        )
        self._http = _http  # shared module-level client

    def synthesize(self, text: str, **kwargs) -> "SarvamTTSStream":
        # **kwargs forwards conn_options added in livekit-agents 0.12.x
        return SarvamTTSStream(tts=self, input_text=text, http=self._http, **kwargs)


class SarvamTTSStream(tts.ChunkedStream):
    def __init__(self, *, tts: SarvamTTS, input_text: str, http: httpx.AsyncClient, **kwargs):
        super().__init__(tts=tts, input_text=input_text, **kwargs)  # forwards conn_options
        self._http = http

    async def _run(self, output_emitter=None) -> None:
        payload = {
            "text":                 self._input_text,
            "target_language_code": _TTS_LANG,
            "speaker":              _TTS_SPEAKER,
            "model":                "bulbul:v3",
            "speech_sample_rate":   8000,
            "pace":                 1.15,
        }
        headers = {
            "api-subscription-key": _SARVAM_KEY,
            "Content-Type":         "application/json",
        }

        for attempt in range(2):
            try:
                resp = await self._http.post(
                    "https://api.sarvam.ai/text-to-speech",
                    headers=headers,
                    json=payload,
                )
                if resp.status_code == 429:
                    logger.warning(f"[TTS] 429 rate-limit — backing off (attempt {attempt + 1})")
                    await asyncio.sleep(1.0)
                    continue
                resp.raise_for_status()
                audios = resp.json().get("audios", [])
                if not audios:
                    logger.error("[TTS] Empty audios list in response")
                    return

                pcm   = _wav_to_pcm(base64.b64decode(audios[0]))
                frame = rtc.AudioFrame(
                    data=bytearray(pcm),
                    sample_rate=8000,
                    num_channels=1,
                    samples_per_channel=len(pcm) // 2,
                )

                if output_emitter is not None:
                    # New API (livekit-agents 0.12+)
                    request_id = uuid4().hex
                    output_emitter.initialize(
                        request_id=request_id,
                        sample_rate=8000,
                        num_channels=1,
                        mime_type="audio/pcm",
                    )
                    output_emitter.push(frame)
                    output_emitter.flush()
                else:
                    # Fallback for older SDK versions
                    self._event_ch.send_nowait(
                        tts.SynthesizedAudio(
                            request_id=getattr(self, "_request_id", uuid4().hex),
                            frame=frame,
                        )
                    )
                return

            except Exception as e:
                logger.error(f"[TTS Error] {e}")
                return


def _wav_to_pcm(wav_bytes: bytes) -> bytes:
    buf = io.BytesIO(wav_bytes)
    with wave.open(buf, "rb") as wf:
        return wf.readframes(wf.getnframes())


# ---------------------------------------------------------------------------
# ConversationManager wrapped as a LiveKit LLM plugin
# ---------------------------------------------------------------------------

class ConversationLLM(llm.LLM):
    """
    Thin adapter: LiveKit calls chat() after each STT transcript.
    We forward the transcript to ConversationManager.process_turn() and
    emit the response as a single ChatChunk.
    """

    def __init__(self, conv: ConversationManager):
        super().__init__()
        self._conv = conv

    def chat(
        self,
        *,
        chat_ctx: llm.ChatContext,
        tools=None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        **kwargs,
    ) -> "ConvStream":
        # Extract the last user message — that is the STT transcript
        transcript = ""
        for msg in reversed(chat_ctx.messages):
            if msg.role == "user" and isinstance(msg.content, str):
                transcript = msg.content
                break
        return ConvStream(
            llm=self,
            chat_ctx=chat_ctx,
            conn_options=conn_options,
            transcript=transcript,
            conv=self._conv,
        )


class ConvStream(llm.LLMStream):
    def __init__(
        self,
        *,
        llm: ConversationLLM,
        chat_ctx: llm.ChatContext,
        conn_options: APIConnectOptions,
        transcript: str,
        conv: ConversationManager,
    ):
        super().__init__(llm=llm, chat_ctx=chat_ctx, fnc_ctx=None, conn_options=conn_options)
        self._transcript = transcript
        self._conv       = conv

    async def _run(self) -> None:
        try:
            response = await self._conv.process_turn(self._transcript)
        except Exception as e:
            logger.error(f"[ConvLLM Error] {e}")
            import traceback; traceback.print_exc()
            response = ""

        if not response:
            return

        logger.info(f"[Agent] {response!r}")
        self._event_ch.send_nowait(
            llm.ChatChunk(
                request_id=uuid4().hex,
                choices=[
                    llm.Choice(
                        delta=llm.ChoiceDelta(role="assistant", content=response),
                        index=0,
                    )
                ],
            )
        )


# ---------------------------------------------------------------------------
# Per-call entrypoint
# ---------------------------------------------------------------------------

async def entrypoint(ctx: JobContext):
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # Wait for the SIP participant (Vobiz caller) to join the room
    participant = await ctx.wait_for_participant()

    # Vobiz SIP attributes — log all keys for debugging, then extract caller number
    attrs = participant.attributes or {}
    logger.info(f"[SIP] Raw attributes: {dict(attrs)}")
    raw_from = (
        attrs.get("sip.phoneNumber")   # LiveKit SIP standard key
        or attrs.get("sip.from")
        or attrs.get("sip.callerNumber")
        or attrs.get("sip.callerid")
        or attrs.get("X-Caller-Number")
        or attrs.get("from")
        or ""
    )
    # Handle SIP URI format: sip:+919876543210@domain.com → +919876543210
    import re as _re
    _m = _re.search(r'(?:sip:|tel:)([+\d]+)', raw_from)
    caller_number = _m.group(1) if _m else raw_from
    logger.info(f"[Call] room={ctx.room.name!r}  caller={caller_number!r}")

    conv = ConversationManager(caller_phone=caller_number)

    # Use Deepgram STT if key is configured, otherwise fall back to Sarvam
    if _DEEPGRAM_API_KEY:
        _stt = DeepgramSTT()
        logger.info("[STT] Using Deepgram nova-2 (hi)")
    else:
        _stt = SarvamSTT()
        logger.info("[STT] Using Sarvam saaras:v3 (Deepgram key not set)")

    # Use Azure TTS if key is configured, otherwise fall back to Sarvam
    if _AZURE_SPEECH_KEY:
        _tts = AzureTTS()
        logger.info(f"[TTS] Using Azure ({_AZURE_TTS_VOICE})")
    else:
        _tts = SarvamTTS()
        logger.info("[TTS] Using Sarvam (Azure key not set)")

    agent = VoicePipelineAgent(
        vad=silero.VAD.load(
            min_speech_duration=0.1,   # catch short words like 'हाँ', 'ji'
            min_silence_duration=0.4,  # 400ms silence = end of utterance
            activation_threshold=0.5,  # default — balanced for phone audio
        ),
        stt=_stt,
        llm=ConversationLLM(conv),
        tts=_tts,
        min_endpointing_delay=0.6,     # 600ms after VAD end — balance latency vs noise
        allow_interruptions=True,      # barge-in on
    )

    agent.start(ctx.room)

    # Play greeting immediately; block interruptions until it finishes
    greeting = conv.get_greeting()
    logger.info(f"[Greeting] {greeting!r}")
    await agent.say(greeting, allow_interruptions=False)

    # Disconnect after final speech finishes playing.
    # agent_stopped_speaking fires when playout ends — safe to hang up immediately.
    _speech_after_done = asyncio.Event()

    @agent.on("agent_stopped_speaking")
    def _on_agent_stopped_speaking(*_):
        if conv.state == State.DONE:
            logger.info("[Call] Final speech finished — disconnecting in 1s")
            _speech_after_done.set()

    async def _watch_done():
        # Primary: wait for agent_stopped_speaking event after DONE
        # Fallback: if event never fires, disconnect 10s after DONE state is set
        done_at = None
        while not _speech_after_done.is_set():
            await asyncio.sleep(0.5)
            if conv.state == State.DONE:
                if done_at is None:
                    done_at = asyncio.get_event_loop().time()
                    logger.info("[Call] DONE state detected (poll fallback)")
                elif asyncio.get_event_loop().time() - done_at > 10.0:
                    logger.info("[Call] Fallback timeout — disconnecting room")
                    try:
                        await ctx.room.disconnect()
                    except Exception:
                        pass
                    return
        await asyncio.sleep(1.0)  # 1s buffer after playout ends
        logger.info("[Call] Booking complete — disconnecting room")
        try:
            await ctx.room.disconnect()
        except Exception:
            pass

    asyncio.create_task(_watch_done())

    call_ended = asyncio.Event()

    @ctx.room.on("participant_disconnected")
    def _on_participant_disconnected(p):
        if p.identity == participant.identity:
            call_ended.set()

    @ctx.room.on("disconnected")
    def _on_room_disconnected(*_):
        call_ended.set()

    await call_ended.wait()


# ---------------------------------------------------------------------------
# Worker bootstrap
# ---------------------------------------------------------------------------

def _prewarm(proc):
    """Called once per worker process before any jobs arrive."""
    # Pre-load Silero VAD model weights so first call is instant
    silero.VAD.load()
    logger.info("[Prewarm] Silero VAD loaded")


if __name__ == "__main__":
    # Start health-check server in background (Railway port liveness check)
    threading.Thread(target=_start_health_server, daemon=True).start()

    # Start cancellation monitor as an asyncio background task
    _calendar = CalendarService()

    async def _startup_monitor():
        asyncio.create_task(run_monitor(_calendar))
        logger.info("[Startup] Cancellation monitor started")

    # Run the LiveKit agent worker
    # Commands:  python agent_livekit.py start
    #            python agent_livekit.py dev      (local testing)
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=_prewarm,
            agent_name="ai-voice-agent",
        )
    )
