"""
AI Voice Agent - Twilio bidirectional media stream handler.

Architecture: two concurrent tasks per call
  1. _twilio_loop   : read Twilio WS -> webrtcvad (local) -> buffer -> Deepgram REST API
  2. _pipeline_loop : asyncio.Queue -> ConversationManager -> TTS -> Twilio

webrtcvad runs in-process (~80ms speech-end detection vs 500ms Deepgram endpointing).
When speech ends, the buffered mulaw clip is POSTed to Deepgram nova-2 REST for transcription.
Transcript is put_nowait() into the asyncio.Queue for the pipeline loop.

STT: Deepgram nova-2 REST API — hi, mulaw 8kHz (no streaming connection needed)
VAD: webrtcvad — aggressiveness=2, 80ms silence-end detection, runs fully in-process
TTS: Sarvam bulbul:v3 (persistent HTTP client in SpeechService)
"""

import asyncio
import audioop
import base64
import json
import os
import re
import time

import httpx
from fastapi import WebSocket

TTS_CHUNK = 1600  # ~200ms of mulaw @ 8kHz per send

# Matches any Hindi (Devanagari), Punjabi (Gurmukhi), or Latin letter
_HAS_LETTER = re.compile(r'[a-zA-Z\u0900-\u097F\u0A00-\u0A7F]')

# Energy VAD tuning — all values in 20ms frame units (Twilio sends 20ms chunks)
# No external package needed — uses audioop.rms which is built-in.
VAD_RMS_THRESHOLD  = 180  # RMS above this = active speech (telephone typical: 150-250)
VAD_SILENCE_FRAMES = 4    # 4 × 20ms = 80ms silence → declare speech ended
VAD_MIN_SPEECH_MS  = 80   # discard clips < 80ms (noise bursts, breathing)


class StreamSession:
    def __init__(self, speech_service):
        self.speech         = speech_service
        self.call_sid       = None
        self.stream_sid     = None
        self.conversation   = None
        self._ws            = None
        self._dg_key        = os.environ.get("DEEPGRAM_API_KEY", "")
        # Persistent HTTP client for Deepgram REST — reuses TCP connection
        self._dg_http       = httpx.AsyncClient(timeout=10)
        self._muted_until   = 0.0
        self._processing    = False
        self._transcript_q  = asyncio.Queue()
        self._filler_mulaw: bytes = b""  # pre-cached "ठीक है जी"
        # VAD state — reset after each utterance
        self._vad_buf       = bytearray()
        self._vad_has_speech = False
        self._vad_silence   = 0
        self._vad_sframes   = 0          # speech frame count

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    async def run(self, websocket: WebSocket):
        self._ws = websocket
        await websocket.accept()

        # Twilio sends "connected" before "start" — discard until "start"
        msg = None
        try:
            async for raw in websocket.iter_text():
                candidate = json.loads(raw)
                if candidate.get("event") == "start":
                    msg = candidate
                    break
                print(f"[Stream] Pre-start event: {candidate.get('event')}")
        except Exception as e:
            print(f"[Stream] Error waiting for start: {e}")
            return

        if msg is None:
            print("[Stream] Never received start event")
            return

        await self._handle_start(msg)

        t_pl = asyncio.create_task(self._pipeline_loop(), name="pl_loop")
        try:
            await self._twilio_loop(websocket)
        except Exception as e:
            from agent.conversation import State
            if not (self.conversation and self.conversation.state == State.DONE):
                print(f"[Stream Error] {self.call_sid}: {e}")
        finally:
            t_pl.cancel()
            await asyncio.gather(t_pl, return_exceptions=True)
            await self._dg_http.aclose()

    # ------------------------------------------------------------------
    # Start
    # ------------------------------------------------------------------

    async def _handle_start(self, msg: dict):
        from agent.conversation import ConversationManager
        self.call_sid   = msg["start"]["callSid"]
        self.stream_sid = msg["start"]["streamSid"]
        caller_number   = msg["start"].get("customParameters", {}).get("callerNumber", "")
        self.conversation = ConversationManager(caller_phone=caller_number)
        print(f"[Call] {self.call_sid} from {caller_number}")

        if not self._dg_key:
            print("[Deepgram] ERROR: DEEPGRAM_API_KEY not set")

        print(f"[VAD] Energy VAD ready — RMS threshold={VAD_RMS_THRESHOLD}, "
              f"silence threshold={VAD_SILENCE_FRAMES * 20}ms")

        # Cache filler FIRST — avoids Sarvam 429 race with greeting TTS.
        # Both calls hit Sarvam sequentially, not concurrently.
        await self._cache_filler()

        # Hold mute until greeting finishes; mark event releases it precisely.
        self._muted_until = time.monotonic() + 30.0
        greeting = self.conversation.get_greeting()
        print(f"[Agent] {greeting}")
        asyncio.create_task(self._speak(greeting), name="speak_greeting")

    # ------------------------------------------------------------------
    # Twilio loop: feed audio to VAD; handle mark/stop
    # ------------------------------------------------------------------

    async def _twilio_loop(self, websocket: WebSocket):
        async for raw in websocket.iter_text():
            msg   = json.loads(raw)
            event = msg.get("event")

            if event == "media":
                if msg["media"].get("track", "inbound") == "inbound":
                    mulaw = base64.b64decode(msg["media"]["payload"])
                    self._vad_process(mulaw)

            elif event == "mark":
                # Twilio echoes this back exactly when audio finishes playing on the phone.
                if msg.get("mark", {}).get("name") == "tts_done":
                    self._muted_until = 0.0
                    print("[Audio] Playback complete — mute released")

            elif event == "stop":
                print(f"[Call] Stream stopped: {self.call_sid}")
                break

    # ------------------------------------------------------------------
    # webrtcvad — local speech boundary detection
    # ------------------------------------------------------------------

    def _vad_process(self, mulaw: bytes):
        """
        Called synchronously for every 20ms Twilio chunk.
        Measures RMS energy of PCM — no external package needed.
        When silence follows speech, fires off _transcribe_and_queue as a task.
        """
        # During TTS playback: reset buffer so echo doesn't bleed into next utterance.
        if time.monotonic() < self._muted_until or self._processing:
            self._reset_vad()
            return

        pcm = audioop.ulaw2lin(mulaw, 2)  # mulaw → 16-bit PCM @ 8kHz
        is_speech = audioop.rms(pcm, 2) > VAD_RMS_THRESHOLD

        if is_speech:
            self._vad_has_speech = True
            self._vad_silence    = 0
            self._vad_sframes   += 1
            self._vad_buf.extend(mulaw)
        elif self._vad_has_speech:
            self._vad_silence += 1
            self._vad_buf.extend(mulaw)  # keep trailing silence for natural audio
            if self._vad_silence >= VAD_SILENCE_FRAMES:
                speech_ms = self._vad_sframes * 20
                if speech_ms >= VAD_MIN_SPEECH_MS:
                    audio = bytes(self._vad_buf)
                    self._reset_vad()
                    asyncio.create_task(self._transcribe_and_queue(audio))
                else:
                    self._reset_vad()

    def _reset_vad(self):
        self._vad_buf        = bytearray()
        self._vad_has_speech = False
        self._vad_silence    = 0
        self._vad_sframes    = 0

    # ------------------------------------------------------------------
    # Deepgram REST transcription
    # ------------------------------------------------------------------

    async def _transcribe_and_queue(self, mulaw_audio: bytes):
        """POST buffered mulaw clip to Deepgram nova-2 REST API, queue the transcript."""
        # Early exit if pipeline already busy — prevents double-fire when two VAD
        # clips race before _processing is set True by the pipeline loop.
        if self._processing:
            return
        try:
            resp = await self._dg_http.post(
                "https://api.deepgram.com/v1/listen",
                params={
                    "model":       "nova-2",
                    "language":    "hi",
                    "encoding":    "mulaw",
                    "sample_rate": "8000",
                    "channels":    "1",
                },
                headers={
                    "Authorization": f"Token {self._dg_key}",
                    "Content-Type":  "audio/mulaw",
                },
                content=mulaw_audio,
            )
            resp.raise_for_status()
            transcript = (
                resp.json()["results"]["channels"][0]["alternatives"][0]["transcript"].strip()
            )
            if not transcript or not _HAS_LETTER.search(transcript):
                return
            # Re-check mute/busy — TTS may have started while we awaited Deepgram
            if time.monotonic() < self._muted_until:
                print(f"[STT] Muted (late): {transcript}")
                return
            if self._processing:
                print(f"[STT] Busy (discarded): {transcript}")
                return
            print(f"[STT] {transcript}")
            self._transcript_q.put_nowait(transcript)
        except Exception as e:
            print(f"[STT Error] {e}")

    # ------------------------------------------------------------------
    # Pipeline loop: consume queue, one turn at a time
    # ------------------------------------------------------------------

    async def _pipeline_loop(self):
        while True:
            try:
                transcript = await self._transcript_q.get()
            except asyncio.CancelledError:
                break
            await self._process(transcript)

    async def _process(self, transcript: str):
        self._processing = True
        try:
            from agent.conversation import State
            # LLM call and filler playback run concurrently.
            # Filler ("ठीक है जी") plays in ~300ms while LLM thinks — fills the silence gap.
            response_task = asyncio.create_task(self.conversation.process_turn(transcript))
            await self._play_filler()
            response = await response_task

            if not response:
                print(f"[Pipeline] Empty response for: {transcript!r}")
                return
            print(f"[Agent] {response}")
            await self._speak(response)

            if self.conversation.state == State.DONE:
                # Wait for Twilio to echo back the mark event — that means audio
                # finished playing on the caller's phone. Only then close.
                deadline = time.monotonic() + 15.0
                while self._muted_until > 0.0 and time.monotonic() < deadline:
                    await asyncio.sleep(0.1)
                await asyncio.sleep(0.3)  # small grace buffer
                try:
                    await self._ws.close()
                except Exception:
                    pass
        except Exception as e:
            print(f"[Pipeline error] {e}")
            import traceback; traceback.print_exc()
        finally:
            self._processing = False

    # ------------------------------------------------------------------
    # Filler audio
    # ------------------------------------------------------------------

    async def _cache_filler(self):
        """Synthesize filler phrase once at call start and store as mulaw bytes."""
        pcm = await self.speech.synthesize("ठीक है जी")
        if pcm:
            self._filler_mulaw = audioop.lin2ulaw(pcm, 2)
            print("[Filler] Pre-cached")
        else:
            print("[Filler] WARNING: synthesis failed — filler will be silent")

    async def _play_filler(self):
        """Send pre-cached filler audio to Twilio. No-op if not cached yet."""
        if not self._filler_mulaw or not self.stream_sid:
            return
        filler_duration = len(self._filler_mulaw) / 8000.0
        self._muted_until = max(self._muted_until, time.monotonic() + filler_duration + 0.3)
        for i in range(0, len(self._filler_mulaw), TTS_CHUNK):
            try:
                await self._ws.send_text(json.dumps({
                    "event":     "media",
                    "streamSid": self.stream_sid,
                    "media":     {"payload": base64.b64encode(self._filler_mulaw[i:i + TTS_CHUNK]).decode()},
                }))
            except Exception:
                return

    # ------------------------------------------------------------------
    # TTS -> Twilio audio
    # ------------------------------------------------------------------

    async def _speak(self, text: str) -> float:
        pcm = await self.speech.synthesize(text)
        if not pcm:
            self._muted_until = 0.0   # TTS failed — let caller speak immediately
            return 0.0

        mulaw    = audioop.lin2ulaw(pcm, 2)
        duration = len(mulaw) / 8000.0
        # Safety fallback; mark event releases mute precisely before this expires
        self._muted_until = time.monotonic() + duration + 2.0

        for i in range(0, len(mulaw), TTS_CHUNK):
            try:
                await self._ws.send_text(json.dumps({
                    "event":     "media",
                    "streamSid": self.stream_sid,
                    "media":     {"payload": base64.b64encode(mulaw[i:i + TTS_CHUNK]).decode()},
                }))
            except Exception:
                self._muted_until = 0.0
                return 0.0

        # Mark event — Twilio echoes it back exactly when audio finishes on the phone
        try:
            await self._ws.send_text(json.dumps({
                "event":     "mark",
                "streamSid": self.stream_sid,
                "mark":      {"name": "tts_done"},
            }))
        except Exception:
            pass

        return duration
