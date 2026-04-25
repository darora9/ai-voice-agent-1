"""
AI Voice Agent - Twilio bidirectional media stream handler.

Architecture: two concurrent tasks per call
  1. _twilio_loop   : read Twilio WS -> convert mulaw->PCM16k -> push to Azure PushAudioInputStream
  2. _pipeline_loop : asyncio.Queue -> ConversationManager -> TTS -> Twilio

Azure Speech SDK fires recognized callbacks from SDK threads.
loop.call_soon_threadsafe(queue.put_nowait, ...) is used — the only safe cross-thread dispatch.

STT: Azure Cognitive Services Speech — hi-IN / en-IN / pa-IN auto-detect, 300ms endpointing
TTS: Sarvam bulbul:v3 (persistent HTTP client in SpeechService)
"""

import asyncio
import audioop
import base64
import json
import os
import re
import time

from fastapi import WebSocket

TTS_CHUNK = 1600  # ~200ms of mulaw @ 8kHz per send

# Matches any Hindi (Devanagari), Punjabi (Gurmukhi), or Latin letter
_HAS_LETTER = re.compile(r'[a-zA-Z\u0900-\u097F\u0A00-\u0A7F]')


class StreamSession:
    def __init__(self, speech_service):
        self.speech         = speech_service
        self.call_sid       = None
        self.stream_sid     = None
        self.conversation   = None
        self._ws            = None
        self._push_stream   = None   # Azure PushAudioInputStream
        self._recognizer    = None   # Azure SpeechRecognizer
        self._loop          = None   # asyncio event loop — needed for thread-safe callbacks
        self._ratecv_state  = None   # stateful upsampler: 8kHz→16kHz (avoids chunk-boundary clicks)
        self._muted_until   = 0.0
        self._processing    = False
        self._transcript_q  = asyncio.Queue()
        self._filler_mulaw: bytes = b""  # pre-cached "ठीक है जी"

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    async def run(self, websocket: WebSocket):
        self._ws   = websocket
        self._loop = asyncio.get_event_loop()
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
            await self._close_azure()

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

        await self._open_azure()

        # Cache filler FIRST — avoids Sarvam 429 race with greeting TTS.
        # Both calls hit Sarvam sequentially, not concurrently.
        await self._cache_filler()

        # Hold mute until greeting finishes; mark event releases it precisely.
        self._muted_until = time.monotonic() + 30.0
        greeting = self.conversation.get_greeting()
        print(f"[Agent] {greeting}")
        asyncio.create_task(self._speak(greeting), name="speak_greeting")

    # ------------------------------------------------------------------
    # Twilio loop: push inbound audio to Azure; handle mark/stop
    # ------------------------------------------------------------------

    async def _twilio_loop(self, websocket: WebSocket):
        async for raw in websocket.iter_text():
            msg   = json.loads(raw)
            event = msg.get("event")

            if event == "media":
                if self._push_stream and msg["media"].get("track", "inbound") == "inbound":
                    mulaw = base64.b64decode(msg["media"]["payload"])
                    # mulaw 8kHz → PCM 16-bit 8kHz → upsample to 16kHz (Azure requires 16kHz)
                    pcm_8k = audioop.ulaw2lin(mulaw, 2)
                    pcm_16k, self._ratecv_state = audioop.ratecv(
                        pcm_8k, 2, 1, 8000, 16000, self._ratecv_state
                    )
                    try:
                        self._push_stream.write(pcm_16k)
                    except Exception:
                        pass

            elif event == "mark":
                # Twilio echoes this back exactly when audio finishes playing on the phone.
                if msg.get("mark", {}).get("name") == "tts_done":
                    self._muted_until = 0.0
                    print("[Audio] Playback complete — mute released")

            elif event == "stop":
                print(f"[Call] Stream stopped: {self.call_sid}")
                break

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
                await asyncio.sleep(1.5)
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
    # Azure Speech SDK connection
    # ------------------------------------------------------------------

    async def _open_azure(self):
        key    = os.environ.get("AZURE_SPEECH_KEY", "")
        region = os.environ.get("AZURE_SPEECH_REGION", "")
        if not key or not region:
            print("[Azure] ERROR: AZURE_SPEECH_KEY or AZURE_SPEECH_REGION not set")
            return
        try:
            import azure.cognitiveservices.speech as speechsdk

            speech_config = speechsdk.SpeechConfig(subscription=key, region=region)

            # 300ms silence triggers end-of-utterance (vs Deepgram's 500ms — 200ms faster)
            speech_config.set_property(
                speechsdk.PropertyId.Speech_SegmentationSilenceTimeoutMs, "300"
            )

            # Auto-detect: Hindi / English / Punjabi in a single stream
            auto_detect = speechsdk.languageconfig.AutoDetectSourceLanguageConfig(
                languages=["hi-IN", "en-IN", "pa-IN"]
            )

            # PCM 16kHz 16-bit mono — what we push after upsampling
            fmt = speechsdk.audio.AudioStreamFormat(
                samples_per_second=16000, bits_per_sample=16, channels=1
            )
            self._push_stream = speechsdk.audio.PushAudioInputStream(stream_format=fmt)
            audio_config = speechsdk.audio.AudioConfig(stream=self._push_stream)

            self._recognizer = speechsdk.SpeechRecognizer(
                speech_config=speech_config,
                audio_config=audio_config,
                auto_detect_source_language_config=auto_detect,
            )

            loop = self._loop

            def _on_recognized(evt):
                """Called from Azure SDK thread — dispatch to asyncio via call_soon_threadsafe."""
                try:
                    transcript = evt.result.text.strip()
                    if not transcript or not _HAS_LETTER.search(transcript):
                        return
                    now = time.monotonic()
                    if now < self._muted_until:
                        print(f"[STT] Muted ({self._muted_until - now:.1f}s left): {transcript}")
                        return
                    if self._processing:
                        print(f"[STT] Busy (discarded): {transcript}")
                        return
                    print(f"[STT] {transcript}")
                    loop.call_soon_threadsafe(self._transcript_q.put_nowait, transcript)
                except Exception as e:
                    print(f"[Azure transcript error] {e}")

            def _on_canceled(evt):
                details = evt.cancellation_details
                print(f"[Azure] Canceled: {details.reason} — {details.error_details}")

            self._recognizer.recognized.connect(_on_recognized)
            self._recognizer.canceled.connect(_on_canceled)
            self._recognizer.start_continuous_recognition()
            print("[Azure] STT connected — hi-IN / en-IN / pa-IN, 300ms endpoint")

        except Exception as e:
            print(f"[Azure] Connection failed: {e}")
            import traceback; traceback.print_exc()
            self._push_stream = None
            self._recognizer  = None

    async def _close_azure(self):
        if self._recognizer:
            try:
                self._recognizer.stop_continuous_recognition()
            except Exception:
                pass
            self._recognizer = None
        if self._push_stream:
            try:
                self._push_stream.close()
            except Exception:
                pass
            self._push_stream = None

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
