"""
Real-time bidirectional WebSocket media stream handler for Twilio <Connect><Stream>.

Receives inbound caller audio (mulaw 8kHz, 20ms chunks).
Runs energy-based VAD to detect end of speech (~400ms silence).
Sends TTS audio back through the same WebSocket -- no HTTP round-trips.

Latency per turn: ~2-3 s   (vs 10-12 s with <Record> approach)
"""

import asyncio
import audioop
import base64
import json
import time

from fastapi import WebSocket

# ---------------------------------------------------------------------------
# VAD tuning
# ---------------------------------------------------------------------------
SPEECH_RMS  = 180   # RMS above this = active speech
SILENCE_END = 16    # consecutive silent 20ms frames -> end of utterance (~320ms)
MIN_SPEECH  = 5     # minimum speech frames to process (~100ms)
TTS_CHUNK   = 3200  # bytes per WebSocket write (~200ms of mulaw @ 8kHz)


class StreamSession:
    """Manages one live call over a Twilio bidirectional media stream."""

    def __init__(self, speech_service):
        self.speech = speech_service
        self.call_sid    = None
        self.stream_sid  = None
        self.conversation = None
        self._ws: WebSocket | None = None

        # VAD state
        self._buf        = bytearray()
        self._has_speech = False
        self._sil        = 0
        self._sframes    = 0

        # Mute inbound audio while TTS is playing (prevents echo re-processing)
        self._muted_until = 0.0

        # Only one STT->LLM->TTS pipeline at a time
        self._processing = False

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    async def run(self, websocket: WebSocket):
        self._ws = websocket
        await websocket.accept()
        try:
            async for raw in websocket.iter_text():
                await self._on_message(json.loads(raw))
        except Exception as e:
            print(f"[Stream Error] {self.call_sid}: {e}")

    # ------------------------------------------------------------------
    # Message dispatch
    # ------------------------------------------------------------------

    async def _on_message(self, msg: dict):
        event = msg.get("event")

        if event == "start":
            from agent.conversation import ConversationManager
            self.call_sid   = msg["start"]["callSid"]
            self.stream_sid = msg["start"]["streamSid"]
            self.conversation = ConversationManager()
            print(f"[Call] Stream started: {self.call_sid}")
            asyncio.create_task(self._speak(self.conversation.get_greeting()))

        elif event == "media":
            if not self.conversation:
                return
            if msg["media"].get("track", "inbound") != "inbound":
                return
            self._vad(base64.b64decode(msg["media"]["payload"]))

        elif event == "stop":
            print(f"[Call] Stream stopped: {self.call_sid}")

    # ------------------------------------------------------------------
    # VAD -- energy-based end-of-speech detection
    # ------------------------------------------------------------------

    def _vad(self, chunk: bytes):
        if time.monotonic() < self._muted_until:
            return  # ignore inbound during TTS playback

        rms = audioop.rms(audioop.ulaw2lin(chunk, 2), 2)

        if rms > SPEECH_RMS:
            self._has_speech = True
            self._sil = 0
            self._sframes += 1
            self._buf.extend(chunk)
        elif self._has_speech:
            self._sil += 1
            self._buf.extend(chunk)
            if self._sil >= SILENCE_END:
                if self._sframes >= MIN_SPEECH and not self._processing:
                    audio = bytes(self._buf)
                    self._reset_vad()
                    asyncio.create_task(self._pipeline(audio))
                else:
                    self._reset_vad()

    def _reset_vad(self):
        self._buf        = bytearray()
        self._has_speech = False
        self._sil        = 0
        self._sframes    = 0

    # ------------------------------------------------------------------
    # STT -> state machine -> TTS pipeline
    # ------------------------------------------------------------------

    async def _pipeline(self, mulaw_audio: bytes):
        self._processing = True
        try:
            # Sarvam Saaras v3 codemix -- purpose-built for Hindi+English names
            transcript = await self.speech.transcribe_mulaw(mulaw_audio)
            print(f"[Caller] {transcript}")

            t = transcript.strip()
            # Discard empty / single-char / punctuation-only transcripts
            import re as _re
            if not t or not _re.search(r'[a-zA-Z\u0900-\u097F]', t):
                return

            from agent.conversation import State
            response = await self.conversation.process_turn(transcript)
            if not response:
                return
            print(f"[Agent] {response}")

            await self._speak(response)

            if self.conversation.state == State.DONE:
                # _speak already waited for audio to finish; just close cleanly
                try:
                    await self._ws.close()
                except Exception:
                    pass  # Twilio may have already closed the socket
        finally:
            self._processing = False

    # ------------------------------------------------------------------
    # TTS -> WebSocket audio injection
    # ------------------------------------------------------------------

    async def _speak(self, text: str):
        pcm = await self.speech.synthesize(text)
        if not pcm:
            return

        mulaw = audioop.lin2ulaw(pcm, 2)
        duration = len(mulaw) / 8000.0
        # Mute inbound for the full TTS duration + 500ms buffer
        self._muted_until = time.monotonic() + duration + 0.5

        for i in range(0, len(mulaw), TTS_CHUNK):
            await self._ws.send_text(json.dumps({
                "event": "media",
                "streamSid": self.stream_sid,
                "media": {"payload": base64.b64encode(mulaw[i:i + TTS_CHUNK]).decode()},
            }))

        # Wait for Twilio to finish playing before returning
        await asyncio.sleep(duration)
