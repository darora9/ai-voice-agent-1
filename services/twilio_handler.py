"""
Twilio Media Stream handler.
Processes real-time audio from caller, runs STT → GPT-4 → TTS pipeline.
"""

import asyncio
import base64
import json
import audioop
from typing import Optional
from fastapi import WebSocket

from agent.conversation import ConversationManager
from services.speech import SpeechService


class TwilioMediaHandler:
    def __init__(
        self,
        websocket: WebSocket,
        conversation: ConversationManager,
        speech_service: SpeechService,
    ):
        self.websocket = websocket
        self.conversation = conversation
        self.speech = speech_service

        self.stream_sid: Optional[str] = None
        self.call_sid: Optional[str] = None

        # Audio buffer: Twilio sends mulaw 8kHz; accumulate chunks before STT
        self._audio_buffer = bytearray()
        self._silence_threshold = 500        # RMS energy below = silence
        self._min_speech_bytes = 16000 * 2   # ~2 seconds of audio before processing
        self._silent_chunks = 0
        self._silent_chunks_threshold = 20   # ~2s silence → trigger STT

        self._is_agent_speaking = False
        self._processing_lock = asyncio.Lock()

    async def run(self):
        """Main loop: receive Twilio events and process audio."""
        async for message in self._receive_messages():
            await self._handle_message(message)

    async def _receive_messages(self):
        while True:
            try:
                raw = await self.websocket.receive_text()
                yield json.loads(raw)
            except Exception:
                break

    async def _handle_message(self, msg: dict):
        event = msg.get("event")

        if event == "start":
            self.stream_sid = msg["start"]["streamSid"]
            self.call_sid = msg["start"]["callSid"]
            print(f"[Stream] Started: {self.stream_sid}")
            # Greet the caller
            await self._speak(self.conversation.get_greeting())

        elif event == "media":
            if self._is_agent_speaking:
                return  # Ignore caller audio while agent is speaking
            payload = msg["media"]["payload"]
            mulaw_chunk = base64.b64decode(payload)
            self._audio_buffer.extend(mulaw_chunk)

            # Check for end-of-speech using silence detection
            await self._check_end_of_speech(mulaw_chunk)

        elif event == "stop":
            print("[Stream] Stopped")

    async def _check_end_of_speech(self, chunk: bytes):
        """Detect silence to know when caller has finished speaking."""
        try:
            pcm = audioop.ulaw2lin(chunk, 2)
            rms = audioop.rms(pcm, 2)
        except Exception:
            rms = 0

        if rms < self._silence_threshold:
            self._silent_chunks += 1
        else:
            self._silent_chunks = 0  # Reset on speech

        # Trigger STT when: enough audio collected + silence detected
        if (
            len(self._audio_buffer) >= self._min_speech_bytes
            and self._silent_chunks >= self._silent_chunks_threshold
        ):
            await self._process_speech()

    async def _process_speech(self):
        """Run STT → GPT-4 → TTS pipeline."""
        async with self._processing_lock:
            audio_data = bytes(self._audio_buffer)
            self._audio_buffer.clear()
            self._silent_chunks = 0

            if not audio_data:
                return

            # Speech-to-Text
            transcript = await self.speech.transcribe_mulaw(audio_data)
            if not transcript or not transcript.strip():
                return

            print(f"[Caller] {transcript}")

            # GPT-4 response
            response_text = await self.conversation.process_turn(transcript)
            print(f"[Agent] {response_text}")

            if response_text:
                await self._speak(response_text)

    async def _speak(self, text: str):
        """Convert text to speech and stream back to Twilio."""
        self._is_agent_speaking = True
        try:
            audio_bytes = await self.speech.synthesize(text)
            # Encode as mulaw 8kHz for Twilio
            mulaw_audio = self.speech.pcm_to_mulaw(audio_bytes)
            await self._send_audio(mulaw_audio)
        finally:
            self._is_agent_speaking = False

    async def _send_audio(self, mulaw_audio: bytes):
        """Send audio chunks to Twilio via media stream."""
        chunk_size = 160  # 20ms at 8kHz
        for i in range(0, len(mulaw_audio), chunk_size):
            chunk = mulaw_audio[i : i + chunk_size]
            payload = base64.b64encode(chunk).decode("utf-8")
            await self.websocket.send_text(
                json.dumps(
                    {
                        "event": "media",
                        "streamSid": self.stream_sid,
                        "media": {"payload": payload},
                    }
                )
            )
            await asyncio.sleep(0.02)  # 20ms pacing

    async def cleanup(self):
        self._audio_buffer.clear()
