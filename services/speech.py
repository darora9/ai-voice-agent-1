"""
Speech services:
  STT → Groq Whisper large-v3  (free tier, excellent Hindi support)
  TTS → Sarvam bulbul:v3        (natural Hindi voice)
Handles mulaw ↔ PCM conversion for Vobiz media streams.
"""

import os
import io
import base64
import audioop
import wave
import httpx
from groq import AsyncGroq


SARVAM_TTS_URL = "https://api.sarvam.ai/text-to-speech"


class SpeechService:
    def __init__(self):
        self._groq = AsyncGroq(api_key=os.environ["GROQ_API_KEY"])
        self._sarvam_key = os.environ["SARVAM_API_KEY"]
        self._tts_speaker = os.getenv("SARVAM_TTS_SPEAKER", "priya")  # Hindi female voice
        self._tts_lang = os.getenv("SARVAM_LANGUAGE_CODE", "hi-IN")

    # ------------------------------------------------------------------
    # Speech-to-Text  (mulaw 8kHz → Hindi transcript via Groq Whisper)
    # ------------------------------------------------------------------

    async def transcribe_mulaw(self, mulaw_audio: bytes) -> str:
        """
        Convert mulaw 8kHz → 16kHz WAV, transcribe with Groq whisper-large-v3.
        language='hi' forces Hindi; Whisper still handles Hinglish fine.
        Returns the transcript string.
        """
        try:
            # mulaw → 16-bit PCM, upsample 8kHz → 16kHz
            pcm_audio = audioop.ulaw2lin(mulaw_audio, 2)
            pcm_16k, _ = audioop.ratecv(pcm_audio, 2, 1, 8000, 16000, None)

            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(pcm_16k)
            wav_buffer.seek(0)
            wav_buffer.name = "audio.wav"

            response = await self._groq.audio.transcriptions.create(
                file=("audio.wav", wav_buffer.read(), "audio/wav"),
                model="whisper-large-v3",
                language="hi",        # Hindi — best accuracy for Indian callers
                response_format="text",
                temperature=0.0,
            )
            return str(response).strip()

        except Exception as e:
            print(f"[STT Error] {e}")
            return ""


    # ------------------------------------------------------------------
    # Text-to-Speech  (text → PCM bytes at 8kHz)
    # ------------------------------------------------------------------

    async def synthesize(self, text: str) -> bytes:
        """
        POST to Sarvam TTS (bulbul:v3) for natural Hindi speech.
        Requests 8kHz output directly to skip resampling.
        Returns raw PCM bytes (8kHz, 16-bit mono).
        """
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                response = await client.post(
                    SARVAM_TTS_URL,
                    headers={
                        "api-subscription-key": self._sarvam_key,
                        "Content-Type": "application/json",
                    },
                    json={
                        "text": text,
                        "target_language_code": self._tts_lang,
                        "speaker": self._tts_speaker,
                        "model": "bulbul:v3",
                        "speech_sample_rate": 8000,
                        "pace": 0.9,   # slightly slower — easier to understand on phone
                    },
                )
                response.raise_for_status()
                audios = response.json().get("audios", [])
                if not audios:
                    return b""

                # Response is base64-encoded WAV — decode and extract raw PCM
                wav_bytes = base64.b64decode(audios[0])
                return self._extract_pcm_from_wav(wav_bytes)

        except Exception as e:
            print(f"[TTS Error] {e}")
            return b""

    # ------------------------------------------------------------------
    # Audio format helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_pcm_from_wav(wav_bytes: bytes) -> bytes:
        """Read raw PCM frames out of a WAV byte string."""
        buf = io.BytesIO(wav_bytes)
        with wave.open(buf, "rb") as wf:
            return wf.readframes(wf.getnframes())

    @staticmethod
    def pcm_to_mulaw(pcm_bytes: bytes, input_rate: int = 8000) -> bytes:
        """
        Convert PCM (8kHz, 16-bit mono) to mulaw 8kHz for Vobiz/Twilio.
        Since Sarvam already outputs 8kHz, no resampling needed by default.
        """
        if input_rate != 8000:
            pcm_bytes, _ = audioop.ratecv(pcm_bytes, 2, 1, input_rate, 8000, None)
        return audioop.lin2ulaw(pcm_bytes, 2)
