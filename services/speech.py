"""
Speech services:
  STT → Groq Whisper large-v3  (free, clean audio from Twilio <Record>)
  TTS → Sarvam bulbul:v3       (natural Hindi voice)
"""

import asyncio
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
        self._tts_speaker = os.getenv("SARVAM_TTS_SPEAKER", "priya")
        self._tts_lang = os.getenv("SARVAM_LANGUAGE_CODE", "hi-IN")
        # Persistent client: reuses TCP connection across all TTS calls (~150ms saved each)
        self._http = httpx.AsyncClient(timeout=15)

    # ------------------------------------------------------------------
    # Speech-to-Text  (mulaw 8kHz → Hindi transcript via Sarvam Saaras v3)
    # ------------------------------------------------------------------

    async def transcribe_wav_bytes(self, wav_bytes: bytes) -> str:
        """
        Transcribe a WAV file (bytes) using Groq Whisper large-v3.
        Used for Twilio <Record> downloads — clean caller audio, no echo.
        """
        try:
            response = await self._groq.audio.transcriptions.create(
                file=("audio.wav", wav_bytes, "audio/wav"),
                model="whisper-large-v3-turbo",
                language="hi",
                prompt="Hindi. Names: Rahul, Rishav, Priya, Dheeraj, Sharma.",
                response_format="text",
                temperature=0.0,
            )
            transcript = str(response).strip()
            print(f"[STT] {transcript}")
            return transcript
        except Exception as e:
            print(f"[STT Error] {e}")
            return ""

    async def transcribe_mulaw(self, mulaw_audio: bytes) -> str:
        """
        Convert mulaw 8kHz → 16kHz WAV, transcribe with Sarvam Saaras v2 codemix.
        Handles Hindi + English + Punjabi natively. Indian names and cities preserved.
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

            response = await self._http.post(
                "https://api.sarvam.ai/speech-to-text",
                headers={"api-subscription-key": self._sarvam_key},
                files={"file": ("audio.wav", wav_buffer.read(), "audio/wav")},
                data={
                    "model": "saaras:v3",
                    "mode": "codemix",
                    "language_code": "unknown",
                },
            )
            response.raise_for_status()
            transcript = response.json().get("transcript", "").strip()

            print(f"[STT] {transcript}")
            return transcript

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
        Retries once on 429 (rate limit) with a 1s back-off.
        """
        payload = {
            "text": text,
            "target_language_code": self._tts_lang,
            "speaker": self._tts_speaker,
            "model": "bulbul:v3",
            "speech_sample_rate": 8000,
            "pace": 1.0,
        }
        headers = {
            "api-subscription-key": self._sarvam_key,
            "Content-Type": "application/json",
        }
        for attempt in range(2):
            try:
                response = await self._http.post(SARVAM_TTS_URL, headers=headers, json=payload)
                if response.status_code == 429:
                    print(f"[TTS] 429 rate limit — waiting 1s (attempt {attempt + 1})")
                    await asyncio.sleep(1.0)
                    continue
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
        print("[TTS Error] Failed after 2 attempts (429)")
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
