"""
Voice pipeline tester (microphone → STT → GPT → TTS → speaker).
Tests the full audio pipeline locally without any phone or hosting.

Requirements (additional):
    pip install sounddevice soundfile numpy

Usage:
    python test_voice.py
"""

import asyncio
import os
import io
import wave
import tempfile
import audioop
from dotenv import load_dotenv

load_dotenv()

try:
    import sounddevice as sd
    import soundfile as sf
    import numpy as np
except ImportError:
    print("Install audio deps first:  pip install sounddevice soundfile numpy")
    raise

from agent.conversation import ConversationManager
from services.speech import SpeechService

SAMPLE_RATE = 16000   # Record at 16kHz
CHANNELS = 1
RECORD_SECONDS = 5    # Press Enter to stop early (see below)


def record_audio(seconds: int = RECORD_SECONDS) -> bytes:
    """Record from default microphone. Returns raw PCM bytes (16kHz, 16-bit mono)."""
    print(f"  [Recording for up to {seconds}s — speak now...]")
    audio = sd.rec(
        int(seconds * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="int16",
    )
    sd.wait()
    print("  [Recording done]")
    return audio.tobytes()


def pcm16k_to_mulaw8k(pcm_16k: bytes) -> bytes:
    """Downsample 16kHz PCM → 8kHz mulaw (what our SpeechService expects)."""
    pcm_8k, _ = audioop.ratecv(pcm_16k, 2, 1, 16000, 8000, None)
    return audioop.lin2ulaw(pcm_8k, 2)


def play_pcm(pcm_bytes: bytes, sample_rate: int = 8000):
    """Play raw PCM (16-bit mono) through speakers."""
    audio_np = np.frombuffer(pcm_bytes, dtype=np.int16)
    sd.play(audio_np, samplerate=sample_rate)
    sd.wait()


async def main():
    print("=" * 55)
    print("  AI Voice Agent — Voice Test (no phone needed)")
    print("  Microphone → STT → GPT → TTS → Speaker")
    print("=" * 55)

    agent = ConversationManager()
    speech = SpeechService()

    # Speak greeting
    greeting = agent.get_greeting()
    print(f"\n[Agent]: {greeting}")
    greeting_audio = await speech.synthesize(greeting)
    if greeting_audio:
        play_pcm(greeting_audio, sample_rate=8000)

    while True:
        input("\nPress Enter to speak (or Ctrl+C to quit)...")

        # Record caller audio
        pcm_16k = record_audio()

        # Convert to mulaw 8kHz (what our pipeline expects from Vobiz)
        mulaw_audio = pcm16k_to_mulaw8k(pcm_16k)

        # STT
        print("  [Transcribing...]")
        transcript = await speech.transcribe_mulaw(mulaw_audio)
        if not transcript:
            print("  [No speech detected, try again]")
            continue
        print(f"[You]: {transcript}")

        # GPT
        print("  [Thinking...]")
        response = await agent.process_turn(transcript)
        print(f"[Agent]: {response}")

        # TTS + play
        print("  [Speaking...]")
        audio = await speech.synthesize(response)
        if audio:
            play_pcm(audio, sample_rate=8000)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nTest ended.")
