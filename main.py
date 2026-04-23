"""
AI Voice Agent - Doctor Appointment Booking
Uses Twilio <Record> + Sarvam STT/TTS — no WebSocket, no RMS tuning.

Flow per turn:
  1. Play TTS audio via <Play>
  2. <Record> captures caller's response
  3. POST to /handle-recording → download WAV → Sarvam STT → LLM → Sarvam TTS → loop
  4. When done → <Play> confirmation + <Hangup>
"""

import os
import io
import uuid
import wave
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import Response, FileResponse
import httpx
import uvicorn
from dotenv import load_dotenv

from agent.conversation import ConversationManager, State
from services.speech import SpeechService

load_dotenv()

# Decode Google service account on Railway (file is gitignored)
_b64 = os.getenv("GOOGLE_SERVICE_ACCOUNT_B64")
if _b64:
    import base64 as _b64mod
    _creds_path = "/tmp/google-service-account.json"
    with open(_creds_path, "wb") as _f:
        _f.write(_b64mod.b64decode(_b64))
    os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"] = _creds_path

app = FastAPI(title="AI Voice Agent - Doctor Appointment Booking")
speech_service = SpeechService()

# In-memory sessions: call_sid → ConversationManager
sessions: dict = {}

# Serve generated TTS audio files
AUDIO_DIR = Path("/tmp/voice_audio")
AUDIO_DIR.mkdir(parents=True, exist_ok=True)


def get_base_url(request: Request) -> str:
    host = request.headers.get("host", "localhost:8000")
    scheme = "https" if ("railway.app" in host or "ngrok" in host) else "http"
    return f"{scheme}://{host}"


async def tts_to_file(text: str, call_sid: str, turn: str) -> str:
    """Generate TTS, save as WAV, return filename."""
    pcm_bytes = await speech_service.synthesize(text)
    filename = f"{call_sid}_{turn}.wav"
    filepath = AUDIO_DIR / filename
    wav_buf = io.BytesIO()
    with wave.open(wav_buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(8000)
        wf.writeframes(pcm_bytes)
    filepath.write_bytes(wav_buf.getvalue())
    return filename


def record_twiml(audio_url: str, action_url: str) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Play>{audio_url}</Play>
    <Record action="{action_url}" method="POST"
            timeout="5" finishOnKey="" playBeep="false"
            maxLength="30" trim="trim-silence"/>
</Response>"""


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "service": "AI Voice Agent"}


@app.get("/audio/{filename}")
async def serve_audio(filename: str):
    filepath = AUDIO_DIR / filename
    if not filepath.exists():
        return Response(status_code=404)
    return FileResponse(str(filepath), media_type="audio/wav")


@app.post("/incoming-call")
async def incoming_call(request: Request):
    form = await request.form()
    call_sid = form.get("CallSid", str(uuid.uuid4()))

    conversation = ConversationManager()
    sessions[call_sid] = conversation

    base_url = get_base_url(request)
    filename = await tts_to_file(conversation.get_greeting(), call_sid, "0")
    audio_url = f"{base_url}/audio/{filename}"
    action_url = f"{base_url}/handle-recording"

    print(f"[Call] Started: {call_sid}")
    return Response(content=record_twiml(audio_url, action_url),
                    media_type="application/xml")


@app.post("/handle-recording")
async def handle_recording(request: Request):
    form = await request.form()
    call_sid = form.get("CallSid", "")
    recording_url = form.get("RecordingUrl", "")

    base_url = get_base_url(request)
    action_url = f"{base_url}/handle-recording"

    conversation = sessions.get(call_sid)
    if not conversation:
        return Response(
            content='<?xml version="1.0"?><Response><Hangup/></Response>',
            media_type="application/xml")

    # Download recording WAV from Twilio (requires auth)
    twilio_sid = os.getenv("TWILIO_ACCOUNT_SID", "")
    twilio_token = os.getenv("TWILIO_AUTH_TOKEN", "")
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get(f"{recording_url}.wav",
                                 auth=(twilio_sid, twilio_token))
            r.raise_for_status()
            wav_bytes = r.content
    except Exception as e:
        print(f"[Recording Error] {e}")
        filename = await tts_to_file(
            "Maafi chahta hoon, kripya dobara bolein.", call_sid, "err")
        return Response(
            content=record_twiml(f"{base_url}/audio/{filename}", action_url),
            media_type="application/xml")

    # STT
    transcript = await speech_service.transcribe_wav_bytes(wav_bytes)
    print(f"[Caller] {transcript}")

    if not transcript.strip():
        filename = await tts_to_file(
            "Kripya apna jawab phir se bolein.", call_sid, "notrans")
        return Response(
            content=record_twiml(f"{base_url}/audio/{filename}", action_url),
            media_type="application/xml")

    # LLM → agent response
    response_text = await conversation.process_turn(transcript)
    print(f"[Agent] {response_text}")

    turn = uuid.uuid4().hex[:8]
    filename = await tts_to_file(response_text, call_sid, turn)
    audio_url = f"{base_url}/audio/{filename}"

    if conversation.state == State.DONE:
        sessions.pop(call_sid, None)
        twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Play>{audio_url}</Play>
    <Pause length="2"/>
    <Hangup/>
</Response>"""
    else:
        twiml = record_twiml(audio_url, action_url)

    return Response(content=twiml, media_type="application/xml")


@app.post("/call-status")
async def call_status(request: Request):
    form = await request.form()
    call_sid = form.get("CallSid")
    status = form.get("CallStatus")
    print(f"[Call Status] SID={call_sid}, Status={status}")
    if status in ("completed", "failed", "busy", "no-answer"):
        sessions.pop(call_sid, None)
    return Response(content="OK")


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)

    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
