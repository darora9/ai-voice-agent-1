"""
AI Voice Agent - Doctor Appointment Booking

Twilio Media Streams (bidirectional WebSocket) for real-time low-latency conversation.

Flow:
  1. POST /incoming-call  -> TwiML <Connect><Stream url="wss://host/stream"/>
  2. Twilio opens WebSocket /stream -- pipes caller audio in real-time (mulaw 8kHz)
  3. VAD detects end of speech -> Groq Whisper -> state machine -> Sarvam TTS
  4. TTS mulaw sent back through the same WebSocket -- Twilio plays it to caller
  5. When state == DONE, WebSocket closes -> <Hangup> fires
"""

import os
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import Response
import uvicorn
from dotenv import load_dotenv

from services.speech import SpeechService
from services.twilio_handler import StreamSession

load_dotenv()

# Decode Google service account credentials (stored as base64 in Railway env var)
_b64 = os.getenv("GOOGLE_SERVICE_ACCOUNT_B64")
if _b64:
    import base64 as _b64mod
    _creds_path = "/tmp/google-service-account.json"
    with open(_creds_path, "wb") as _f:
        _f.write(_b64mod.b64decode(_b64))
    os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"] = _creds_path

app = FastAPI(title="AI Voice Agent")
speech_service = SpeechService()


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/incoming-call")
async def incoming_call(request: Request):
    host = request.headers.get("host", "localhost:8000")
    stream_url = f"wss://{host}/stream"
    print(f"[Call] Incoming -> stream at {stream_url}")

    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <Stream url="{stream_url}"/>
    </Connect>
    <Hangup/>
</Response>"""
    return Response(content=twiml, media_type="application/xml")


@app.websocket("/stream")
async def stream_ws(websocket: WebSocket):
    session = StreamSession(speech_service)
    await session.run(websocket)


@app.post("/call-status")
async def call_status(request: Request):
    form = await request.form()
    print(f"[Call Status] SID={form.get('CallSid')}, Status={form.get('CallStatus')}")
    return Response(content="OK")


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
