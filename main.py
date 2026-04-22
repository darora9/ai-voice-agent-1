"""
AI Voice Agent - Doctor Appointment Booking
Receives calls via Exotel (Indian number), books appointments, schedules in Google Calendar
"""

import os
import json
import base64
import asyncio
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
import uvicorn
from dotenv import load_dotenv

from agent.conversation import ConversationManager
from services.twilio_handler import TwilioMediaHandler
from services.speech import SpeechService

load_dotenv()

# On Railway, credentials are stored as base64 env var (file is gitignored)
_b64 = os.getenv("GOOGLE_SERVICE_ACCOUNT_B64")
if _b64:
    import base64 as _b64mod
    _creds_path = "/tmp/google-service-account.json"
    with open(_creds_path, "wb") as _f:
        _f.write(_b64mod.b64decode(_b64))
    os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"] = _creds_path

app = FastAPI(title="AI Voice Agent - Doctor Appointment Booking")

speech_service = SpeechService()


@app.get("/health")
async def health():
    return {"status": "ok", "service": "AI Voice Agent"}


@app.post("/incoming-call")
async def incoming_call(request: Request):
    """
    Exotel passthru webhook - called when someone dials the virtual number.
    Returns TwiML to connect to media stream WebSocket.
    """
    host = request.headers.get("host")
    ws_scheme = "wss" if "ngrok" in host or os.getenv("USE_WSS", "false") == "true" else "ws"

    # Vobiz XML — same structure as TwiML; <Speak> replaces <Say>
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Speak language="hi-IN">
        Namaste! Aap doctor ke appointment booking service mein aaye hain. Please hold on.
    </Speak>
    <Connect>
        <Stream url="{ws_scheme}://{host}/media-stream" />
    </Connect>
</Response>"""

    return Response(content=twiml, media_type="application/xml")


@app.post("/call-status")
async def call_status(request: Request):
    """Exotel call status callback."""
    form = await request.form()
    call_sid = form.get("CallSid")
    status = form.get("CallStatus")
    print(f"[Call Status] SID={call_sid}, Status={status}")
    return Response(content="OK")


@app.websocket("/media-stream")
async def media_stream(websocket: WebSocket):
    """
    WebSocket endpoint for Exotel Media Streams.
    Handles real-time bidirectional audio between caller and AI agent.
    """
    await websocket.accept()
    print("[WebSocket] New media stream connection")

    conversation = ConversationManager()
    handler = TwilioMediaHandler(websocket, conversation, speech_service)

    try:
        await handler.run()
    except WebSocketDisconnect:
        print("[WebSocket] Client disconnected")
    except Exception as e:
        print(f"[WebSocket] Error: {e}")
    finally:
        await handler.cleanup()


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
