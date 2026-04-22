# AI Voice Agent — Doctor Appointment Booking (India)

Receives phone calls on an **Indian Twilio number**, converses with the patient in Hindi or English using GPT-4, and books the appointment in **Google Calendar**.

---

## Architecture

```
Caller (Indian Number)
     ↓ (PSTN)
  Twilio
     ↓ (TwiML / Media Stream WebSocket)
  FastAPI Server  ──→  OpenAI Whisper (STT)
                  ──→  GPT-4o (conversation)
                  ──→  OpenAI TTS (speech)
                  ──→  Google Calendar API (book slot)
```

---

## Prerequisites

- Python 3.10+
- [Twilio account](https://www.twilio.com/) with an Indian number (see note below)
- [OpenAI API key](https://platform.openai.com/)
- Google Cloud service account with Calendar API enabled
- [ngrok](https://ngrok.com/) (for local development) OR a deployed server with HTTPS

---

## 1. Install dependencies

```bash
pip install -r requirements.txt
```

> **Python 3.13+**: `audioop` was removed — `audioop-lts` is already in requirements.txt.  
> **Python 3.12 and below**: remove `audioop-lts` from requirements.txt.

---

## 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and fill in all values:

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | Your OpenAI API key |
| `TWILIO_ACCOUNT_SID` | Twilio Account SID |
| `TWILIO_AUTH_TOKEN` | Twilio Auth Token |
| `TWILIO_PHONE_NUMBER` | Your Twilio Indian number (`+91XXXXXXXXXX`) |
| `GOOGLE_SERVICE_ACCOUNT_JSON` | Path to Google service account JSON file |
| `GOOGLE_CALENDAR_ID` | Google Calendar ID to book into |
| `DOCTOR_NAME` | Doctor's name (shown in calendar events) |
| `CLINIC_NAME` | Clinic name (spoken to callers) |

---

## 3. Set up Google Calendar

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a project → enable **Google Calendar API**
3. Create a **Service Account** → download JSON key → save as `credentials/google-service-account.json`
4. Share your Google Calendar with the service account email (give it **"Make changes to events"** permission)
5. Copy the Calendar ID from Calendar settings into `GOOGLE_CALENDAR_ID` in `.env`

---

## 4. Get a Twilio Indian number

1. Sign up at [twilio.com](https://www.twilio.com/)
2. Buy a number → search for India (+91)
   - Twilio provides Indian numbers via their **Elastic SIP Trunking** or **Local Numbers** (availability varies; you may need to submit regulatory documents for Indian numbers)
   - Alternatively, use an **Exotel** or **Plivo** Indian number — both support Twilio-compatible webhooks
3. Set the **Voice webhook** URL to: `https://your-domain.com/incoming-call` (POST)
4. Set the **Status callback** URL to: `https://your-domain.com/call-status` (POST)

---

## 5. Run locally with ngrok

```bash
# Terminal 1 — start the server
python main.py

# Terminal 2 — expose via ngrok
ngrok http 8000
```

Copy the ngrok HTTPS URL (e.g. `https://abc123.ngrok.io`) and paste it as the webhook URL in Twilio.

---

## 6. Deploy to production

You can deploy to any HTTPS-enabled server:

```bash
# Example: fly.io
fly launch
fly deploy

# Example: Railway / Render / EC2 — just ensure port 8000 is exposed
```

Set `USE_WSS=true` in `.env` when deployed (forces `wss://` for media streams).

---

## Project Structure

```
ai voice/
├── main.py                        # FastAPI app + Twilio webhooks
├── agent/
│   ├── conversation.py            # GPT-4 conversation manager
│   └── prompts.py                 # System prompt + greeting
├── services/
│   ├── twilio_handler.py          # Media stream audio pipeline
│   ├── speech.py                  # Whisper STT + OpenAI TTS
│   └── calendar_service.py        # Google Calendar integration
├── credentials/
│   └── google-service-account.json  # (you create this)
├── requirements.txt
├── .env.example
└── README.md
```

---

## How it works

1. **Call arrives** → Twilio calls `/incoming-call` webhook
2. **TwiML response** → connects audio to `/media-stream` WebSocket
3. **Real-time audio** → mulaw 8kHz chunks stream in
4. **Silence detection** → detects end-of-speech
5. **Whisper STT** → transcribes caller audio (Hindi/English)
6. **GPT-4o** → continues conversation, collects: name, date, time, reason
7. **Booking** → once all details collected, GPT-4o emits a JSON block
8. **Google Calendar** → event created with patient details + IST timezone
9. **TTS** → confirmation spoken back to caller

---

## Indian Number Alternatives

If Twilio doesn't issue a direct Indian number immediately, these providers work well and support webhook-based voice:

| Provider | Notes |
|---|---|
| **Exotel** | Popular in India, webhook support, easy KYC |
| **Plivo** | Indian numbers, Twilio-compatible API |
| **Knowlarity** | Enterprise, India-focused |
| **MCUBE** | India-specific virtual numbers |

All can be configured to POST to your `/incoming-call` endpoint.
