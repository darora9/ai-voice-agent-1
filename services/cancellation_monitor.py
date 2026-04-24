"""
Cancellation monitor — polls Google Calendar every 5 minutes for deleted appointments.
When a booked appointment is cancelled, sends an SMS to the patient with:
  - Apology message
  - Clinic phone number to rebook
  - Next available slot
"""

import asyncio
import os
import httpx

POLL_INTERVAL = 300  # 5 minutes

_CLINIC_NAME    = os.getenv("CLINIC_NAME", "Clinic")
_DOCTOR_NAME    = os.getenv("DOCTOR_NAME", "Doctor")
_CLINIC_PHONE   = os.getenv("CLINIC_PHONE_NUMBER", "")   # e.g. +15709899044 or local number
_TWILIO_FROM    = os.getenv("TWILIO_PHONE_NUMBER", "")
_ACCOUNT_SID    = os.getenv("TWILIO_ACCOUNT_SID", "")
_AUTH_TOKEN     = os.getenv("TWILIO_AUTH_TOKEN", "")


def _human_date(date_str: str, time_str: str) -> tuple[str, str]:
    """Return (human_date, time) strings for SMS."""
    from datetime import date as _date
    try:
        d = _date.fromisoformat(date_str)
        days   = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        months = ["January","February","March","April","May","June",
                  "July","August","September","October","November","December"]
        return f"{d.day} {months[d.month-1]} ({days[d.weekday()]})", time_str
    except Exception:
        return date_str, time_str


async def _send_sms(to: str, body: str) -> None:
    if not all([_ACCOUNT_SID, _AUTH_TOKEN, _TWILIO_FROM, to]):
        print(f"[CancelMonitor] SMS skipped — missing config or no phone for {to}")
        return
    url = f"https://api.twilio.com/2010-04-01/Accounts/{_ACCOUNT_SID}/Messages.json"
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                url,
                auth=(_ACCOUNT_SID, _AUTH_TOKEN),
                data={"From": _TWILIO_FROM, "To": to, "Body": body},
                timeout=10,
            )
        if resp.status_code == 201:
            print(f"[CancelMonitor] SMS sent to {to}")
        else:
            print(f"[CancelMonitor] SMS failed ({resp.status_code}): {resp.text[:200]}")
    except Exception as e:
        print(f"[CancelMonitor] SMS error: {e}")


async def run_monitor(calendar_service) -> None:
    """
    Background coroutine. Pass the shared CalendarService instance from main.py.
    Bootstraps syncToken on first run, then polls incrementally.
    """
    sync_token: str | None = None

    # Bootstrap — get initial syncToken without sending any SMS
    print("[CancelMonitor] Bootstrapping syncToken...")
    _, sync_token = calendar_service.get_cancelled_since(None)
    print(f"[CancelMonitor] Ready. Polling every {POLL_INTERVAL}s.")

    while True:
        await asyncio.sleep(POLL_INTERVAL)
        try:
            cancelled, sync_token = calendar_service.get_cancelled_since(sync_token)
            for ev in cancelled:
                phone     = ev.get("phone", "")
                name      = ev.get("name", "Patient")
                date_str  = ev.get("date", "")
                time_str  = ev.get("time", "")

                if not phone:
                    print(f"[CancelMonitor] No phone for cancelled event, skipping")
                    continue

                cancelled_date, cancelled_time = _human_date(date_str, time_str)

                # Find next available slot
                next_slot = calendar_service.get_next_available_slot()
                if next_slot:
                    next_date, next_time = _human_date(next_slot["date"], next_slot["time"])
                    next_slot_text = f"Next available slot: {next_date} at {next_time}."
                else:
                    next_slot_text = "Please call us to check available slots."

                clinic_contact = f" Call {_CLINIC_PHONE} to book." if _CLINIC_PHONE else ""

                body = (
                    f"Dear {name}, we regret to inform you that due to unforeseen reasons, "
                    f"{_DOCTOR_NAME} will not be available for your appointment on "
                    f"{cancelled_date} at {cancelled_time}. "
                    f"We apologise for the inconvenience.{clinic_contact} "
                    f"{next_slot_text} — {_CLINIC_NAME}"
                )

                await _send_sms(phone, body)

        except Exception as e:
            print(f"[CancelMonitor] Poll error: {e}")
