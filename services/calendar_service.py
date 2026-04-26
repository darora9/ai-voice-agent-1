"""
Google Calendar service for scheduling doctor appointments.
"""

import os
import json
from datetime import datetime, timedelta
from typing import Optional
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


SCOPES = ["https://www.googleapis.com/auth/calendar"]

# In-memory registry: event_id -> {phone, name, date, time}
# Populated on booking, used by cancellation monitor to look up who to SMS
_event_registry: dict[str, dict] = {}


class CalendarService:
    def __init__(self):
        creds_path = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")
        self.calendar_id = os.environ.get("GOOGLE_CALENDAR_ID", "primary")
        self.doctor_name = os.environ.get("DOCTOR_NAME", "Doctor")
        self.clinic_name = os.environ.get("CLINIC_NAME", "Clinic")
        self.appointment_duration = int(os.environ.get("APPOINTMENT_DURATION_MINS", "30"))

        credentials = service_account.Credentials.from_service_account_file(
            creds_path, scopes=SCOPES
        )
        self.service = build("calendar", "v3", credentials=credentials)

    def book_appointment(
        self,
        patient_name: str,
        patient_phone: str,
        date_str: str,   # "YYYY-MM-DD"
        time_str: str,   # "HH:MM" 24h
        reason: Optional[str] = None,
        city: Optional[str] = None,
    ) -> dict:
        """
        Create a calendar event for the appointment.
        Returns the created event dict on success.
        """
        try:
            start_dt = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
            end_dt = start_dt + timedelta(minutes=self.appointment_duration)

            # IST is UTC+5:30
            tz = "Asia/Kolkata"

            description_parts = [
                f"Patient: {patient_name}",
                f"Phone: {patient_phone}",
            ]
            if city:
                description_parts.append(f"City: {city}")
            if reason:
                description_parts.append(f"Reason: {reason}")

            event = {
                "summary": f"Appointment - {patient_name}",
                "description": "\n".join(description_parts),
                "location": self.clinic_name,
                "start": {
                    "dateTime": start_dt.strftime("%Y-%m-%dT%H:%M:00"),
                    "timeZone": tz,
                },
                "end": {
                    "dateTime": end_dt.strftime("%Y-%m-%dT%H:%M:00"),
                    "timeZone": tz,
                },
                "reminders": {
                    "useDefault": False,
                    "overrides": [
                        {"method": "popup", "minutes": 60},
                        {"method": "popup", "minutes": 15},
                    ],
                },
            }

            created = (
                self.service.events()
                .insert(calendarId=self.calendar_id, body=event)
                .execute()
            )
            print(f"[Calendar] Event created: {created.get('htmlLink')}")
            # Register for cancellation monitoring
            _event_registry[created["id"]] = {
                "phone": patient_phone,
                "name":  patient_name,
                "date":  date_str,
                "time":  time_str,
            }
            return {"success": True, "event": created}

        except HttpError as e:
            print(f"[Calendar Error] {e}")
            return {"success": False, "error": str(e)}
        except Exception as e:
            print(f"[Calendar Error] {e}")
            return {"success": False, "error": str(e)}

    def get_available_slots(self, date_str: str) -> list[str]:
        """
        Return list of available HH:MM slots on a given date.
        Checks existing events and excludes booked times.
        Returns None on API error, [] if genuinely no slots.
        """
        try:
            # Clinic is Mon-Sat only; Sunday has no slots
            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
            if date_obj.weekday() == 6:  # 6 = Sunday
                return "SUNDAY"  # special sentinel so caller gets a clear message
            tz = "Asia/Kolkata"
            start_of_day = datetime.strptime(f"{date_str} 09:00", "%Y-%m-%d %H:%M")
            end_of_day = datetime.strptime(f"{date_str} 18:00", "%Y-%m-%d %H:%M")

            events_result = (
                self.service.events()
                .list(
                    calendarId=self.calendar_id,
                    timeMin=start_of_day.strftime("%Y-%m-%dT%H:%M:00+05:30"),
                    timeMax=end_of_day.strftime("%Y-%m-%dT%H:%M:00+05:30"),
                    singleEvents=True,
                    orderBy="startTime",
                )
                .execute()
            )

            booked_ranges = []  # list of (start_dt, end_dt) for overlap check
            for event in events_result.get("items", []):
                ev_start = event["start"].get("dateTime") or event["start"].get("date")
                ev_end   = event["end"].get("dateTime")   or event["end"].get("date")
                if not ev_start or not ev_end:
                    continue
                try:
                    # Handle both date-only (all-day) and dateTime events
                    if "T" in ev_start:
                        s = datetime.fromisoformat(ev_start).replace(tzinfo=None)
                        e = datetime.fromisoformat(ev_end).replace(tzinfo=None)
                    else:
                        s = datetime.strptime(ev_start, "%Y-%m-%d").replace(hour=0, minute=0)
                        e = datetime.strptime(ev_end,   "%Y-%m-%d").replace(hour=23, minute=59)
                    booked_ranges.append((s, e))
                except Exception:
                    pass

            def _is_blocked(slot_dt: datetime) -> bool:
                slot_end = slot_dt + timedelta(minutes=self.appointment_duration)
                for s, e in booked_ranges:
                    # Overlap: slot starts before event ends AND slot ends after event starts
                    if slot_dt < e and slot_end > s:
                        return True
                return False

            # Generate all slots (every 30 mins, 9am–6pm)
            all_slots = []
            current = start_of_day
            while current < end_of_day:
                slot = current.strftime("%H:%M")
                if not _is_blocked(current):
                    all_slots.append(slot)
                current += timedelta(minutes=self.appointment_duration)

            return all_slots

        except Exception as e:
            print(f"[Calendar Error] {e}")
            return None  # None = error, [] = genuinely no slots

    def get_next_available_slot(self) -> dict | None:
        """Return {date, time} of the earliest available slot from today, or None."""
        import datetime as _dt
        IST = _dt.timezone(_dt.timedelta(hours=5, minutes=30))
        today = _dt.datetime.now(tz=IST).date()
        for day_offset in range(30):  # look up to 30 days ahead
            check_date = today + _dt.timedelta(days=day_offset)
            if check_date.weekday() == 6:  # skip Sunday
                continue
            slots = self.get_available_slots(check_date.isoformat())
            if not slots or slots == "SUNDAY":
                continue
            # Filter past slots if today
            from datetime import datetime as _dtt
            now_ist = _dtt.now(tz=IST)
            for s in slots:
                slot_dt = _dtt.strptime(
                    f"{check_date.isoformat()} {s}", "%Y-%m-%d %H:%M"
                ).replace(tzinfo=IST)
                if slot_dt > now_ist:
                    return {"date": check_date.isoformat(), "time": s}
        return None

    def get_next_available_after(self, from_date_str: str) -> dict | None:
        """Return {date, time} of the earliest available slot strictly after from_date_str, or None."""
        import datetime as _dt
        IST = _dt.timezone(_dt.timedelta(hours=5, minutes=30))
        try:
            start = _dt.date.fromisoformat(from_date_str) + _dt.timedelta(days=1)
        except Exception:
            start = _dt.datetime.now(tz=IST).date()
        now_ist = _dt.datetime.now(tz=IST)
        from datetime import datetime as _dtt
        for day_offset in range(30):
            check_date = start + _dt.timedelta(days=day_offset)
            if check_date.weekday() == 6:  # skip Sunday
                continue
            slots = self.get_available_slots(check_date.isoformat())
            if not slots or slots == "SUNDAY":
                continue
            for s in slots:
                slot_dt = _dtt.strptime(
                    f"{check_date.isoformat()} {s}", "%Y-%m-%d %H:%M"
                ).replace(tzinfo=IST)
                if slot_dt > now_ist:
                    return {"date": check_date.isoformat(), "time": s}
        return None

    def get_cancelled_since(self, sync_token: str | None) -> tuple[list[dict], str]:
        """
        Incremental sync using Google's syncToken.
        Returns (cancelled_events, new_sync_token).
        Each cancelled event is a dict from _event_registry.
        First call (sync_token=None) bootstraps the token without processing history.
        """
        try:
            params = {"calendarId": self.calendar_id, "singleEvents": True}
            if sync_token:
                params["syncToken"] = sync_token
            else:
                # Bootstrap: full sync just to get a fresh token, ignore results
                import datetime as _dt
                IST = _dt.timezone(_dt.timedelta(hours=5, minutes=30))
                params["updatedMin"] = _dt.datetime.now(tz=IST).strftime("%Y-%m-%dT%H:%M:%S+05:30")

            cancelled = []
            page_token = None
            new_sync_token = None

            while True:
                if page_token:
                    params["pageToken"] = page_token
                result = self.service.events().list(**params).execute()

                if not sync_token:
                    # Bootstrap run — just capture the token, skip processing
                    new_sync_token = result.get("nextSyncToken")
                    if not result.get("nextPageToken"):
                        break
                    page_token = result["nextPageToken"]
                    continue

                for ev in result.get("items", []):
                    if ev.get("status") == "cancelled":
                        ev_id = ev["id"]
                        if ev_id in _event_registry:
                            cancelled.append({"event_id": ev_id, **_event_registry[ev_id]})
                            del _event_registry[ev_id]

                new_sync_token = result.get("nextSyncToken")
                page_token = result.get("nextPageToken")
                if not page_token:
                    break

            return cancelled, new_sync_token

        except HttpError as e:
            if e.resp.status == 410:  # syncToken expired — resync
                print("[Calendar] syncToken expired, resyncing")
                return [], None
            print(f"[Calendar Error] get_cancelled_since: {e}")
            return [], sync_token
        except Exception as e:
            print(f"[Calendar Error] get_cancelled_since: {e}")
            return [], sync_token
