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

# ---------------------------------------------------------------------------
# Clinic appointment windows — edit capacities here
# Each entry: (window_start "HH:MM", window_end "HH:MM", max_appointments)
# ---------------------------------------------------------------------------
SLOT_WINDOWS = [
    ("09:30", "10:00",  5),   # 9:30–10:00 AM  —  5 slots
    ("10:00", "11:00", 10),   # 10:00–11:00 AM — 10 slots
    ("11:00", "12:00", 10),   # 11:00–12:00 PM — 10 slots
    ("12:00", "13:00", 10),   # 12:00–1:00 PM  — 10 slots
    ("13:00", "14:00", 10),   # 1:00–2:00 PM   — 10 slots
    ("17:00", "18:00", 12),   # 5:00–6:00 PM   — 12 slots
    ("18:00", "18:30",  6),   # 6:00–6:30 PM   —  6 slots
]


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

    # ------------------------------------------------------------------
    # Window-based capacity helpers
    # ------------------------------------------------------------------

    def _get_window(self, time_str: str):
        """Return (start, end, capacity) for the window containing time_str, or None."""
        try:
            h, m = map(int, time_str.split(":"))
            t_mins = h * 60 + m
            for ws, we, cap in SLOT_WINDOWS:
                wsh, wsm = map(int, ws.split(":"))
                weh, wem = map(int, we.split(":"))
                if wsh * 60 + wsm <= t_mins < weh * 60 + wem:
                    return (ws, we, cap)
        except Exception:
            pass
        return None

    def _fetch_day_events(self, date_str: str) -> list | None:
        """Fetch all calendar events for a date. Returns list or None on error."""
        try:
            day_start = datetime.strptime(f"{date_str} 00:00", "%Y-%m-%d %H:%M")
            day_end   = datetime.strptime(f"{date_str} 23:59", "%Y-%m-%d %H:%M")
            result = (
                self.service.events()
                .list(
                    calendarId=self.calendar_id,
                    timeMin=day_start.strftime("%Y-%m-%dT%H:%M:00+05:30"),
                    timeMax=day_end.strftime("%Y-%m-%dT%H:%M:00+05:30"),
                    singleEvents=True,
                )
                .execute()
            )
            return result.get("items", [])
        except Exception as e:
            print(f"[Calendar Error] _fetch_day_events: {e}")
            return None

    def _count_per_window(self, events: list) -> dict:
        """Return {(ws, we): count} of bookings per window."""
        counts = {(ws, we): 0 for ws, we, _ in SLOT_WINDOWS}
        for event in events:
            ev_start = event["start"].get("dateTime")
            if not ev_start:
                continue
            try:
                dt = datetime.fromisoformat(ev_start).replace(tzinfo=None)
                win = self._get_window(dt.strftime("%H:%M"))
                if win:
                    key = (win[0], win[1])
                    if key in counts:
                        counts[key] += 1
            except Exception:
                pass
        return counts

    def is_time_available(self, date_str: str, time_str: str) -> tuple[bool, int, int]:
        """
        Check capacity for the window containing time_str.
        Returns (available, booked_count, window_capacity).
        Returns (False, 0, 0) if time is outside clinic hours.
        """
        window = self._get_window(time_str)
        if not window:
            return False, 0, 0
        ws, we, cap = window
        events = self._fetch_day_events(date_str)
        if events is None:
            return False, 0, 0
        counts = self._count_per_window(events)
        booked = counts.get((ws, we), 0)
        return booked < cap, booked, cap

    def get_available_windows(self, date_str: str) -> list | None:
        """
        Return windows with remaining capacity as list of dicts:
        {"start": "HH:MM", "end": "HH:MM", "capacity": N, "remaining": N}
        Returns None on API error, [] if no windows open.
        """
        try:
            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
            if date_obj.weekday() == 6:
                return []
            events = self._fetch_day_events(date_str)
            if events is None:
                return None
            counts = self._count_per_window(events)
            return [
                {
                    "start": ws,
                    "end": we,
                    "capacity": cap,
                    "remaining": cap - counts.get((ws, we), 0),
                }
                for ws, we, cap in SLOT_WINDOWS
                if counts.get((ws, we), 0) < cap
            ]
        except Exception as e:
            print(f"[Calendar Error] get_available_windows: {e}")
            return None

    def get_available_slots(self, date_str: str) -> list[str]:
        """
        Window-aware slot availability. Returns window-start times with capacity,
        'SUNDAY', or None on error.
        Replaces the old per-30min slot generation.
        """
        try:
            # Clinic is Mon-Sat only; Sunday has no slots
            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
            if date_obj.weekday() == 6:  # 6 = Sunday
                return "SUNDAY"  # special sentinel so caller gets a clear message
            tz = "Asia/Kolkata"
            windows = self.get_available_windows(date_str)
            if windows is None:
                return None
            return [w["start"] for w in windows]

        except Exception as e:
            print(f"[Calendar Error] {e}")
            return None  # None = error, [] = genuinely no slots

    def get_next_available_slot(self) -> dict | None:
        """Return {date, time} of the earliest available slot from today, or None."""
        import datetime as _dt
        IST = _dt.timezone(_dt.timedelta(hours=5, minutes=30))
        today = _dt.datetime.now(tz=IST).date()
        now_ist = _dt.datetime.now(tz=IST)
        for day_offset in range(30):  # look up to 30 days ahead
            check_date = today + _dt.timedelta(days=day_offset)
            if check_date.weekday() == 6:  # skip Sunday
                continue
            windows = self.get_available_windows(check_date.isoformat())
            if not windows:
                continue
            for w in windows:
                slot_dt = _dt.datetime.strptime(
                    f"{check_date.isoformat()} {w['start']}", "%Y-%m-%d %H:%M"
                ).replace(tzinfo=IST)
                if slot_dt > now_ist:
                    return {"date": check_date.isoformat(), "time": w["start"]}
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
        for day_offset in range(30):
            check_date = start + _dt.timedelta(days=day_offset)
            if check_date.weekday() == 6:  # skip Sunday
                continue
            windows = self.get_available_windows(check_date.isoformat())
            if not windows:
                continue
            for w in windows:
                slot_dt = _dt.datetime.strptime(
                    f"{check_date.isoformat()} {w['start']}", "%Y-%m-%d %H:%M"
                ).replace(tzinfo=IST)
                if slot_dt > now_ist:
                    return {"date": check_date.isoformat(), "time": w["start"]}
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
