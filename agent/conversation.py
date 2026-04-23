"""
State-machine conversation manager.

Python controls every step of the flow.
LLM is only used for:
  - Extracting name from free text
  - Extracting date/time (handles Hindi: kal, parso, shaam, subah etc.)

Flow:
  WAIT_NAME
    → WAIT_DATETIME     (tell clinic hours, ask date+time)
      → WAIT_TIME       (have date, need time)
      → WAIT_DATE       (have time, need date)
    → WAIT_CONFIRM      (slot available, ask yes/no confirmation)
    → WAIT_SLOT_CHOICE  (slot taken, offered alternatives)
    → DONE
"""

import os
import json
from enum import Enum
from groq import AsyncGroq

from agent.prompts import CLINIC_HOURS, CLINIC_NAME, DOCTOR_NAME, get_today_iso
from services.calendar_service import CalendarService


class State(Enum):
    WAIT_NAME        = "wait_name"
    WAIT_DATETIME    = "wait_datetime"    # need both date and time
    WAIT_DATE        = "wait_date"        # have time, need date
    WAIT_TIME        = "wait_time"        # have date, need time
    WAIT_CONFIRM     = "wait_confirm"     # slot available, awaiting yes/no
    WAIT_SLOT_CHOICE = "wait_slot_choice" # slot taken, picking alternative
    DONE             = "done"


# ---------------------------------------------------------------------------
# Fixed Hindi response templates — no LLM variation, always consistent
# ---------------------------------------------------------------------------

def _greeting_with_hours(name: str) -> str:
    return (
        f"{name} ji, namaste! "
        f"Hamare clinic ke kaam ke ghante hain: {CLINIC_HOURS}. "
        "Aap kaunse din aur kis samay appointment lena chahenge?"
    )

def _ask_time(date: str) -> str:
    return f"{date} theek hai. Aap kis samay aana chahenge? (Subah 9 baje se shaam 6 baje ke beech)"

def _ask_date(time: str) -> str:
    return f"{time} ka samay theek hai. Aap kaunse din aana chahenge?"

def _slot_available_confirm(name: str, date: str, time: str) -> str:
    return (
        f"{name} ji, {date} ko {time} baje ka slot available hai. "
        "Kya main yeh appointment confirm karun?"
    )

def _nearby_slots(requested: str, available: list) -> tuple[str | None, str | None]:
    """Return (slot_before, slot_after) closest to requested time from available list."""
    if not available:
        return None, None
    before = None
    after = None
    for s in sorted(available):
        if s < requested:
            before = s
        elif s > requested and after is None:
            after = s
    return before, after


def _slot_taken_nearby(date: str, time: str, before: str | None, after: str | None) -> str:
    parts = []
    if before:
        parts.append(before)
    if after:
        parts.append(after)
    nearby_str = " aur ".join(parts) if parts else None
    if nearby_str:
        return (
            f"Maafi kijiye, {date} ko {time} baje ka slot available nahi hai. "
            f"Nearby slots hain: {nearby_str} baje. Kaunsa theek rahega?"
        )
    return (
        f"Maafi kijiye, {date} ko {time} baje ka slot available nahi hai. "
        "Kripya koi aur samay batayein."
    )


def _slot_taken(date: str, time: str, suggestions: list) -> str:
    # fallback — not used for main flow anymore
    slots = ", ".join(suggestions)
    return (
        f"Maafi kijiye, {date} ko {time} baje ka slot available nahi hai. "
        f"In slots mein se chunein: {slots} baje. Aap kaunsa samay prefer karenge?"
    )

def _no_slots_on_date(date: str, is_today: bool = False) -> str:
    if is_today:
        return "Aaj pure din mein koi slot available nahi hai. Kripya kisi aur din ki taareekh batayein."
    return f"{date} ko pure din mein koi bhi slot available nahi hai. Kripya koi aur taareekh batayein."

def _booking_confirmed(name: str, date: str, time: str) -> str:
    return (
        f"Bilkul! {name} ji, aapki appointment confirm ho gayi hai. "
        f"Taareekh: {date}, Samay: {time} baje. "
        f"Aapko {CLINIC_NAME} mein milenge. Dhanyavad!"
    )

def _booking_failed() -> str:
    return (
        "Maafi chahta hoon, booking mein kuch samasya aayi. "
        "Kripya dobaara taareekh aur samay batayein."
    )


GREETING = (
    f"Namaste! {CLINIC_NAME} mein aapka swagat hai. "
    "Main aapki appointment book karne mein madad karunga. "
    "Kripya apna poora naam batayein."
)


# ---------------------------------------------------------------------------
# Conversation Manager
# ---------------------------------------------------------------------------

class ConversationManager:
    def __init__(self):
        self.client = AsyncGroq(api_key=os.environ["GROQ_API_KEY"])
        self.model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        self.calendar = CalendarService()

        self.state = State.WAIT_NAME

        # Collected data
        self.patient_name: str = ""
        self.date: str = ""         # YYYY-MM-DD
        self.time: str = ""         # HH:MM
        self.available_slots: list = []

    def get_greeting(self) -> str:
        return GREETING

    async def process_turn(self, user_input: str) -> str:
        user_input = user_input.strip()
        if not user_input:
            return ""

        if self.state == State.WAIT_NAME:
            return await self._handle_name(user_input)
        elif self.state == State.WAIT_DATETIME:
            # Check if it's a slot availability question before normal date/time handling
            if await self._is_slot_query(user_input):
                return await self._handle_slot_query(user_input)
            return await self._handle_datetime(user_input)
        elif self.state == State.WAIT_DATE:
            return await self._handle_date_only(user_input)
        elif self.state == State.WAIT_TIME:
            return await self._handle_time_only(user_input)
        elif self.state == State.WAIT_CONFIRM:
            return await self._handle_confirm(user_input)
        elif self.state == State.WAIT_SLOT_CHOICE:
            # Also handle availability questions during slot choice
            if await self._is_slot_query(user_input):
                return await self._handle_slot_query(user_input)
            return await self._handle_slot_choice(user_input)
        elif self.state == State.DONE:
            return "Aapki appointment pehle se book ho chuki hai. Dhanyavad!"
        return ""

    # ------------------------------------------------------------------
    # State handlers
    # ------------------------------------------------------------------

    async def _handle_name(self, text: str) -> str:
        name = await self._extract_name(text)
        if not name:
            return "Maafi chahta hoon, kripya apna poora naam batayein."
        self.patient_name = name
        self.state = State.WAIT_DATETIME
        return _greeting_with_hours(self.patient_name)

    async def _handle_datetime(self, text: str) -> str:
        # If caller is correcting their name ("mera naam X hai"), re-extract it
        t_lower = text.lower()
        if any(kw in t_lower for kw in ("my name is", "naam hai", "naam he", "mera naam", "main hoon", "i am")):
            name = await self._extract_name(text)
            if name:
                self.patient_name = name
                return _greeting_with_hours(self.patient_name)

        dt = await self._extract_datetime(text)
        date, time = dt.get("date"), dt.get("time")

        if date and self._is_past_date(date):
            return "Yeh taareekh guzar chuki hai. Kripya aaj ya aane wali taareekh batayein."

        if date and time:
            self.date, self.time = date, time
            return await self._check_slot()
        elif date:
            self.date = date
            self.state = State.WAIT_TIME
            return _ask_time(self.date)
        elif time:
            self.time = time
            self.state = State.WAIT_DATE
            return _ask_date(self.time)
        else:
            return (
                "Kripya taareekh aur samay batayein, "
                "jaise 'kal shaam 4 baje' ya 'agle shukrawar subah 11 baje'."
            )

    async def _handle_date_only(self, text: str) -> str:
        dt = await self._extract_datetime(text)
        date = dt.get("date")
        if not date:
            return "Kripya taareekh batayein, jaise 'kal', 'parso', ya '25 April'."
        if self._is_past_date(date):
            return "Yeh taareekh guzar chuki hai. Kripya aaj ya aane wali taareekh batayein."
        self.date = date
        return await self._check_slot()

    async def _handle_time_only(self, text: str) -> str:
        dt = await self._extract_datetime(text)
        time = dt.get("time")
        if not time:
            return "Kripya samay batayein, jaise 'subah 10 baje' ya 'shaam 3 baje'."
        # Apply PM flip: hour < 9 outside clinic hours means PM was intended
        try:
            h, m = map(int, time.split(":"))
            if h < 9:
                time = f"{h + 12:02d}:{m:02d}"
        except Exception:
            pass
        self.time = time
        return await self._check_slot()

    async def _handle_slot_choice(self, text: str) -> str:
        dt = await self._extract_datetime(text)
        new_date = dt.get("date")
        time = dt.get("time")

        # If user gave a completely new date, restart the slot check for that date
        if new_date and new_date != self.date:
            if self._is_past_date(new_date):
                return "Yeh taareekh guzar chuki hai. Kripya aaj ya aane wali taareekh batayein."
            self.date = new_date
            if time:
                # Apply PM flip before storing
                try:
                    h, m = map(int, time.split(":"))
                    if h < 9:
                        time = f"{h + 12:02d}:{m:02d}"
                except Exception:
                    pass
                self.time = time
            return await self._check_slot()

        if not time:
            slots_str = ", ".join(self.available_slots)
            return f"Kripya inme se ek samay chunein: {slots_str}"

        # If LLM returned an AM time (hour < 9) that's outside clinic hours,
        # flip it to PM — e.g. "02:00" from "2 bje" → "14:00"
        try:
            h, m = map(int, time.split(":"))
            if h < 9:
                time = f"{h + 12:02d}:{m:02d}"
        except Exception:
            pass

        matched = self._match_slot(time)
        if matched:
            self.time = matched
            self.state = State.WAIT_CONFIRM
            return _slot_available_confirm(self.patient_name, self.date, self.time)
        else:
            before, after = _nearby_slots(time, self.available_slots)
            return _slot_taken_nearby(self.date, time, before, after)

    async def _handle_confirm(self, text: str) -> str:
        import re as _re
        text_lower = text.lower().strip()
        # Matches both romanized AND Devanagari affirmatives/negatives
        affirm = bool(_re.search(
            r"(haan|hnji|ha\b|yes|bilkul|theek|ok\b|okay|confirm|karo|krdo|kar\s*do|zaroor|sahi|done"
            r"|हाँ|हां|हा\b|हजी|हन्जी|बिल्कुल|ठीक|करो|ज़रूर|जरूर|सही|दीजिए|dijiye|kijiye|कीजिए)",
            text_lower
        ))
        deny = bool(_re.search(
            r"(nahi|nahin|no\b|nope|cancel|mat\b|naa\b|नहीं|नहि|नहीं|मत\b|ना\b)",
            text_lower
        ))

        if affirm and not deny:
            return await self._book_now()
        elif deny:
            self.date = ""
            self.time = ""
            self.state = State.WAIT_DATETIME
            return "Theek hai. Kripya naya din aur samay batayein."
        else:
            return f"{self.patient_name} ji, {self.date} ko {self.time} baje — kya confirm karun?"

    async def _book_now(self) -> str:
        result = self.calendar.book_appointment(
            patient_name=self.patient_name,
            patient_phone="",
            date_str=self.date,
            time_str=self.time,
        )
        if result.get("success"):
            self.state = State.DONE
            return _booking_confirmed(self.patient_name, self.date, self.time)
        else:
            self.date = ""
            self.time = ""
            self.state = State.WAIT_DATETIME
            return _booking_failed()

    # ------------------------------------------------------------------
    # Availability check (shared)
    # ------------------------------------------------------------------

    async def _check_slot(self) -> str:
        import re as _re
        if not _re.fullmatch(r"\d{4}-\d{2}-\d{2}", self.date):
            # LLM returned a placeholder instead of a real date
            self.date = ""
            self.time = ""
            self.state = State.WAIT_DATETIME
            return "Kripya taareekh phir se batayein, jaise 'kal', 'parso', ya '25 April'."

        slots = self.calendar.get_available_slots(self.date)
        if slots is None:
            return "Calendar se jaankari nahi mil payi. Kripya thodi der baad phir try karein."

        self.available_slots = slots

        # Filter out slots that are already in the past (relevant for today)
        self.available_slots = [
            s for s in slots
            if not self._is_past_slot(self.date, s)
        ]
        slots = self.available_slots

        if not slots:
            import datetime as _dt
            saved_date = self.date
            is_today = saved_date == _dt.date.today().isoformat()
            self.date = ""
            self.time = ""
            self.state = State.WAIT_DATETIME
            return _no_slots_on_date(saved_date, is_today=is_today)

        matched = self._match_slot(self.time)
        if matched and self._is_past_slot(self.date, matched):
            matched = None
        if matched:
            self.time = matched
            self.state = State.WAIT_CONFIRM
            return _slot_available_confirm(self.patient_name, self.date, self.time)
        else:
            before, after = _nearby_slots(self.time, slots)
            self.state = State.WAIT_SLOT_CHOICE
            return _slot_taken_nearby(self.date, self.time, before, after)

    def _is_past_date(self, date: str) -> bool:
        import datetime as _dt
        try:
            return _dt.date.fromisoformat(date) < _dt.date.today()
        except Exception:
            return False

    def _is_past_slot(self, date: str, time: str) -> bool:
        """Returns True if the date+time combination is already in the past (IST-aware)."""
        import datetime as _dt
        try:
            IST = _dt.timezone(_dt.timedelta(hours=5, minutes=30))
            slot_dt = _dt.datetime.strptime(f"{date} {time}", "%Y-%m-%d %H:%M").replace(tzinfo=IST)
            return slot_dt <= _dt.datetime.now(tz=IST)
        except Exception:
            return False

    def _match_slot(self, time: str) -> str:
        if time in self.available_slots:
            return time
        padded = time.zfill(5)
        if padded in self.available_slots:
            return padded
        return None

    # ------------------------------------------------------------------
    # Slot availability queries (interactive calendar questions)
    # ------------------------------------------------------------------

    async def _is_slot_query(self, text: str) -> bool:
        """Detect if the caller is asking about slot availability rather than booking."""
        t = text.lower()
        keywords = [
            "kya slot", "koi slot", "available", "available hai", "khali",
            "kab available", "slots hain", "slots hai", "slots bata",
            "कोई slot", "कब available", "खाली", "available है",
            "kab hai", "kab milega", "kab milegi", "when is", "kaun sa slot",
        ]
        return any(kw in t for kw in keywords)

    async def _handle_slot_query(self, text: str) -> str:
        """Answer a caller's question about available slots on a date."""
        dt = await self._extract_datetime(text)
        query_date = dt.get("date") or self.date
        query_time = dt.get("time")

        if not query_date:
            return "Kripya taareekh batayein jiske baare mein jaanna chahte hain."

        import re as _re
        if not _re.fullmatch(r"\d{4}-\d{2}-\d{2}", query_date):
            return "Kripya sahi taareekh batayein."

        all_slots = self.calendar.get_available_slots(query_date)
        if all_slots is None:
            return "Calendar se jaankari nahi mil payi. Kripya thodi der baad phir try karein."
        slots = [s for s in all_slots if not self._is_past_slot(query_date, s)]

        # Remember the queried date so follow-up "book kar do" works without re-asking
        self.date = query_date
        if not query_time:
            # We have a date but no time — move to WAIT_TIME for follow-up booking
            self.state = State.WAIT_TIME
        else:
            # We have both date and time from the query — prime for slot choice
            self.time = query_time
            self.state = State.WAIT_SLOT_CHOICE
        self.available_slots = slots

        if not slots:
            import datetime as _dt
            is_today = query_date == _dt.date.today().isoformat()
            return _no_slots_on_date(query_date, is_today=is_today)

        if query_time:
            # Caller asked about a specific time on that date
            if query_time in slots:
                return f"{query_date} ko {query_time} baje ka slot available hai. Kya main yeh book karun?"
            before, after = _nearby_slots(query_time, slots)
            return _slot_taken_nearby(query_date, query_time, before, after)
        else:
            # Caller asked generally — give first and last available as a range
            first, last = slots[0], slots[-1]
            return (
                f"{query_date} ko {first} baje se {last} baje tak slots available hain. "
                "Aap kaunsa samay prefer karenge?"
            )

    # ------------------------------------------------------------------
    # LLM extraction helpers (extraction only — no conversation logic)
    # ------------------------------------------------------------------

    async def _extract_name(self, text: str) -> str:
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Extract the person's name from the text. "
                            "It can be a single name or full name — both are valid. "
                            "Capitalize it properly. "
                            'Return only JSON: {"name": "Name"} or {"name": null} if no name is present.'
                        ),
                    },
                    {"role": "user", "content": text},
                ],
                response_format={"type": "json_object"},
                temperature=0,
                max_tokens=30,
            )
            data = json.loads(response.choices[0].message.content)
            return data.get("name") or None
        except Exception as e:
            print(f"[LLM Error] _extract_name: {e}")
            return None

    async def _extract_datetime(self, text: str) -> dict:
        import datetime as _dt
        today_obj = _dt.date.today()
        today    = today_obj.isoformat()
        tomorrow = (today_obj + _dt.timedelta(days=1)).isoformat()
        day_after = (today_obj + _dt.timedelta(days=2)).isoformat()
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            f"Today is {today} (YYYY-MM-DD). "
                            "Extract date and/or time from Hindi/English text. "
                            "ALWAYS return actual calendar dates in YYYY-MM-DD format, never placeholder words. "
                            'Return JSON: {"date": "YYYY-MM-DD", "time": "HH:MM"} — use null if not present. '
                            "Relative dates: 'aaj'=today, 'kal'=tomorrow, 'parso'=day after tomorrow, 'agle X'=next X. "
                            "Time words: 'subah'=morning (AM), 'dopahar'=noon (12:00-15:00), "
                            "'shaam'=evening (add 12 if hour<8, e.g. shaam 5=17:00), 'raat'=night (add 12 if hour<8). "
                            "No qualifier: if hour 1-8 assume PM (add 12). If hour 9-12 assume AM. "
                            "Examples: '2 bje'->14:00, '3 baje'->15:00, '10 baje'->10:00, '11 bje'->11:00. "
                            f"Examples using today={today}: "
                            f"'kal shaam 5 baje'->" + '{"date":"' + tomorrow + '","time":"17:00"}, '
                            f"'parso shaam 5 bje'->" + '{"date":"' + day_after + '","time":"17:00"}, '
                            f"'subah 10 baje'->" + '{"date":null,"time":"10:00"}, '
                            f"'parso'->" + '{"date":"' + day_after + '","time":null}.'
                        ),
                    },
                    {"role": "user", "content": text},
                ],
                response_format={"type": "json_object"},
                temperature=0,
                max_tokens=50,
            )
            data = json.loads(response.choices[0].message.content)
            date = data.get("date") if data.get("date") not in (None, "null", "") else None
            time = data.get("time") if data.get("time") not in (None, "null", "") else None
            return {"date": date, "time": time}
        except Exception as e:
            print(f"[LLM Error] _extract_datetime: {e}")
            return {"date": None, "time": None}

