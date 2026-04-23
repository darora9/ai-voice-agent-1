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
    import datetime as _dt
    try:
        d = _dt.date.fromisoformat(date)
        day_names = ["Somwar", "Mangalwar", "Budhwar", "Guruwar", "Shukrawar", "Shaniwar", "Itwar"]
        month_names = ["January","February","March","April","May","June",
                       "July","August","September","October","November","December"]
        human_date = f"{d.day} {month_names[d.month-1]} ({day_names[d.weekday()]})"
    except Exception:
        human_date = date
    return (
        f"Bilkul! {name} ji, aapki appointment confirm ho gayi hai. "
        f"Taareekh: {human_date}, Samay: {time} baje. "
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
            if await self._is_slot_query(user_input):
                return await self._handle_slot_query(user_input)
            return await self._handle_date_only(user_input)
        elif self.state == State.WAIT_TIME:
            if await self._is_slot_query(user_input):
                return await self._handle_slot_query(user_input)
            return await self._handle_time_only(user_input)
        elif self.state == State.WAIT_CONFIRM:
            if await self._is_slot_query(user_input):
                return await self._handle_slot_query(user_input)
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
            else:
                self.time = ""  # clear old failed time so _check_slot asks fresh
            return await self._check_slot()

        if not time:
            if not self.available_slots:
                self.state = State.WAIT_DATETIME
                return "Kripya taareekh aur samay batayein."
            slots_str = ", ".join(self.available_slots)
            return f"Kripya inme se ek samay chunein: {slots_str} baje"

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
        if slots == "SUNDAY":
            self.date = ""
            self.time = ""
            self.state = State.WAIT_DATETIME
            return "Itwar ko clinic band rehti hai. Kripya koi aur din chunein, Monday se Saturday."

        self.available_slots = slots

        # Filter out slots that are already in the past (relevant for today)
        self.available_slots = [
            s for s in slots
            if not self._is_past_slot(self.date, s)
        ]
        slots = self.available_slots

        # If the requested time has passed today, give a specific message
        if self.time and self._is_past_slot(self.date, self.time):
            import datetime as _dt
            IST = _dt.timezone(_dt.timedelta(hours=5, minutes=30))
            is_today = self.date == _dt.datetime.now(tz=IST).date().isoformat()
            if is_today:
                if slots:
                    before, after = _nearby_slots(self.time, slots)
                    next_slot = after or before
                    requested = self.time
                    self.time = ""
                    self.state = State.WAIT_SLOT_CHOICE
                    if next_slot:
                        return f"{requested} baje ka samay nikal chuka hai. Agle available slot {next_slot} baje hai — confirm karun?"
                    else:
                        self.date = ""
                        self.state = State.WAIT_DATETIME
                        return "Aaj ke baaki saare slots nikal chuke hain. Kripya koi aur din batayein."
                else:
                    self.date = ""
                    self.time = ""
                    self.state = State.WAIT_DATETIME
                    return "Aaj ke baaki saare slots nikal chuke hain. Kripya koi aur din batayein."

        if not slots:
            import datetime as _dt
            IST = _dt.timezone(_dt.timedelta(hours=5, minutes=30))
            saved_date = self.date
            is_today = saved_date == _dt.datetime.now(tz=IST).date().isoformat()
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
        elif not self.time:
            # No preferred time — just ask for one
            self.state = State.WAIT_TIME
            first, last = slots[0], slots[-1]
            return (
                f"{self.date} ko {first} baje se {last} baje tak slots available hain. "
                "Aap kaunsa samay prefer karenge?"
            )
        else:
            before, after = _nearby_slots(self.time, slots)
            self.state = State.WAIT_SLOT_CHOICE
            return _slot_taken_nearby(self.date, self.time, before, after)

    def _is_past_date(self, date: str) -> bool:
        import datetime as _dt
        try:
            IST = _dt.timezone(_dt.timedelta(hours=5, minutes=30))
            today_ist = _dt.datetime.now(tz=IST).date()
            return _dt.date.fromisoformat(date) < today_ist
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
            "kya slot", "koi slot", "slot available", "slots available",
            "kab available", "slots hain", "slots hai", "slots bata",
            "khali slot", "koi jagah", "slot khali",
            "कोई slot", "खाली slot", "slot खाली",
            "kab hai slot", "kab milega slot", "kaun sa slot", "kon sa slot",
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
        if all_slots == "SUNDAY":
            self.date = ""
            self.state = State.WAIT_DATETIME
            return "Itwar ko clinic band rehti hai. Kripya koi aur din chunein, Monday se Saturday."
        slots = [s for s in all_slots if not self._is_past_slot(query_date, s)]

        # Remember the queried date so follow-up "book kar do" works without re-asking
        self.date = query_date
        self.available_slots = slots

        if not slots:
            import datetime as _dt
            IST = _dt.timezone(_dt.timedelta(hours=5, minutes=30))
            is_today = query_date == _dt.datetime.now(tz=IST).date().isoformat()
            self.date = ""
            self.state = State.WAIT_DATETIME
            return _no_slots_on_date(query_date, is_today=is_today)

        if query_time:
            # Apply PM flip same as other handlers
            try:
                h, m = map(int, query_time.split(":"))
                if h < 9:
                    query_time = f"{h + 12:02d}:{m:02d}"
            except Exception:
                pass
            # Check if requested time is already in the past (today only)
            import datetime as _dt
            IST = _dt.timezone(_dt.timedelta(hours=5, minutes=30))
            if self._is_past_slot(query_date, query_time) and query_date == _dt.datetime.now(tz=IST).date().isoformat():
                if slots:
                    before, after = _nearby_slots(query_time, slots)
                    next_slot = after or before
                    if next_slot:
                        self.time = ""
                        self.state = State.WAIT_SLOT_CHOICE
                        return f"{query_time} baje ka samay nikal chuka hai. Agle available slot {next_slot} baje hai — confirm karun?"
                return "Aaj ke baaki saare slots nikal chuke hain. Kripya koi aur din batayein."
            # Caller asked about a specific time on that date
            if query_time in slots:
                # Slot IS available — prime for confirmation
                self.time = query_time
                self.state = State.WAIT_CONFIRM
                return f"{query_date} ko {query_time} baje ka slot available hai. Kya main yeh book karun?"
            before, after = _nearby_slots(query_time, slots)
            # Slot not available — prime for slot choice
            self.time = query_time
            self.state = State.WAIT_SLOT_CHOICE
            return _slot_taken_nearby(query_date, query_time, before, after)
        else:
            # No time given — ask for time next
            self.state = State.WAIT_TIME
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
        IST = _dt.timezone(_dt.timedelta(hours=5, minutes=30))
        today_obj = _dt.datetime.now(tz=IST).date()
        today    = today_obj.isoformat()
        tomorrow = (today_obj + _dt.timedelta(days=1)).isoformat()
        day_after = (today_obj + _dt.timedelta(days=2)).isoformat()

        # Pre-compute the next occurrence of every weekday (always future, never today)
        weekday_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        hindi_weekdays = ["Somwar", "Mangalwar", "Budhwar", "Guruwar", "Shukrawar", "Shaniwar", "Itwar"]
        weekday_map: dict[str, str] = {}
        for idx, (en, hi) in enumerate(zip(weekday_names, hindi_weekdays)):
            days_ahead = (idx - today_obj.weekday()) % 7 or 7  # never 0 (=today)
            target = (today_obj + _dt.timedelta(days=days_ahead)).isoformat()
            weekday_map[en] = target
            weekday_map[hi] = target

        weekday_examples = "; ".join(
            f"'next {en}' or '{en}' or '{hi}'->{date}"
            for (en, hi), date in zip(zip(weekday_names, hindi_weekdays), weekday_map[n] for n in weekday_names)
        )

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
                            f"Weekday names always mean the NEXT upcoming occurrence: {weekday_examples}. "
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

