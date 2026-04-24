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
        f"{name} जी, नमस्ते! "
        f"Clinic का समय: {CLINIC_HOURS}. "
        "आप किस दिन और समय appointment लेना चाहेंगे?"
    )

def _ask_time(date: str) -> str:
    return f"{date} ठीक है। किस समय आना चाहेंगे?"

def _ask_date(time: str) -> str:
    return f"{time} ठीक है। कौनसा दिन आना चाहेंगे?"

def _slot_available_confirm(name: str, date: str, time: str) -> str:
    return f"{name} जी, {date} को {time} बजे slot available है। Confirm करूँ?"

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


def _has_time_qualifier(text: str) -> bool:
    """Return True if the text contains an explicit AM/PM qualifier.
    When True, the LLM already resolved the correct hour — skip our PM flip."""
    t = text.lower()
    qualifiers = (
        "subah", "सुबह", "morning",
        "dopahar", "दोपहर", "duphar", "noon",
        "shaam", "शाम", "sham", "evening",
        "raat", "रात", "night",
        "am", "pm",
    )
    return any(q in t for q in qualifiers)


def _slot_taken_nearby(date: str, time: str, before: str | None, after: str | None) -> str:
    parts = []
    if before:
        parts.append(before)
    if after:
        parts.append(after)
    nearby_str = " और ".join(parts) if parts else None
    if nearby_str:
        return f"{time} बजे slot नहीं है। पास में: {nearby_str} बजे। कौनसा ठीक रहेगा?"
    return f"{time} बजे slot नहीं है। कोई और समय बताएं।"


def _slot_taken(date: str, time: str, suggestions: list) -> str:
    # fallback — unused, kept for reference
    slots = ", ".join(suggestions)
    return (
        f"माफ़ी चाहते हैं, {date} को {time} बजे का slot available नहीं है। "
        f"इन slots में से चुनें: {slots} बजे।"
    )

def _no_slots_on_date(date: str, is_today: bool = False) -> str:
    if is_today:
        return "आज कोई slot नहीं है। कोई और दिन बताएं।"
    return f"{date} को कोई slot नहीं है। कोई और तारीख़ बताएं।"

def _booking_confirmed(name: str, date: str, time: str) -> str:
    import datetime as _dt
    try:
        d = _dt.date.fromisoformat(date)
        day_names = ["सोमवार", "मंगलवार", "बुधवार", "गुरुवार", "शुक्रवार", "शनिवार", "रविवार"]
        month_names = ["January","February","March","April","May","June",
                       "July","August","September","October","November","December"]
        human_date = f"{d.day} {month_names[d.month-1]} ({day_names[d.weekday()]})"
    except Exception:
        human_date = date
    return (
        f"बिल्कुल! {name} जी, {human_date} को {time} बजे appointment confirm हो गई। धन्यवाद!"
    )

def _booking_failed() -> str:
    return "माफ़ी चाहते हैं, booking नहीं हो पाई। दोबारा तारीख़ और समय बताएं।"


GREETING = (
    f"नमस्ते! {CLINIC_NAME} में आपका स्वागत है। "
    "अपना नाम बताएं।"
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
            return "आपकी appointment पहले से book हो चुकी है। धन्यवाद!"
        return ""

    # ------------------------------------------------------------------
    # State handlers
    # ------------------------------------------------------------------

    async def _handle_name(self, text: str) -> str:
        name = await self._extract_name(text)
        if not name:
            return "माफ़ी चाहते हैं, कृपया अपना पूरा नाम बताएं।"
        self.patient_name = name
        self.state = State.WAIT_DATETIME
        return _greeting_with_hours(self.patient_name)

    async def _handle_datetime(self, text: str) -> str:
        # If caller is correcting their name, re-extract it
        t_lower = text.lower()
        if any(kw in t_lower for kw in (
            "my name is", "naam hai", "naam he", "mera naam", "main hoon", "i am",
            "मेरा नाम", "नाम है", "नाम हे", "मैं हूँ", "मैं हूं",
        )):
            name = await self._extract_name(text)
            if name:
                self.patient_name = name
                return _greeting_with_hours(self.patient_name)

        dt = await self._extract_datetime(text)
        date, time = dt.get("date"), dt.get("time")

        if date and self._is_past_date(date):
            return "यह तारीख़ गुज़र चुकी है। कृपया आज या आने वाली तारीख़ बताएं।"

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
                "कृपया तारीख़ और समय बताएं, "
                "जैसे 'कल शाम 4 बजे' या 'अगले शुक्रवार सुबह 11 बजे'।"
            )

    async def _handle_date_only(self, text: str) -> str:
        dt = await self._extract_datetime(text)
        date = dt.get("date")
        if not date:
            return "कृपया तारीख़ बताएं, जैसे 'कल', 'परसों', या '25 April'।"
        if self._is_past_date(date):
            return "यह तारीख़ गुज़र चुकी है। कृपया आज या आने वाली तारीख़ बताएं।"
        self.date = date
        return await self._check_slot()

    async def _handle_time_only(self, text: str) -> str:
        dt = await self._extract_datetime(text)
        time = dt.get("time")
        new_date = dt.get("date")
        if not time:
            return "कृपया समय बताएं, जैसे 'सुबह 10 बजे' या 'शाम 3 बजे'।"
        # If LLM also found a date (e.g. 'कल सुबह 8 बजे' said while in WAIT_TIME), use it
        if new_date and not self._is_past_date(new_date):
            self.date = new_date
        # PM flip: only when no explicit qualifier (subah/shaam/raat)
        if not _has_time_qualifier(text):
            try:
                h, m = map(int, time.split(":"))
                if h < 7:
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
                return "यह तारीख़ गुज़र चुकी है। कृपया आज या आने वाली तारीख़ बताएं।"
            self.date = new_date
            if time:
                # Apply PM flip only when no explicit qualifier
                if not _has_time_qualifier(text):
                    try:
                        h, m = map(int, time.split(":"))
                        if h < 7:
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
                return "कृपया तारीख़ और समय बताएं।"
            slots_str = ", ".join(self.available_slots)
            return f"कृपया इनमें से एक समय चुनें: {slots_str} बजे"

        # PM flip: only when no explicit qualifier (subah/shaam/raat)
        # Hours 7-8 are never flipped (could be genuine morning)
        if not _has_time_qualifier(text):
            try:
                h, m = map(int, time.split(":"))
                if h < 7:
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
            return "ठीक है। कृपया नया दिन और समय बताएं।"
        else:
            return f"{self.patient_name} जी, {self.date} को {self.time} बजे — क्या confirm करूँ?"

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
            return "कृपया तारीख़ फिर से बताएं, जैसे 'कल', 'परसों', या '25 April'।"

        slots = self.calendar.get_available_slots(self.date)
        if slots is None:
            return "Calendar से जानकारी नहीं मिल पाई। कृपया थोड़ी देर बाद फिर try करें।"
        if slots == "SUNDAY":
            self.date = ""
            self.time = ""
            self.state = State.WAIT_DATETIME
            return "इतवार को clinic बंद रहती है। कृपया कोई और दिन चुनें, Monday से Saturday।"

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
                        return f"{requested} बजे का समय निकल चुका है। अगला available slot {next_slot} बजे है — confirm करूँ?"
                    else:
                        self.date = ""
                        self.state = State.WAIT_DATETIME
                        return "आज के बाकी सारे slots निकल चुके हैं। कृपया कोई और दिन बताएं।"
                else:
                    self.date = ""
                    self.time = ""
                    self.state = State.WAIT_DATETIME
                    return "आज के बाकी सारे slots निकल चुके हैं। कृपया कोई और दिन बताएं।"

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
                f"{self.date} को {first} बजे से {last} बजे तक slots available हैं। "
                "आप कौनसा समय prefer करेंगे?"
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

    @staticmethod
    def _preparse_date(text: str, today: 'object') -> 'str | None':
        """
        Pure-Python extraction for unambiguous date references.
        Runs BEFORE the LLM and its result overrides LLM output.
        Handles: aaj/kal/parso, 'DD Month', 'Month DD'.
        Weekday names are left to the LLM (need next-occurrence logic).
        """
        import re as _re
        import datetime as _dt
        tl = text.lower()

        # Relative day words (romanized + Devanagari + English)
        if _re.search(r'\baaj\b|\btoday\b|आज', tl):
            return today.isoformat()
        # 'kal' — check not part of longer Latin word (e.g. 'calendar')
        if _re.search(r'(?<![a-z])kal(?![a-z])|\btomorrow\b|कल', tl):
            return (today + _dt.timedelta(days=1)).isoformat()
        if _re.search(r'\bparso\b|\bparson\b|परसों|परसो\b', tl):
            return (today + _dt.timedelta(days=2)).isoformat()

        # Explicit 'DD Month' or 'Month DD'
        months = {
            'jan': 1, 'january': 1,
            'feb': 2, 'february': 2,
            'mar': 3, 'march': 3,
            'apr': 4, 'april': 4, 'अप्रैल': 4,
            'may': 5, 'मई': 5,
            'jun': 6, 'june': 6,
            'jul': 7, 'july': 7,
            'aug': 8, 'august': 8,
            'sep': 9, 'sept': 9, 'september': 9,
            'oct': 10, 'october': 10,
            'nov': 11, 'november': 11,
            'dec': 12, 'december': 12,
        }
        for mn, mnum in months.items():
            m = _re.search(
                rf'(\d{{1,2}})\s+{_re.escape(mn)}\b|\b{_re.escape(mn)}\s+(\d{{1,2}})\b',
                tl
            )
            if m:
                day = int(m.group(1) or m.group(2))
                year = today.year
                try:
                    candidate = _dt.date(year, mnum, day)
                    if candidate < today:
                        candidate = _dt.date(year + 1, mnum, day)
                    return candidate.isoformat()
                except ValueError:
                    pass
        return None


    async def _is_slot_query(self, text: str) -> bool:
        """Detect if the caller is asking about slot availability rather than booking."""
        t = text.lower()
        keywords = [
            "kya slot", "koi slot", "slot available", "slots available",
            "kab available", "slots hain", "slots hai", "slots bata",
            "khali slot", "koi jagah", "slot khali",
            "कोई slot", "खाली slot", "slot खाली", "slot है", "slot हैं",
            "kab hai slot", "kab milega slot", "kaun sa slot", "kon sa slot",
            "कब मिलेगा", "कब available", "कौनसा slot", "slot मिलेगा", "slot बताएं",
        ]
        return any(kw in t for kw in keywords)

    async def _handle_slot_query(self, text: str) -> str:
        """Answer a caller's question about available slots on a date."""
        dt = await self._extract_datetime(text)
        query_date = dt.get("date") or self.date
        query_time = dt.get("time")

        if not query_date:
            return "कृपया तारीख़ बताएं जिसके बारे में जानना चाहते हैं।"

        import re as _re
        if not _re.fullmatch(r"\d{4}-\d{2}-\d{2}", query_date):
            return "कृपया सही तारीख़ बताएं।"

        all_slots = self.calendar.get_available_slots(query_date)
        if all_slots is None:
            return "Calendar से जानकारी नहीं मिल पाई। कृपया थोड़ी देर बाद फिर try करें।"
        if all_slots == "SUNDAY":
            self.date = ""
            self.state = State.WAIT_DATETIME
            return "इतवार को clinic बंद रहती है। कृपया कोई और दिन चुनें, Monday से Saturday।"
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
            # Apply PM flip only when no explicit qualifier (subah/shaam/raat)
            if not _has_time_qualifier(text):
                try:
                    h, m = map(int, query_time.split(":"))
                    if h < 7:
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
                        return f"{query_time} बजे का समय निकल चुका है। अगला available slot {next_slot} बजे है — confirm करूँ?"
                return "आज के बाकी सारे slots निकल चुके हैं। कृपया कोई और दिन बताएं।"
            # Caller asked about a specific time on that date
            if query_time in slots:
                # Slot IS available — prime for confirmation
                self.time = query_time
                self.state = State.WAIT_CONFIRM
                return f"{query_date} को {query_time} बजे का slot available है। क्या मैं यह book करूँ?"
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
                f"{query_date} को {first} बजे से {last} बजे तक slots available हैं। "
                "आप कौनसा समय prefer करेंगे?"
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
                            "The input may be Hindi (Devanagari script), English, or a mix of both. "
                            "It can be a single name or full name — both are valid. "
                            "Capitalize it properly using Latin script (e.g. 'Krishna', 'Priya', 'Rahul'). "
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

    @staticmethod
    def _preparse_date(text: str, today: object) -> 'str | None':
        """
        Pure-Python extraction for unambiguous date references.
        Result overrides LLM output — more reliable for simple patterns.
        Handles: aaj/kal/parso, 'DD Month', 'Month DD'.
        Weekday names are left to the LLM (need calendar math).
        """
        import re as _re
        import datetime as _dt
        tl = text.lower()

        # Relative day words
        if _re.search(r'\baaj\b|\btoday\b|आज', tl):
            return today.isoformat()
        # 'kal' — avoid matching 'calendar', 'kalyan' etc.
        if _re.search(r'(?<![a-z])kal(?![a-z])|\btomorrow\b|कल', tl):
            return (today + _dt.timedelta(days=1)).isoformat()
        if _re.search(r'\bparso\b|\bparson\b|परसों|परसो\b', tl):
            return (today + _dt.timedelta(days=2)).isoformat()

        # Explicit 'DD Month' or 'Month DD' in any language
        months = {
            'jan': 1, 'january': 1, 'feb': 2, 'february': 2,
            'mar': 3, 'march': 3, 'apr': 4, 'april': 4, 'अप्रैल': 4,
            'may': 5, 'मई': 5, 'jun': 6, 'june': 6,
            'jul': 7, 'july': 7, 'aug': 8, 'august': 8,
            'sep': 9, 'sept': 9, 'september': 9,
            'oct': 10, 'october': 10, 'nov': 11, 'november': 11,
            'dec': 12, 'december': 12,
        }
        for mn, mnum in months.items():
            m = _re.search(
                rf'(\d{{1,2}})\s+{_re.escape(mn)}\b|\b{_re.escape(mn)}\s+(\d{{1,2}})\b', tl
            )
            if m:
                day = int(m.group(1) or m.group(2))
                year = today.year
                try:
                    candidate = _dt.date(year, mnum, day)
                    # If date is in the past this year, assume next year
                    if candidate < today:
                        candidate = _dt.date(year + 1, mnum, day)
                    return candidate.isoformat()
                except ValueError:
                    pass
        return None

    async def _extract_datetime(self, text: str) -> dict:
        import datetime as _dt
        IST = _dt.timezone(_dt.timedelta(hours=5, minutes=30))
        today_obj = _dt.datetime.now(tz=IST).date()
        today    = today_obj.isoformat()
        tomorrow = (today_obj + _dt.timedelta(days=1)).isoformat()
        day_after = (today_obj + _dt.timedelta(days=2)).isoformat()

        # Python pre-parse: aaj/kal/parso + DD-Month resolved without LLM
        preparse_date = self._preparse_date(text, today_obj)

        # Weekday map: English + romanized + Devanagari (Sarvam STT may output any)
        weekday_en  = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        weekday_rom = ["Somwar", "Mangalwar", "Budhwar", "Guruwar", "Shukrawar", "Shaniwar", "Itvaar"]
        weekday_dev = ["सोमवार", "मंगलवार", "बुधवार", "गुरुवार", "शुक्रवार", "शनिवार", "रविवार"]
        weekday_map: dict[str, str] = {}
        for idx in range(7):
            days_ahead = (idx - today_obj.weekday()) % 7 or 7  # never 0 (=today)
            target = (today_obj + _dt.timedelta(days=days_ahead)).isoformat()
            weekday_map[weekday_en[idx]]  = target
            weekday_map[weekday_rom[idx]] = target
            weekday_map[weekday_dev[idx]] = target
        weekday_map["इतवार"] = weekday_map["Sunday"]  # colloquial Sunday

        weekday_examples = "; ".join(
            f"'{weekday_en[i]}'='{weekday_rom[i]}'='{weekday_dev[i]}'->{weekday_map[weekday_en[i]]}"
            for i in range(7)
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
                            "Return date as YYYY-MM-DD ONLY if a date is explicitly mentioned — otherwise return null. "
                            "Return time as HH:MM ONLY if a time is explicitly mentioned — otherwise return null. "
                            'Return JSON: {"date": "YYYY-MM-DD or null", "time": "HH:MM or null"}. '
                            "The input may be Hindi (Devanagari), English, or a mix. "
                            "Relative dates: "
                            "'aaj'/'आज'=today, 'kal'/'कल'=tomorrow, 'parso'/'परसों'=day after tomorrow, "
                            "'agle'/'अगले'/'agli'/'अगली' X=next X. "
                            f"Weekday names always mean the NEXT upcoming occurrence: {weekday_examples}. "
                            "Time words: 'subah'/'सुबह'=morning (AM), 'dopahar'/'दोपहर'=noon (12:00-15:00), "
                            "'shaam'/'शाम'=evening (add 12 if hour<8, e.g. shaam 5=17:00), "
                            "'raat'/'रात'=night (add 12 if hour<8). "
                            "No qualifier: if hour 1-8 assume PM (add 12). If hour 9-12 assume AM. "
                            "'baje'/'बजे'/'bje' means o'clock. Examples: '2 bje'->14:00, '3 बजे'->15:00, '10 बजे'->10:00. "
                            "'sawa'/'सवा'=+15min (सवा 3=3:15), 'saadhe'/'साढ़े'=+30min (साढ़े 3=3:30). "
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
            # Python preparse always wins for date — deterministic beats LLM
            if preparse_date:
                date = preparse_date
            return {"date": date, "time": time}
        except Exception as e:
            print(f"[LLM Error] _extract_datetime: {e}")
            return {"date": preparse_date, "time": None}

