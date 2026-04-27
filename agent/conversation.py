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
import asyncio
from enum import Enum
from groq import AsyncGroq

from agent.prompts import CLINIC_HOURS, CLINIC_NAME, DOCTOR_NAME, get_today_iso
from services.calendar_service import CalendarService


class State(Enum):
    WAIT_NAME         = "wait_name"
    WAIT_NAME_CONFIRM = "wait_name_confirm" # read name back, await yes/no
    WAIT_CITY         = "wait_city"         # ask which city patient is from
    WAIT_DATETIME     = "wait_datetime"    # need both date and time
    WAIT_DATE         = "wait_date"        # have time, need date
    WAIT_TIME         = "wait_time"        # have date, need time
    WAIT_CONFIRM      = "wait_confirm"     # slot available, awaiting yes/no
    WAIT_SLOT_CHOICE  = "wait_slot_choice" # slot taken, picking alternative
    DONE              = "done"


# ---------------------------------------------------------------------------
# Fixed Hindi response templates — no LLM variation, always consistent
# ---------------------------------------------------------------------------

def _fmt_time(time_24: str) -> str:
    """Convert HH:MM (24h) to 12-hour for Hindi speech: '15:00' → '3', '17:30' → '5:30'"""
    try:
        h, m = map(int, time_24.split(":"))
        h12 = h % 12 or 12
        if m == 0:
            return str(h12)
        return f"{h12}:{m:02d}"
    except Exception:
        return time_24

def _human_date(date_iso: str) -> str:
    """Convert YYYY-MM-DD to Hindi human-readable: 'सोमवार, 27 April'"""
    import datetime as _dt
    try:
        d = _dt.date.fromisoformat(date_iso)
        day_names = ["सोमवार", "मंगलवार", "बुधवार", "गुरुवार", "शुक्रवार", "शनिवार", "रविवार"]
        month_names = ["January","February","March","April","May","June",
                       "July","August","September","October","November","December"]
        return f"{day_names[d.weekday()]}, {d.day} {month_names[d.month-1]}"
    except Exception:
        return date_iso

def _greeting_with_hours(name: str) -> str:
    return (
        f"{name} जी, नमस्ते! "
        f"Clinic का समय: {CLINIC_HOURS}. "
        "आप किस दिन और समय appointment लेना चाहेंगे?"
    )

def _ask_time(date: str) -> str:
    return f"{_human_date(date)} ठीक है। किस समय आना चाहेंगे?"

def _ask_date(time: str) -> str:
    return f"{_fmt_time(time)} बजे ठीक है। कौनसा दिन आना चाहेंगे?"

def _slot_available_confirm(name: str, date: str, time: str) -> str:
    return f"{name} जी, {_human_date(date)} को {_fmt_time(time)} बजे slot available है। Confirm करूँ?"

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
        "subah", "सुबह", "morning", "ਸਵੇਰੇ", "ਸਵੇਰ",
        "dopahar", "दोपहर", "duphar", "noon", "ਦੁਪਹਿਰ",
        "shaam", "शाम", "sham", "evening", "ਸ਼ਾਮ",
        "raat", "रात", "night", "ਰਾਤ",
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
        nearby_str = " और ".join(_fmt_time(s) for s in parts)
        return f"{_fmt_time(time)} बजे slot नहीं है। पास में: {nearby_str} बजे। कौनसा ठीक रहेगा?"
    return f"{_fmt_time(time)} बजे slot नहीं है। कोई और समय बताएं।"


def _slot_taken(date: str, time: str, suggestions: list) -> str:
    # fallback — unused, kept for reference
    slots = ", ".join(suggestions)
    return (
        f"माफ़ी चाहते हैं, {date} को {_fmt_time(time)} बजे का slot available नहीं है। "
        f"इन slots में से चुनें: {', '.join(_fmt_time(s) for s in suggestions)} बजे।"
    )

def _no_slots_on_date(date: str, is_today: bool = False, next_slot: dict | None = None) -> str:
    next_hint = (
        f" अगला available slot: {_human_date(next_slot['date'])} {_fmt_time(next_slot['time'])} बजे।"
        if next_slot else ""
    )
    if is_today:
        return f"आज कोई slot नहीं है।{next_hint} कोई और दिन बताएं।"
    return f"{_human_date(date)} को कोई slot नहीं है।{next_hint} कोई और तारीख़ बताएं।"

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
        f"बिल्कुल! {name} जी, {human_date} को {_fmt_time(time)} बजे appointment confirm हो गई। धन्यवाद!"
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
    def __init__(self, caller_phone: str = ""):
        self.client = AsyncGroq(api_key=os.environ["GROQ_API_KEY"])
        self.model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        self.calendar = CalendarService()

        self.state = State.WAIT_NAME

        # Collected data
        self.patient_name: str = ""
        self.patient_city: str = ""
        self.patient_phone: str = caller_phone  # from Twilio 'From' field
        self._city_retried: bool = False        # allow one re-ask
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
        elif self.state == State.WAIT_NAME_CONFIRM:
            return await self._handle_name_confirm(user_input)
        elif self.state == State.WAIT_CITY:
            return await self._handle_city(user_input)
        elif self.state == State.WAIT_DATETIME:
            # "next available / agla slot" check first (most specific)
            if self._is_next_slot_query(user_input):
                return await self._handle_next_slot_query(user_input)
            if await self._is_slot_query(user_input):
                return await self._handle_slot_query(user_input)
            return await self._handle_datetime(user_input)
        elif self.state == State.WAIT_DATE:
            if self._is_next_slot_query(user_input):
                return await self._handle_next_slot_query(user_input)
            if await self._is_slot_query(user_input):
                return await self._handle_slot_query(user_input)
            return await self._handle_date_only(user_input)
        elif self.state == State.WAIT_TIME:
            if self._is_next_slot_query(user_input):
                return await self._handle_next_slot_query(user_input)
            if await self._is_slot_query(user_input):
                return await self._handle_slot_query(user_input)
            return await self._handle_time_only(user_input)
        elif self.state == State.WAIT_CONFIRM:
            if self._is_next_slot_query(user_input):
                return await self._handle_next_slot_query(user_input)
            if await self._is_slot_query(user_input):
                return await self._handle_slot_query(user_input)
            return await self._handle_confirm(user_input)
        elif self.state == State.WAIT_SLOT_CHOICE:
            if self._is_next_slot_query(user_input):
                return await self._handle_next_slot_query(user_input)
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
        self.state = State.WAIT_NAME_CONFIRM
        return f"{name} — क्या मैं आपका नाम सही ले रही हूँ?"

    async def _handle_name_confirm(self, text: str) -> str:
        tl = text.lower().strip()
        _deny = ("no", "nahi", "nahin", "नहीं", "galat", "गलत", "wrong", "nope",
                 "ਨਹੀਂ", "ਗਲਤ", "ਨਹੀ")
        if any(d in tl for d in _deny):
            # May have corrected name inline: "नहीं, मेरा नाम Raju है"
            corrected = await self._extract_name(text)
            if corrected and corrected.lower() != self.patient_name.lower():
                self.patient_name = corrected
                self.state = State.WAIT_NAME_CONFIRM
                return f"{corrected} — क्या मैं आपका नाम सही ले रही हूँ?"
            self.patient_name = ""
            self.state = State.WAIT_NAME
            return "कृपया अपना सही नाम बताएं।"

        _affirm = ("yes", "haan", "ha", "हाँ", "हां", "ji", "जी", "bilkul", "sahi",
                   "सही", "correct", "theek", "ठीक", "ਹਾਂ", "ਹਾਂਜੀ", "ਜੀ", "ਸਹੀ", "ਠੀਕ", "ਬਿਲਕੁਲ")
        is_affirm = any(a in tl for a in _affirm)

        # If user also mentioned date/time (e.g. "हाँ, kal 4 baje"), capture it
        _dt_hints = ("kal", "aaj", "parso", "कल", "आज", "परसों", "tomorrow", "monday",
                     "tuesday", "wednesday", "thursday", "friday", "saturday", "baje",
                     "बजे", "subah", "shaam", "सुबह", "शाम", "tarikh", "तारीख़", "din", "दिन")
        if is_affirm and any(h in tl for h in _dt_hints):
            dt = await self._extract_datetime(text)
            d, t = dt.get("date"), dt.get("time")
            if d or t:
                if d: self.date = d
                if t: self.time = t
                self.state = State.WAIT_DATETIME
                if d and t:
                    if self._is_past_date(d):
                        return "यह तारीख़ गुज़र चुकी है। कृपया आज या आने वाली तारीख़ बताएं।"
                    return await self._check_slot()
                elif d:
                    self.state = State.WAIT_TIME
                    return _ask_time(d)
                else:
                    self.state = State.WAIT_DATE
                    return _ask_date(t)

        if is_affirm:
            self.state = State.WAIT_CITY
            return "आप किस शहर से हैं?"

        # Caller may have corrected the name directly
        corrected = await self._extract_name(text)
        if corrected and corrected.lower() != self.patient_name.lower():
            self.patient_name = corrected
            self.state = State.WAIT_NAME_CONFIRM
            return f"{corrected} — क्या मैं आपका नाम सही ले रही हूँ?"

        # Treat anything else as confirmation
        self.state = State.WAIT_CITY
        return "आप किस शहर से हैं?"

    async def _handle_city(self, text: str) -> str:
        _filler = ("theek", "thik", "okay", "ok", "haan", "ji", "ha ", "ठीक", "हाँ", "जी", "हां")
        tl = text.lower().strip()
        is_filler = not tl or len(tl) <= 2 or any(tl.startswith(f) for f in _filler)

        city = None if is_filler else await self._extract_city(text)

        if city:
            self.patient_city = city
            self.state = State.WAIT_DATETIME
            return _greeting_with_hours(self.patient_name)

        # No city — check if patient gave date/time instead (skipped city question)
        _dt_hints = ("kal", "aaj", "parso", "कल", "आज", "परसों", "tomorrow", "monday",
                     "tuesday", "wednesday", "thursday", "friday", "saturday", "baje",
                     "बजे", "subah", "shaam", "सुबह", "शाम", "tarikh", "तारीख़")
        if not is_filler and any(h in tl for h in _dt_hints):
            dt = await self._extract_datetime(text)
            d, t = dt.get("date"), dt.get("time")
            if d or t:
                if d: self.date = d
                if t: self.time = t
                self.state = State.WAIT_DATETIME
                if d and t:
                    if self._is_past_date(d):
                        return "यह तारीख़ गुज़र चुकी है। कृपया आज या आने वाली तारीख़ बताएं।"
                    return await self._check_slot()
                elif d:
                    if self._is_past_date(d):
                        return "यह तारीख़ गुज़र चुकी है। कृपया आज या आने वाली तारीख़ बताएं।"
                    import datetime as _dt2
                    if _dt2.date.fromisoformat(d).weekday() == 6:
                        self.date = ""
                        return "इतवार को clinic बंद रहती है। कोई और दिन चुनें, Monday से Saturday।"
                    self.state = State.WAIT_TIME
                    return _ask_time(d)
                else:
                    self.state = State.WAIT_DATE
                    return _ask_date(t)

        if not self._city_retried:
            self._city_retried = True
            # Skip city immediately on first failure — don't block flow
            self.state = State.WAIT_DATETIME
            return _greeting_with_hours(self.patient_name)
        # Second failure — already skipped above, shouldn't reach here
        self.state = State.WAIT_DATETIME
        return _greeting_with_hours(self.patient_name)

    async def _extract_city(self, text: str) -> str:
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Extract the city name from the text. "
                            "Input may be Punjabi (Gurmukhi), Hindi (Devanagari), English, or romanized. "
                            "Punjab cities and towns: Ludhiana, Amritsar, Jalandhar, Patiala, Bathinda, Mohali, "
                            "Phagwara, Hoshiarpur, Gurdaspur, Pathankot, Firozpur, Moga, Barnala, Sangrur, "
                            "Faridkot, Muktsar, Fazilka, Kapurthala, Ropar, Nawanshahr, SAS Nagar, Fatehgarh Sahib, "
                            "Tarn Taran, Mansa, Malerkotla, Khanna, Rajpura, Zirakpur, Derabassi, Morinda, "
                            "Sirhind, Abohar, Anandpur Sahib, Nangal, Rupnagar, Sunam, Rampura Phul, Zira, "
                            "Nakodar, Sultanpur Lodhi, Qadian, Dera Baba Nanak, Kartarpur, Batala, Dinanagar, "
                            "Mukerian, Dasuya, Garhshankar, Balachaur, Banga, Nawanshahr, Phillaur, Rahon, "
                            "Samrala, Machhiwara, Doraha, Payal, Maloud, Kharar, Kurali, Chamkaur Sahib, "
                            "Chandigarh, Delhi, Mumbai, Bangalore, Hyderabad, Chennai, Kolkata, Pune. "
                            "Return city in proper English (e.g. 'Ludhiana', 'Patiala'). "
                            'Return only JSON: {"city": "CityName"} or {"city": null} if no city found.'
                        ),
                    },
                    {"role": "user", "content": text},
                ],
                response_format={"type": "json_object"},
                temperature=0,
                max_tokens=20,
            )
            data = json.loads(response.choices[0].message.content)
            return data.get("city") or None
        except Exception as e:
            print(f"[LLM Error] _extract_city: {e}")
            return None

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

        # Bare affirmative with no date/time info — nudge gently instead of repeating full prompt
        _bare_affirm = ("हाँ", "हां", "हाँ।", "हां।", "haan", "ha", "ok", "okay", "ठीक", "जी", "ji")
        if t_lower.strip().rstrip(".!।") in _bare_affirm or t_lower.strip() in _bare_affirm:
            return "किस दिन और कितने बजे appointment चाहिए?"

        dt = await self._extract_datetime(text)
        date, time = dt.get("date"), dt.get("time")

        if date and self._is_past_date(date):
            return "यह तारीख़ गुज़र चुकी है। कृपया आज या आने वाली तारीख़ बताएं।"

        if date and time:
            self.date, self.time = date, time
            return await self._check_slot()
        elif date:
            self.date = date
            # Early Sunday check — no point asking for time
            import datetime as _dt2
            if _dt2.date.fromisoformat(date).weekday() == 6:
                self.date = ""
                return "इतवार को clinic बंद रहती है। कोई और दिन चुनें, Monday से Saturday।"
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
        # Detect "kaun se din" type availability queries
        tl = text.lower()
        if any(kw in tl for kw in ("कौन से दिन", "कौनसे दिन", "kaun se din", "kaunse din", "kon se din",
                                    "ਕਿਹੜੇ ਦਿਨ", "ਕਿਹੜਾ ਦਿਨ", "ਕਿਹੜੇ ਦਿਨਾਂ")):
            return f"हम Monday से Saturday, {CLINIC_HOURS} तक available हैं। किस दिन आना चाहेंगे?"
        dt = await self._extract_datetime(text)
        date = dt.get("date")
        if not date:
            return "कृपया तारीख़ बताएं, जैसे 'कल', 'परसों', या '25 April'।"
        if self._is_past_date(date):
            return "यह तारीख़ गुज़र चुकी है। कृपया आज या आने वाली तारीख़ बताएं।"
        # Early Sunday check
        import datetime as _dt2
        if _dt2.date.fromisoformat(date).weekday() == 6:
            return "इतवार को clinic बंद रहती है। कोई और दिन चुनें, Monday से Saturday।"
        self.date = date
        return await self._check_slot()

    async def _handle_time_only(self, text: str) -> str:
        # Detect clinic/doctor hours query
        tl = text.lower()
        if any(kw in tl for kw in (
            "kab aayenge", "kab available", "kitne baje", "timing", "time kya",
            "कितने बजे", "कब आएंगे", "कब available", "क्या समय", "टाइमिंग"
        )):
            return f"Clinic का समय {CLINIC_HOURS} है। आप किस समय आना चाहेंगे?"

        # "कोई भी" / "जो भी available हो" — pick first available slot
        if any(kw in tl for kw in ("koi bhi", "jo bhi", "kuch bhi", "any", "कोई भी", "जो भी", "कुछ भी", "ਕੋਈ ਵੀ")):
            if self.date:
                slots = self.calendar.get_available_slots(self.date)
                if slots and slots != "SUNDAY":
                    avail = [s for s in slots if not self._is_past_slot(self.date, s)]
                    if avail:
                        self.time = avail[0]
                        self.available_slots = avail
                        self.state = State.WAIT_CONFIRM
                        return _slot_available_confirm(self.patient_name, self.date, self.time)
            return "कृपया एक समय बताएं, जैसे 'सुबह 10 बजे' या 'शाम 3 बजे'।"
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
        # Out-of-hours check: clinic ends at 18:00, last slot is 17:30
        # If requested time >= 18:00, steer toward last available slot immediately
        try:
            h, m = map(int, time.split(":"))
            if h >= 18:
                # Fetch slots now to offer the last one directly
                if self.date:
                    slots = self.calendar.get_available_slots(self.date)
                    if slots and slots != "SUNDAY":
                        from services.calendar_service import CalendarService as _CS
                        last = slots[-1]
                        self.time = last
                        self.available_slots = slots
                        self.state = State.WAIT_CONFIRM
                        return f"Clinic {time} बजे बंद हो जाती है। आखिरी slot {last} बजे है — confirm करूँ?"
        except Exception:
            pass
        self.time = time
        return await self._check_slot()

    async def _handle_slot_choice(self, text: str) -> str:
        tl = text.lower()

        # "कोई भी" or "जो available हो" — pick first
        if any(kw in tl for kw in ("koi bhi", "jo bhi", "kuch bhi", "any", "कोई भी", "जो भी", "कुछ भी", "ਕੋਈ ਵੀ")):
            if self.available_slots:
                avail = [s for s in self.available_slots if not self._is_past_slot(self.date, s)]
                if avail:
                    self.time = avail[0]
                    self.state = State.WAIT_CONFIRM
                    return _slot_available_confirm(self.patient_name, self.date, self.time)

        # Bare denial — ask what they'd prefer
        import re as _re
        if _re.search(r'^(nahi|nahin|no|नहीं|नहि|ना|ਨਹੀਂ)[\.!।]?$', tl.strip()):
            if self.available_slots:
                first, last = self.available_slots[0], self.available_slots[-1]
                return f"{first} से {last} बजे के बीच में कौनसा समय ठीक रहेगा?"
            self.state = State.WAIT_DATETIME
            return "कोई और दिन और समय बताएं।"

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
            first, last = self.available_slots[0], self.available_slots[-1]
            return f"{first} से {last} बजे तक slots हैं। कौनसा समय ठीक रहेगा?"

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
        affirm = bool(_re.search(
            r"(haan|hnji|ji\b|ha\b|yes|bilkul|theek|ok\b|okay|confirm|karo|krdo|kar\s*do|kar\s*de"
            r"|karde|kardo|zaroor|sahi|done|book\b|dijiye|kijiye|kar\s*du|kar\s*dun|kar\s*doon"
            r"|हाँ|हां|हा\b|जी|हजी|हन्जी|बिल्कुल|ठीक|करो|कर\s*दो|करदो|कर\s*दे|करदे|कर\s*दीजिए"
            r"|कर\s*दूं|कर\s*दूँ|कर\s*दू"
            r"|ज़रूर|जरूर|सही|दीजिए|कीजिए|बुक"
            r"|ਹਾਂ|ਹਾਂਜੀ|ਜੀ|ਸਹੀ|ਠੀਕ|ਬਿਲਕੁਲ|ਕਰੋ|ਜ਼ਰੂਰ|ਕਰ\s*ਦੋ|ਭਰ\s*ਦੋ|ਕਰ\s*ਦੇ|ਕਰਦੋ|ਕਰਦੇ)",
            text_lower
        ))
        deny = bool(_re.search(
            r"(nahi|nahin|no\b|nope|cancel|mat\b|naa\b|नहीं|नहि|मत\b|ना\b|ਨਹੀਂ|ਨਹੀ|ਨਾ\b)",
            text_lower
        ))

        if affirm and not deny:
            return await self._book_now()

        if deny:
            # Check if deny includes a new time (same date, different time)
            dt = await self._extract_datetime(text)
            new_date = dt.get("date")
            new_time = dt.get("time")

            if new_time and not _has_time_qualifier(text):
                try:
                    h, m = map(int, new_time.split(":"))
                    if h < 7:
                        new_time = f"{h + 12:02d}:{m:02d}"
                except Exception:
                    pass

            if new_date and new_date != self.date:
                # Completely new date+time
                if self._is_past_date(new_date):
                    return "यह तारीख़ गुज़र चुकी है। कृपया आज या आने वाली तारीख़ बताएं।"
                self.date = new_date
                self.time = new_time or ""
                return await self._check_slot()
            elif new_time:
                # Same date, new time
                self.time = new_time
                return await self._check_slot()
            else:
                # Plain denial — ask fresh
                self.date = ""
                self.time = ""
                self.state = State.WAIT_DATETIME
                return "ठीक है। कोई और दिन और समय बताएं।"

        # Ambiguous — re-confirm
        return f"{self.patient_name} जी, {_human_date(self.date)} को {self.time} बजे — क्या confirm करूँ?"

    async def _book_now(self) -> str:
        result = self.calendar.book_appointment(
            patient_name=self.patient_name,
            patient_phone=self.patient_phone,
            date_str=self.date,
            time_str=self.time,
            city=self.patient_city,
        )
        if result.get("success"):
            self.state = State.DONE
            confirmation = _booking_confirmed(self.patient_name, self.date, self.time)
            # Fire-and-forget SMS — don't block or fail the call if SMS fails
            asyncio.create_task(self._send_sms_confirmation())
            return confirmation
        else:
            self.date = ""
            self.time = ""
            self.state = State.WAIT_DATETIME
            return _booking_failed()

    async def _send_sms_confirmation(self) -> None:
        """Send booking confirmation SMS to the caller via Twilio REST API."""
        to_number = self.patient_phone
        if not to_number:
            return
        account_sid = os.getenv("TWILIO_ACCOUNT_SID")
        auth_token  = os.getenv("TWILIO_AUTH_TOKEN")
        from_number = os.getenv("TWILIO_PHONE_NUMBER")  # e.g. +1XXXXXXXXXX
        if not all([account_sid, auth_token, from_number]):
            print("[SMS] Skipped — TWILIO_ACCOUNT_SID / AUTH_TOKEN / PHONE_NUMBER not set")
            return

        import datetime as _dt
        try:
            d = _dt.date.fromisoformat(self.date)
            day_names = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
            month_names = ["January","February","March","April","May","June",
                           "July","August","September","October","November","December"]
            human_date = f"{d.day} {month_names[d.month-1]} ({day_names[d.weekday()]})"
        except Exception:
            human_date = self.date

        body = (
            f"Dear {self.patient_name}, your appointment at {CLINIC_NAME} "
            f"is confirmed for {human_date} at {self.time}. "
            f"Doctor: {DOCTOR_NAME}. Thank you!"
        )

        url = f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Messages.json"
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    url,
                    auth=(account_sid, auth_token),
                    data={"From": from_number, "To": to_number, "Body": body},
                    timeout=10,
                )
            if resp.status_code == 201:
                print(f"[SMS] Sent to {to_number}")
            else:
                print(f"[SMS] Failed ({resp.status_code}): {resp.text[:200]}")
        except Exception as e:
            print(f"[SMS] Error: {e}")

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
                        saved = self.date
                        self.date = ""
                        self.state = State.WAIT_DATETIME
                        ns = self.calendar.get_next_available_after(saved)
                        ns_hint = f" अगला slot: {_human_date(ns['date'])} {ns['time']} बजे।" if ns else ""
                        return f"आज के बाकी सारे slots निकल चुके हैं।{ns_hint} कोई और दिन बताएं।"
                else:
                    saved = self.date
                    self.date = ""
                    self.time = ""
                    self.state = State.WAIT_DATETIME
                    ns = self.calendar.get_next_available_after(saved)
                    ns_hint = f" अगला slot: {_human_date(ns['date'])} {ns['time']} बजे।" if ns else ""
                    return f"आज के बाकी सारे slots निकल चुके हैं।{ns_hint} कोई और दिन बताएं।"

        if not slots:
            import datetime as _dt
            IST = _dt.timezone(_dt.timedelta(hours=5, minutes=30))
            saved_date = self.date
            is_today = saved_date == _dt.datetime.now(tz=IST).date().isoformat()
            next_avail = self.calendar.get_next_available_after(saved_date)
            self.date = ""
            self.time = ""
            self.state = State.WAIT_DATETIME
            return _no_slots_on_date(saved_date, is_today=is_today, next_slot=next_avail)

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
                f"{_human_date(self.date)} को {first} बजे से {last} बजे तक slots available हैं। "
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

        # Relative day words (romanized + Devanagari + Gurmukhi + English)
        if _re.search(r'\baaj\b|\btoday\b|आज|ਅੱਜ', tl):
            return today.isoformat()
        # 'kal' — check not part of longer Latin word (e.g. 'calendar')
        if _re.search(r'(?<![a-z])kal(?![a-z])|\btomorrow\b|कल|ਕੱਲ੍ਹ|ਕੱਲ', tl):
            return (today + _dt.timedelta(days=1)).isoformat()
        if _re.search(r'\bparso\b|\bparson\b|परसों|परसो\b|ਪਰਸੋਂ|ਪਰਸੋ', tl):
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


    def _is_next_slot_query(self, text: str) -> bool:
        """
        Detect queries asking for the next/earliest available slot overall,
        regardless of date — e.g. "jo bhi agla slot hai", "sabse jaldi ka slot",
        "book next available", "koi bhi slot de do".
        """
        t = text.lower()
        keywords = [
            # Romanized / English
            "agla slot", "agla available", "agli slot", "next available", "next slot",
            "first available", "earliest slot", "earliest available",
            "jaldi se jaldi", "jaldse jaldi", "jaldi wala slot", "jaldi ka slot",
            "sabse jaldi", "sabse paas wala slot", "sabse pahle",
            "jo bhi available", "jo bhi avb", "jo vi available", "jo vi slot",
            "jo bhi slot", "koi bhi slot dedo", "koi bhi slot do",
            "any slot", "any available slot", "kab se slot hai", "kab slot milega",
            "abhi ka slot", "abhi available", "abhi slot",
            # Hindi (Devanagari)
            "अगला slot", "अगला available", "अगली slot",
            "जो भी available", "जो भी slot", "कोई भी slot दो", "कोई भी slot दे दो",
            "सबसे जल्दी", "सबसे पास", "जल्दी वाला slot",
            "पहला available", "पहला slot",
            # Punjabi (Gurmukhi)
            "ਅਗਲਾ slot", "ਜੋ ਵੀ available", "ਕੋਈ ਵੀ slot",
            "ਜਲਦੀ ਵਾਲਾ slot", "ਸਭ ਤੋਂ ਜਲਦੀ",
        ]
        return any(kw in t for kw in keywords)

    async def _handle_next_slot_query(self, text: str) -> str:
        """Find the overall earliest available slot and offer/confirm it."""
        ns = self.calendar.get_next_available_slot()
        if not ns:
            return "अगले 30 दिनों में कोई slot available नहीं है। कृपया बाद में call करें।"

        self.date = ns["date"]
        self.time = ns["time"]
        self.state = State.WAIT_CONFIRM

        # Detect booking intent: "book kar do", "fix kar do", etc.
        t = text.lower()
        book_intent = any(kw in t for kw in (
            "book", "kar do", "kardo", "kar de", "karde", "krdo", "krde",
            "fix", "confirm", "laga do", "laga de",
            "बुक", "कर दो", "करदो", "कर दे", "करदे", "लगा दो", "लगा दे",
            "ਬੁੱਕ", "ਕਰ ਦੋ", "ਕਰਦੋ", "ਲਾ ਦੇ",
        ))

        human = _human_date(ns["date"])
        if book_intent:
            return (
                f"अगला available slot {human} को {ns['time']} बजे है। "
                "Confirm करूँ?"
            )
        return (
            f"अगला available slot {human} को {ns['time']} बजे है। "
            "क्या इसे book करूँ?"
        )

    async def _is_slot_query(self, text: str) -> bool:
        """Detect if the caller is asking about slot availability rather than booking."""
        t = text.lower()
        keywords = [
            "kya slot", "koi slot", "slot available", "slots available",
            "kab available", "slots hain", "slots hai", "slots bata",
            "khali slot", "koi jagah", "slot khali",
            "kaun kaun", "kaun-kaun", "kon kon", "kon-kon",
            "कौन से slot", "कौनसे slot", "कौन कौन", "कौन-कौन",
            "कौनसे स्लॉट", "खाली स्लॉट",
            "कोई slot", "खाली slot", "slot खाली", "slot है", "slot हैं",
            "kab hai slot", "kab milega slot", "kaun sa slot", "kon sa slot",
            "कब मिलेगा", "कब available", "कौनसा slot", "slot मिलेगा", "slot बताएं",
            "कोईन सी slot", "available slot", "which slot", "what slot",
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
            next_avail = self.calendar.get_next_available_after(query_date)
            self.date = ""
            self.state = State.WAIT_DATETIME
            return _no_slots_on_date(query_date, is_today=is_today, next_slot=next_avail)

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
                ns = self.calendar.get_next_available_after(query_date)
                ns_hint = f" अगला slot: {_human_date(ns['date'])} {ns['time']} बजे।" if ns else ""
                self.date = ""
                self.state = State.WAIT_DATETIME
                return f"आज के बाकी सारे slots निकल चुके हैं।{ns_hint} कोई और दिन बताएं।"
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
            # No time given — list slots and ask
            self.state = State.WAIT_TIME
            first, last = slots[0], slots[-1]
            return (
                f"{_human_date(query_date)} को {first} बजे से {last} बजे तक slots available हैं। "
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
                            "The input may be Hindi (Devanagari), Punjabi (Gurmukhi), English, or a mix. "
                            "It can be a single name or full name — both are valid. "
                            "Capitalize it properly using Latin script (e.g. 'Gaganjot', 'Bhagwanjot', 'Priya'). "
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

        # Weekday map: English + romanized + Devanagari + Gurmukhi (Sarvam STT may output any)
        weekday_en  = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        weekday_rom = ["Somwar", "Mangalwar", "Budhwar", "Guruwar", "Shukrawar", "Shaniwar", "Itvaar"]
        weekday_dev = ["सोमवार", "मंगलवार", "बुधवार", "गुरुवार", "शुक्रवार", "शनिवार", "रविवार"]
        weekday_gur = ["ਸੋਮਵਾਰ", "ਮੰਗਲਵਾਰ", "ਬੁੱਧਵਾਰ", "ਵੀਰਵਾਰ", "ਸ਼ੁੱਕਰਵਾਰ", "ਸ਼ਨੀਵਾਰ", "ਐਤਵਾਰ"]
        weekday_map: dict[str, str] = {}
        for idx in range(7):
            days_ahead = (idx - today_obj.weekday()) % 7 or 7  # never 0 (=today)
            target = (today_obj + _dt.timedelta(days=days_ahead)).isoformat()
            weekday_map[weekday_en[idx]]  = target
            weekday_map[weekday_rom[idx]] = target
            weekday_map[weekday_dev[idx]] = target
            weekday_map[weekday_gur[idx]] = target
        weekday_map["इतवार"] = weekday_map["Sunday"]  # colloquial Hindi Sunday
        weekday_map["ਇਤਵਾਰ"] = weekday_map["Sunday"]  # colloquial Punjabi Sunday

        weekday_examples = "; ".join(
            f"'{weekday_en[i]}'='{weekday_rom[i]}'='{weekday_dev[i]}'='{weekday_gur[i]}'->{weekday_map[weekday_en[i]]}"
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
                            "Extract date and/or time from Hindi/Punjabi/English text. "
                            "Return date as YYYY-MM-DD ONLY if a date is explicitly mentioned — otherwise return null. "
                            "Return time as HH:MM ONLY if a time is explicitly mentioned — otherwise return null. "
                            'Return JSON: {"date": "YYYY-MM-DD or null", "time": "HH:MM or null"}. '
                            "The input may be Hindi (Devanagari), Punjabi (Gurmukhi), English, or a mix. "
                            "Relative dates: "
                            "'aaj'/'आज'/'ਅੱਜ'=today, 'kal'/'कल'/'ਕੱਲ੍ਹ'=tomorrow, 'parso'/'परसों'/'ਪਰਸੋਂ'=day after tomorrow, "
                            "'agle'/'अगले'/'agli'/'अगली'/'ਅਗਲੇ' X=next X. "
                            f"Weekday names always mean the NEXT upcoming occurrence: {weekday_examples}. "
                            "Time words: 'subah'/'सुबह'/'ਸਵੇਰੇ'=morning (AM), 'dopahar'/'दोपहर'/'ਦੁਪਹਿਰ'=noon (12:00-15:00), "
                            "'shaam'/'शाम'/'ਸ਼ਾਮ'=evening (add 12 if hour<8, e.g. shaam 5=17:00), "
                            "'raat'/'रात'/'ਰਾਤ'=night (add 12 if hour<8). "
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

