"""Time-based rules: is a zone restricted at a given moment?

A zone schedule has:
    default     status outside every listed window ("restricted" or "allowed")
    allowed     list of windows when the zone is allowed
    restricted  list of windows when the zone is restricted

Precedence (safety first):  restricted window  >  allowed window  >  default.

A window is {days, start, end} in the CAMPUS time zone, start inclusive, end exclusive.
If end <= start the window crosses midnight, and `days` refers to the day it STARTS:
    days [fri], 22:00–06:00  covers Friday 22:00 → Saturday 06:00.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time
from zoneinfo import ZoneInfo

DAY_NAMES = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")  # index = datetime.weekday()
STATUSES = ("restricted", "allowed")


@dataclass(frozen=True)
class TimeWindow:
    days: frozenset[int]   # 0 = Monday … 6 = Sunday (start day of the window)
    start: time
    end: time              # time(0, 0) with end_is_midnight=True means "24:00"
    end_is_midnight: bool = False

    @property
    def crosses_midnight(self) -> bool:
        return not self.end_is_midnight and self.end <= self.start

    def contains(self, local: datetime) -> bool:
        t, day = local.time(), local.weekday()
        if self.end_is_midnight:                       # e.g. 18:00–24:00
            return day in self.days and t >= self.start
        if not self.crosses_midnight:                  # e.g. 06:00–20:00
            return day in self.days and self.start <= t < self.end
        # Overnight, e.g. 22:00–06:00:
        #   evening part belongs to today's window, early-morning part to yesterday's.
        yesterday = (day - 1) % 7
        return (day in self.days and t >= self.start) or (yesterday in self.days and t < self.end)


@dataclass(frozen=True)
class Schedule:
    default: str = "restricted"
    allowed: tuple[TimeWindow, ...] = field(default_factory=tuple)
    restricted: tuple[TimeWindow, ...] = field(default_factory=tuple)

    def status_at(self, when: datetime, tz: ZoneInfo) -> str:
        """Return "restricted" or "allowed" for a timezone-AWARE datetime."""
        if when.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware (naive datetimes are ambiguous)")
        local = when.astimezone(tz)
        if any(w.contains(local) for w in self.restricted):
            return "restricted"
        if any(w.contains(local) for w in self.allowed):
            return "allowed"
        return self.default


def parse_hhmm(value: str, *, allow_24: bool) -> tuple[time, bool]:
    """'HH:MM' → (time, is_24h). Raises ValueError with a readable message."""
    if not isinstance(value, str) or len(value.split(":")) != 2:
        raise ValueError(f"time must be a string 'HH:MM', got {value!r}")
    hh, mm = value.split(":")
    if not (hh.isdigit() and mm.isdigit()):
        raise ValueError(f"time must be 'HH:MM', got {value!r}")
    h, m = int(hh), int(mm)
    if allow_24 and h == 24 and m == 0:
        return time(0, 0), True
    if not (0 <= h <= 23 and 0 <= m <= 59):
        raise ValueError(f"invalid time {value!r}")
    return time(h, m), False


def parse_window(raw: dict) -> TimeWindow:
    if not isinstance(raw, dict):
        raise ValueError(f"window must be a mapping with days/start/end, got {raw!r}")
    missing = {"days", "start", "end"} - raw.keys()
    if missing:
        raise ValueError(f"window missing {sorted(missing)}")
    days = raw["days"]
    if not isinstance(days, list) or not days:
        raise ValueError("window 'days' must be a non-empty list, e.g. [mon, tue]")
    idx = set()
    for d in days:
        name = str(d).strip().lower()[:3]
        if name not in DAY_NAMES:
            raise ValueError(f"unknown day {d!r}; use mon, tue, wed, thu, fri, sat, sun")
        idx.add(DAY_NAMES.index(name))
    start, _ = parse_hhmm(raw["start"], allow_24=False)
    end, end_24 = parse_hhmm(raw["end"], allow_24=True)
    if not end_24 and start == end:
        raise ValueError(f"window start and end are equal ({raw['start']}); use 00:00–24:00 for a full day")
    return TimeWindow(frozenset(idx), start, end, end_24)


def parse_schedule(raw: dict | None) -> Schedule:
    raw = raw or {}
    default = raw.get("default", "restricted")
    if default not in STATUSES:
        raise ValueError(f"schedule default must be one of {STATUSES}, got {default!r}")
    return Schedule(
        default=default,
        allowed=tuple(parse_window(w) for w in raw.get("allowed", []) or []),
        restricted=tuple(parse_window(w) for w in raw.get("restricted", []) or []),
    )

