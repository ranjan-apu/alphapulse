"""
Market session controller for Indian equity markets.
Owns session boundaries, phase transitions, entry cutoff, and forced square-off.

Session Phases:
- PRE_OPEN:        before 09:15 IST
- OPENING_BUILD:   09:15-09:30
- ACTIVE_TRADING:  09:30-entry_cutoff
- MANAGEMENT_ONLY: entry_cutoff-squareoff_time
- FORCED_SQUAREOFF: squareoff_time-15:30
- CLOSED:          after 15:30
"""
from dataclasses import dataclass
from datetime import datetime, time
from enum import Enum
from typing import Optional

import pytz

IST = pytz.timezone("Asia/Kolkata")


class SessionPhase(str, Enum):
    PRE_OPEN = "PRE_OPEN"
    OPENING_BUILD = "OPENING_BUILD"
    ACTIVE_TRADING = "ACTIVE_TRADING"
    MANAGEMENT_ONLY = "MANAGEMENT_ONLY"
    FORCED_SQUAREOFF = "FORCED_SQUAREOFF"
    CLOSED = "CLOSED"


@dataclass
class SessionConfig:
    """Configuration for Indian equity session times."""
    session_start: time = time(9, 15)        # 09:15 IST
    decision_start: time = time(9, 30)       # 09:30 IST
    new_entry_cutoff: time = time(15, 0)     # 15:00 IST
    force_squareoff_time: time = time(15, 20)  # 15:20 IST
    session_end: time = time(15, 30)         # 15:30 IST
    minimum_minutes_for_new_trade: int = 45  # Must have 45 min left
    timezone: str = "Asia/Kolkata"


class MarketSessionController:
    """
    Controls session phases and enforces entry/exit rules.

    Rules:
    1. BUY/SELL only in ACTIVE_TRADING with enough time remaining
    2. MANAGEMENT_ONLY: no new entries, only HOLD/EXIT
    3. FORCED_SQUAREOFF: force close all positions
    4. CLOSED: no decisions at all
    """

    def __init__(self, config: Optional[SessionConfig] = None):
        self.config = config or SessionConfig()

    def get_phase(self, ts: datetime) -> SessionPhase:
        """
        Determine the session phase for a given timestamp.

        Args:
            ts: Timestamp (will be converted to IST if tz-naive)

        Returns:
            Current SessionPhase
        """
        if ts.tzinfo is None:
            ts = IST.localize(ts)
        else:
            ts = ts.astimezone(IST)

        t = ts.time()

        if t < self.config.session_start:
            return SessionPhase.PRE_OPEN
        elif t < self.config.decision_start:
            return SessionPhase.OPENING_BUILD
        elif t < self.config.new_entry_cutoff:
            return SessionPhase.ACTIVE_TRADING
        elif t < self.config.force_squareoff_time:
            return SessionPhase.MANAGEMENT_ONLY
        elif t <= self.config.session_end:
            return SessionPhase.FORCED_SQUAREOFF
        else:
            return SessionPhase.CLOSED

    def can_open_new_position(self, ts: datetime) -> bool:
        """Check if new positions can be opened at this time."""
        phase = self.get_phase(ts)
        if phase != SessionPhase.ACTIVE_TRADING:
            return False
        return self._has_sufficient_time_remaining(ts)

    def can_make_decision(self, ts: datetime) -> bool:
        """Check if decisions can be made at this time."""
        phase = self.get_phase(ts)
        return phase not in (SessionPhase.PRE_OPEN, SessionPhase.CLOSED)

    def must_square_off(self, ts: datetime) -> bool:
        """Check if positions must be force-closed at this time."""
        phase = self.get_phase(ts)
        return phase == SessionPhase.FORCED_SQUAREOFF

    def is_management_only(self, ts: datetime) -> bool:
        """Check if only position management is allowed (no new entries)."""
        phase = self.get_phase(ts)
        return phase in (SessionPhase.MANAGEMENT_ONLY, SessionPhase.FORCED_SQUAREOFF)

    def minutes_to_session_end(self, ts: datetime) -> float:
        """Minutes remaining until session end."""
        if ts.tzinfo is None:
            ts = IST.localize(ts)
        else:
            ts = ts.astimezone(IST)

        session_end_dt = ts.replace(
            hour=self.config.session_end.hour,
            minute=self.config.session_end.minute,
            second=0, microsecond=0
        )
        return (session_end_dt - ts).total_seconds() / 60

    def minutes_to_squareoff(self, ts: datetime) -> float:
        """Minutes remaining until forced square-off."""
        if ts.tzinfo is None:
            ts = IST.localize(ts)
        else:
            ts = ts.astimezone(IST)

        squareoff_dt = ts.replace(
            hour=self.config.force_squareoff_time.hour,
            minute=self.config.force_squareoff_time.minute,
            second=0, microsecond=0
        )
        return (squareoff_dt - ts).total_seconds() / 60

    def _has_sufficient_time_remaining(self, ts: datetime) -> bool:
        """Check if enough time remains for a new trade to resolve."""
        minutes_left = self.minutes_to_squareoff(ts)
        return minutes_left >= self.config.minimum_minutes_for_new_trade

    def get_session_summary(self, ts: datetime) -> dict:
        """Get a summary of session state at the given timestamp."""
        phase = self.get_phase(ts)
        return {
            "phase": phase.value,
            "can_open_new": self.can_open_new_position(ts),
            "can_decide": self.can_make_decision(ts),
            "must_square_off": self.must_square_off(ts),
            "management_only": self.is_management_only(ts),
            "minutes_to_session_end": round(self.minutes_to_session_end(ts), 1),
            "minutes_to_squareoff": round(self.minutes_to_squareoff(ts), 1),
        }

    def get_session_times(self, ts: datetime) -> dict:
        """Get all session time boundaries for the day."""
        if ts.tzinfo is None:
            ts = IST.localize(ts)
        else:
            ts = ts.astimezone(IST)

        base = ts.replace(hour=0, minute=0, second=0, microsecond=0)

        return {
            "session_start": base.replace(hour=self.config.session_start.hour, minute=self.config.session_start.minute),
            "decision_start": base.replace(hour=self.config.decision_start.hour, minute=self.config.decision_start.minute),
            "new_entry_cutoff": base.replace(hour=self.config.new_entry_cutoff.hour, minute=self.config.new_entry_cutoff.minute),
            "force_squareoff": base.replace(hour=self.config.force_squareoff_time.hour, minute=self.config.force_squareoff_time.minute),
            "session_end": base.replace(hour=self.config.session_end.hour, minute=self.config.session_end.minute),
        }
