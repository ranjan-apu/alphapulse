"""
Session level lifecycle management and warm-start rebuild.

Implements deterministic level state transitions (Section 9.4.7):
- ACTIVE → TESTED → REJECTED / BROKEN → FLIPPED → INVALIDATED / EXPIRED

And warm-start session map rebuild (Section 14.1):
- Replay session_events to rebuild session_levels state
- Restore portfolio and position state from last snapshot
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np


class LevelState(str, Enum):
    ACTIVE = "ACTIVE"
    TESTED = "TESTED"
    REJECTED = "REJECTED"
    BROKEN = "BROKEN"
    FLIPPED_SUPPORT = "FLIPPED_SUPPORT"
    FLIPPED_RESISTANCE = "FLIPPED_RESISTANCE"
    INVALIDATED = "INVALIDATED"
    EXPIRED = "EXPIRED"


@dataclass
class SessionLevel:
    """A price level tracked during the session with lifecycle state."""
    level_id: str
    price: float
    level_type: str  # 'support', 'resistance', 'vwap', 'poc', 'vah', 'val',
                     # 'swing_high', 'swing_low', 'opening_range_high', 'opening_range_low'
    state: LevelState = LevelState.ACTIVE
    strength: int = 0
    first_identified: datetime = field(default_factory=lambda: datetime.now().astimezone())
    last_updated: datetime = field(default_factory=lambda: datetime.now().astimezone())


@dataclass
class SessionEvent:
    """A recorded event for session-level lifecycle."""
    event_id: str
    session_id: str
    event_time: datetime
    event_type: str  # LEVEL_IDENTIFIED, LEVEL_TESTED, LEVEL_BROKEN, etc.
    event_data: dict
    created_at: datetime = field(default_factory=lambda: datetime.now().astimezone())


class LevelLifecycleManager:
    """
    Manages deterministic lifecycle transitions for session levels.

    Transitions (Section 9.4.7):

    ACTIVE → TESTED
        Candle high/low reaches within (atr * 0.3) of the level.

    TESTED → REJECTED
        Price touched level, then next completed candle closed away
        by at least body_size.

    TESTED → BROKEN
        Completed candle body closed beyond level by at least (atr * 0.2).
        Wick-only violation does NOT trigger BROKEN.

    BROKEN → FLIPPED_SUPPORT
        Two completed candles above broken resistance with closes above.

    BROKEN → FLIPPED_RESISTANCE
        Two completed candles below broken support with closes below.

    Any → INVALIDATED
        Three consecutive candles ignore the level (no reaction within atr*0.3),
        OR level violated twice in same session.

    Any → EXPIRED
        Session phase moves to FORCED_SQUAREOFF or CLOSED.
    """

    def __init__(self, atr: float = 0.0):
        self.atr = atr
        self._candles_since_test: Dict[str, int] = {}  # level_id -> count
        self._violation_count: Dict[str, int] = {}      # level_id -> count

    def process_candle(
        self,
        candle: dict,  # {'open', 'high', 'low', 'close', 'volume'}
        candle_time: datetime,
        levels: List[SessionLevel],
    ) -> List[SessionEvent]:
        """
        Process one candle against all active levels and return events.

        Args:
            candle: OHLCV dict for the completed candle
            candle_time: Timestamp of the candle
            levels: List of SessionLevel to check

        Returns:
            List of SessionEvent triggered by this candle
        """
        events = []
        high = candle["high"]
        low = candle["low"]
        close = candle["close"]
        open_p = candle["open"]
        body = abs(close - open_p)
        atr_threshold = self.atr * 0.3 if self.atr > 0 else (high * 0.001)

        for level in levels:
            if level.state in (LevelState.EXPIRED, LevelState.INVALIDATED):
                continue

            level_id = level.level_id
            price = level.price

            # Calculate distance from candle to level
            dist_high = abs(high - price)
            dist_low = abs(low - price)
            min_dist = min(dist_high, dist_low)

            # ---- ACTIVE → TESTED ----
            if level.state == LevelState.ACTIVE:
                if min_dist <= atr_threshold:
                    level.state = LevelState.TESTED
                    level.last_updated = candle_time
                    self._candles_since_test[level_id] = 0
                    events.append(SessionEvent(
                        event_id=f"evt_{level_id}_tested_{int(candle_time.timestamp())}",
                        session_id="",
                        event_time=candle_time,
                        event_type="LEVEL_TESTED",
                        event_data={
                            "level_id": level_id,
                            "level_price": price,
                            "level_type": level.level_type,
                            "candle_high": high,
                            "candle_low": low,
                            "candle_close": close,
                        },
                    ))

            # ---- TESTED → REJECTED or BROKEN ----
            elif level.state == LevelState.TESTED:
                self._candles_since_test[level_id] = self._candles_since_test.get(level_id, 0) + 1

                # Check REJECTED: price touched then closed away
                if min_dist <= atr_threshold:
                    # Price is near the level again
                    if level.level_type in ("resistance", "swing_high"):
                        # For resistance: close below = rejection
                        if close < price - atr_threshold:
                            level.state = LevelState.REJECTED
                            level.last_updated = candle_time
                            events.append(SessionEvent(
                                event_id=f"evt_{level_id}_rejected_{int(candle_time.timestamp())}",
                                session_id="",
                                event_time=candle_time,
                                event_type="LEVEL_REJECTED",
                                event_data={
                                    "level_id": level_id,
                                    "level_price": price,
                                    "reason": "Price rejected from resistance",
                                },
                            ))
                    elif level.level_type in ("support", "swing_low"):
                        # For support: close above = rejection (price bounced)
                        if close > price + atr_threshold:
                            level.state = LevelState.REJECTED
                            level.last_updated = candle_time
                            events.append(SessionEvent(
                                event_id=f"evt_{level_id}_rejected_{int(candle_time.timestamp())}",
                                session_id="",
                                event_time=candle_time,
                                event_type="LEVEL_REJECTED",
                                event_data={
                                    "level_id": level_id,
                                    "level_price": price,
                                    "reason": "Price rejected from support",
                                },
                            ))

                # Check BROKEN: body close beyond level by atr * 0.2
                break_threshold = self.atr * 0.2 if self.atr > 0 else (price * 0.002)

                if level.level_type in ("resistance", "swing_high"):
                    if close > price + break_threshold:
                        level.state = LevelState.BROKEN
                        level.last_updated = candle_time
                        self._violation_count[level_id] = 0
                        events.append(SessionEvent(
                            event_id=f"evt_{level_id}_broken_{int(candle_time.timestamp())}",
                            session_id="",
                            event_time=candle_time,
                            event_type="LEVEL_BROKEN",
                            event_data={
                                "level_id": level_id,
                                "level_price": price,
                                "break_side": "above",
                                "candle_close": close,
                            },
                        ))
                elif level.level_type in ("support", "swing_low"):
                    if close < price - break_threshold:
                        level.state = LevelState.BROKEN
                        level.last_updated = candle_time
                        self._violation_count[level_id] = 0
                        events.append(SessionEvent(
                            event_id=f"evt_{level_id}_broken_{int(candle_time.timestamp())}",
                            session_id="",
                            event_time=candle_time,
                            event_type="LEVEL_BROKEN",
                            event_data={
                                "level_id": level_id,
                                "level_price": price,
                                "break_side": "below",
                                "candle_close": close,
                            },
                        ))

            # ---- BROKEN → FLIPPED ----
            elif level.state == LevelState.BROKEN:
                self._violation_count[level_id] = self._violation_count.get(level_id, 0) + 1

                if level.level_type in ("resistance", "swing_high"):
                    # Two closes above broken resistance = flipped to support
                    if close > price and self._violation_count[level_id] >= 2:
                        level.state = LevelState.FLIPPED_SUPPORT
                        level.level_type = "support"
                        level.last_updated = candle_time
                        events.append(SessionEvent(
                            event_id=f"evt_{level_id}_flipped_{int(candle_time.timestamp())}",
                            session_id="",
                            event_time=candle_time,
                            event_type="LEVEL_FLIPPED",
                            event_data={
                                "level_id": level_id,
                                "flipped_to": "support",
                                "original_type": "resistance",
                            },
                        ))
                elif level.level_type in ("support", "swing_low"):
                    if close < price and self._violation_count[level_id] >= 2:
                        level.state = LevelState.FLIPPED_RESISTANCE
                        level.level_type = "resistance"
                        level.last_updated = candle_time
                        events.append(SessionEvent(
                            event_id=f"evt_{level_id}_flipped_{int(candle_time.timestamp())}",
                            session_id="",
                            event_time=candle_time,
                            event_type="LEVEL_FLIPPED",
                            event_data={
                                "level_id": level_id,
                                "flipped_to": "resistance",
                                "original_type": "support",
                            },
                        ))

            # ---- INVALIDATED check (applies to any state) ----
            if level.state not in (LevelState.EXPIRED, LevelState.INVALIDATED):
                candles_no_reaction = self._candles_since_test.get(level_id, 0)
                # Three consecutive candles with no reaction
                if candles_no_reaction >= 3:
                    level.state = LevelState.INVALIDATED
                    events.append(SessionEvent(
                        event_id=f"evt_{level_id}_invalidated_{int(candle_time.timestamp())}",
                        session_id="",
                        event_time=candle_time,
                        event_type="LEVEL_INVALIDATED",
                        event_data={
                            "level_id": level_id,
                            "reason": "No reaction for 3 consecutive candles",
                        },
                    ))

        return events

    def expire_all_levels(
        self,
        levels: List[SessionLevel],
        candle_time: datetime,
    ) -> List[SessionEvent]:
        """Mark all non-expired levels as EXPIRED (at session end)."""
        events = []
        for level in levels:
            if level.state not in (LevelState.EXPIRED, LevelState.INVALIDATED):
                level.state = LevelState.EXPIRED
                level.last_updated = candle_time
                events.append(SessionEvent(
                    event_id=f"evt_{level.level_id}_expired_{int(candle_time.timestamp())}",
                    session_id="",
                    event_time=candle_time,
                    event_type="LEVEL_EXPIRED",
                    event_data={"level_id": level.level_id, "reason": "Session end"},
                ))
        return events


class SessionRebuilder:
    """
    Rebuilds session state after a mid-session restart (warm-start).

    Procedure (Section 14.1):
    1. Load all session_events for the day, ordered by timestamp
    2. Replay events sequentially to rebuild session_levels state
    3. For gaps since the last event, replay through candle data
       (deterministic level-check logic, NOT LLM/tool decisions)
    4. Restore portfolio state from last portfolio_snapshot
    5. Restore position state from positions where active = true
    """

    def __init__(self, atr: float = 0.0):
        self.lifecycle = LevelLifecycleManager(atr=atr)

    def rebuild_from_events(
        self,
        events: List[SessionEvent],
        candle_data: pd.DataFrame,
        last_event_time: datetime,
    ) -> Tuple[List[SessionLevel], List[SessionEvent]]:
        """
        Rebuild session levels by replaying events and missing candles.

        Args:
            events: Recorded session events (from Postgres)
            candle_data: OHLCV DataFrame for the session
            last_event_time: Timestamp of the last recorded event

        Returns:
            (levels, new_events) - rebuilt levels and any newly generated events
        """
        levels: Dict[str, SessionLevel] = {}
        all_events = list(events)

        # Step 1: Replay recorded events
        for event in sorted(events, key=lambda e: e.event_time):
            if event.event_type == "LEVEL_IDENTIFIED":
                data = event.event_data
                level = SessionLevel(
                    level_id=data.get("level_id", event.event_id),
                    price=data["level_price"],
                    level_type=data["level_type"],
                )
                levels[level.level_id] = level

            elif event.event_type == "LEVEL_TESTED":
                level_id = event.event_data.get("level_id")
                if level_id in levels:
                    levels[level_id].state = LevelState.TESTED

            elif event.event_type == "LEVEL_REJECTED":
                level_id = event.event_data.get("level_id")
                if level_id in levels:
                    levels[level_id].state = LevelState.REJECTED

            elif event.event_type == "LEVEL_BROKEN":
                level_id = event.event_data.get("level_id")
                if level_id in levels:
                    levels[level_id].state = LevelState.BROKEN

            elif event.event_type == "LEVEL_FLIPPED":
                level_id = event.event_data.get("level_id")
                if level_id in levels:
                    flipped_to = event.event_data.get("flipped_to", "support")
                    if flipped_to == "support":
                        levels[level_id].state = LevelState.FLIPPED_SUPPORT
                        levels[level_id].level_type = "support"
                    else:
                        levels[level_id].state = LevelState.FLIPPED_RESISTANCE
                        levels[level_id].level_type = "resistance"

            elif event.event_type == "LEVEL_INVALIDATED":
                level_id = event.event_data.get("level_id")
                if level_id in levels:
                    levels[level_id].state = LevelState.INVALIDATED

            elif event.event_type == "LEVEL_EXPIRED":
                level_id = event.event_data.get("level_id")
                if level_id in levels:
                    levels[level_id].state = LevelState.EXPIRED

        # Step 2: Replay missing candles since last event
        level_list = list(levels.values())
        if not candle_data.empty and last_event_time:
            missing_candles = candle_data[candle_data.index > last_event_time]
            for idx, row in missing_candles.iterrows():
                candle = {
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": int(row["volume"]),
                }
                new_events = self.lifecycle.process_candle(candle, idx, level_list)
                all_events.extend(new_events)

        # Remove expired/invalidated levels from active map
        active_levels = [
            lvl for lvl in level_list
            if lvl.state not in (LevelState.EXPIRED, LevelState.INVALIDATED)
        ]

        return active_levels, all_events

    def warm_start_portfolio_state(
        self,
        portfolio_snapshots: List[dict],
        positions: List[dict],
    ) -> dict:
        """
        Reconstruct portfolio state from snapshots and positions.

        Returns dict with:
        - portfolio_state: reconstructed PortfolioState
        - open_position: reconstructed OpenPosition or None
        """
        # Get the latest snapshot
        latest_snapshot = None
        if portfolio_snapshots:
            latest_snapshot = sorted(
                portfolio_snapshots,
                key=lambda s: s.get("timestamp", ""),
                reverse=True,
            )[0]

        # Get the active position
        active_position = None
        for pos in positions:
            if pos.get("active", False):
                active_position = pos
                break

        return {
            "portfolio_snapshot": latest_snapshot,
            "open_position": active_position,
            "can_trade": (
                latest_snapshot is not None
                and latest_snapshot.get("cash_available", 0) > 0
                and latest_snapshot.get("daily_loss_used", 0)
                < latest_snapshot.get("max_daily_loss", float("inf"))
            ),
        }

    @staticmethod
    def classify_startup_type(
        has_prior_events: bool,
        has_open_position: bool,
        is_mid_session: bool,
    ) -> str:
        """
        Classify startup type (Section 14.1).

        Returns:
            'COLD_START_BEFORE_SESSION'
            'WARM_START_MID_SESSION'
            'RECOVERY_WITH_OPEN_POSITION'
            'FAILED_RECOVERY'
        """
        if not is_mid_session:
            return "COLD_START_BEFORE_SESSION"

        if has_prior_events:
            if has_open_position:
                return "RECOVERY_WITH_OPEN_POSITION"
            return "WARM_START_MID_SESSION"

        if has_open_position:
            return "RECOVERY_WITH_OPEN_POSITION"

        return "FAILED_RECOVERY"
