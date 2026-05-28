"""
Cooldown and re-entry policy for the trading agent.
Deterministic behavioral brakes to prevent revenge trading and
repeated same-level entries.

Cooldown Events:
- AFTER_STOP_LOSS       -> cooldown N candles
- AFTER_TARGET_HIT      -> cooldown M candles (usually shorter)
- AFTER_AGENT_EXIT      -> cooldown N candles
- AFTER_REJECTED_SIGNAL -> optional small cooldown
- AFTER_SCHEMA_FAILURE  -> no trade for that decision only
"""
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional


class CooldownReason(str, Enum):
    AFTER_STOP_LOSS = "AFTER_STOP_LOSS"
    AFTER_TARGET_HIT = "AFTER_TARGET_HIT"
    AFTER_AGENT_EXIT = "AFTER_AGENT_EXIT"
    AFTER_REJECTED_SIGNAL = "AFTER_REJECTED_SIGNAL"
    AFTER_SCHEMA_FAILURE = "AFTER_SCHEMA_FAILURE"
    SAME_LEVEL_REPEATED = "SAME_LEVEL_REPEATED"
    MAX_SAME_DIRECTION_LOSSES = "MAX_SAME_DIRECTION_LOSSES"


@dataclass
class CooldownConfig:
    """Configuration for cooldown and re-entry policy."""
    # Candle-based cooldowns (candles of the decision interval, e.g., 15min candles)
    cooldown_after_stop_candles: int = 2
    cooldown_after_exit_candles: int = 2
    cooldown_after_target_candles: int = 1
    cooldown_after_rejection_candles: int = 0
    cooldown_after_schema_failure_candles: int = 0

    # Re-entry rules
    same_direction_reentry_candles: int = 3
    max_attempts_per_level_per_day: int = 2
    max_same_direction_losses_per_day: int = 2

    # Decision interval in minutes (used to convert candles to time)
    decision_interval_minutes: int = 15


@dataclass
class TradeLock:
    """A lock preventing trades in specific conditions."""
    lock_id: str
    run_id: str
    symbol: str
    direction: Optional[str]
    level_zone: Optional[str]
    reason: CooldownReason
    expires_at: datetime
    created_at: datetime = field(default_factory=lambda: datetime.now().astimezone())


class CooldownController:
    """
    Manages cooldown periods and re-entry locks.

    Tracks stop/target/EXIT events, blocks immediate revenge trades,
    blocks repeated same-level trades, enforces max trades/day and
    max losses per direction.
    """

    def __init__(self, config: Optional[CooldownConfig] = None):
        self.config = config or CooldownConfig()
        self._locks: List[TradeLock] = []
        self._same_direction_losses: Dict[str, int] = {}  # direction -> count
        self._level_attempts: Dict[str, int] = {}          # level_zone -> count
        self._trades_today: int = 0
        self._current_date = None

    def reset_for_day(self, date):
        """Reset daily counters for a new trading day."""
        self._same_direction_losses.clear()
        self._level_attempts.clear()
        self._trades_today = 0
        self._current_date = date
        # Clear expired locks
        self._cleanup_expired_locks()

    def add_lock(
        self,
        run_id: str,
        symbol: str,
        reason: CooldownReason,
        direction: Optional[str] = None,
        level_zone: Optional[str] = None,
        custom_candles: Optional[int] = None,
    ) -> TradeLock:
        """Add a trade lock that prevents new entries until it expires."""
        # Determine duration in candles
        candle_duration = custom_candles or self._get_default_candles(reason)

        # Convert candles to time
        duration_minutes = candle_duration * self.config.decision_interval_minutes

        import uuid
        lock_id = f"lock_{uuid.uuid4().hex[:12]}"

        lock = TradeLock(
            lock_id=lock_id,
            run_id=run_id,
            symbol=symbol,
            direction=direction,
            level_zone=level_zone,
            reason=reason,
            expires_at=datetime.now().astimezone() + timedelta(minutes=duration_minutes),
        )

        self._locks.append(lock)

        # Track level attempts
        if level_zone:
            self._level_attempts[level_zone] = self._level_attempts.get(level_zone, 0) + 1

        return lock

    def can_open_position(
        self,
        direction: str,
        level_zone: Optional[str] = None,
    ) -> tuple:
        """
        Check if a new position can be opened.

        Returns:
            (allowed: bool, reason: Optional[str])
        """
        # Clean expired locks
        self._cleanup_expired_locks()

        now = datetime.now().astimezone()

        # Check for active locks
        for lock in self._locks:
            if lock.expires_at > now:
                # Direction-specific lock
                if lock.direction and lock.direction == direction:
                    return False, f"Cooldown active: {lock.reason.value} for {direction}"
                # General lock (no direction filter)
                if lock.direction is None:
                    return False, f"Cooldown active: {lock.reason.value}"

        # Check level attempts
        if level_zone and self._level_attempts.get(level_zone, 0) >= self.config.max_attempts_per_level_per_day:
            return False, f"Max attempts ({self.config.max_attempts_per_level_per_day}) at level zone '{level_zone}'"

        # Check same-direction losses
        losses_this_direction = self._same_direction_losses.get(direction, 0)
        if losses_this_direction >= self.config.max_same_direction_losses_per_day:
            return False, f"Max same-direction losses ({self.config.max_same_direction_losses_per_day}) for {direction}"

        return True, None

    def record_loss(self, direction: str, level_zone: Optional[str] = None):
        """Record a losing trade."""
        self._same_direction_losses[direction] = self._same_direction_losses.get(direction, 0) + 1
        self._trades_today += 1

        # Auto-add lock after stop loss
        if self.config.cooldown_after_stop_candles > 0:
            import uuid
            lock = TradeLock(
                lock_id=f"lock_{uuid.uuid4().hex[:12]}",
                run_id="",
                symbol="",
                direction=direction,
                level_zone=level_zone,
                reason=CooldownReason.AFTER_STOP_LOSS,
                expires_at=datetime.now().astimezone() + timedelta(
                    minutes=self.config.cooldown_after_stop_candles * self.config.decision_interval_minutes
                ),
            )
            self._locks.append(lock)

    def record_win(self, direction: str = None):
        """Record a winning trade."""
        self._trades_today += 1

        # Optional shorter cooldown after target hit
        if self.config.cooldown_after_target_candles > 0:
            import uuid
            lock = TradeLock(
                lock_id=f"lock_{uuid.uuid4().hex[:12]}",
                run_id="",
                symbol="",
                direction=direction,
                level_zone=None,
                reason=CooldownReason.AFTER_TARGET_HIT,
                expires_at=datetime.now().astimezone() + timedelta(
                    minutes=self.config.cooldown_after_target_candles * self.config.decision_interval_minutes
                ),
            )
            self._locks.append(lock)

    def record_exit(self, direction: str = None):
        """Record an agent-initiated exit (thesis failure)."""
        self._trades_today += 1

    def record_rejection(self):
        """Record a rejected signal."""
        if self.config.cooldown_after_rejection_candles > 0:
            import uuid
            lock = TradeLock(
                lock_id=f"lock_{uuid.uuid4().hex[:12]}",
                run_id="",
                symbol="",
                direction=None,
                level_zone=None,
                reason=CooldownReason.AFTER_REJECTED_SIGNAL,
                expires_at=datetime.now().astimezone() + timedelta(
                    minutes=self.config.cooldown_after_rejection_candles * self.config.decision_interval_minutes
                ),
            )
            self._locks.append(lock)

    def get_state(self) -> dict:
        """Get current cooldown state for the agent prompt."""
        self._cleanup_expired_locks()
        now = datetime.now().astimezone()

        active_locks = [l for l in self._locks if l.expires_at > now]

        return {
            "active_locks": [
                {
                    "reason": lock.reason.value,
                    "direction": lock.direction,
                    "level_zone": lock.level_zone,
                    "expires_in_minutes": round((lock.expires_at - now).total_seconds() / 60, 1),
                }
                for lock in active_locks
            ],
            "trades_today": self._trades_today,
            "max_trades_per_day": 5,  # Could be configurable
            "same_direction_losses": dict(self._same_direction_losses),
            "max_same_direction_losses": self.config.max_same_direction_losses_per_day,
            "level_attempts": dict(self._level_attempts),
            "max_level_attempts": self.config.max_attempts_per_level_per_day,
        }

    def _get_default_candles(self, reason: CooldownReason) -> int:
        """Get default cooldown duration in candles for a reason."""
        mapping = {
            CooldownReason.AFTER_STOP_LOSS: self.config.cooldown_after_stop_candles,
            CooldownReason.AFTER_AGENT_EXIT: self.config.cooldown_after_exit_candles,
            CooldownReason.AFTER_TARGET_HIT: self.config.cooldown_after_target_candles,
            CooldownReason.AFTER_REJECTED_SIGNAL: self.config.cooldown_after_rejection_candles,
            CooldownReason.AFTER_SCHEMA_FAILURE: self.config.cooldown_after_schema_failure_candles,
            CooldownReason.SAME_LEVEL_REPEATED: self.config.same_direction_reentry_candles,
            CooldownReason.MAX_SAME_DIRECTION_LOSSES: self.config.same_direction_reentry_candles * 2,
        }
        return mapping.get(reason, 2)

    def _cleanup_expired_locks(self):
        """Remove locks that have expired."""
        now = datetime.now().astimezone()
        self._locks = [l for l in self._locks if l.expires_at > now]
