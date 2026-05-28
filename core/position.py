"""
Position state tracker using Redis.
Tracks whether we have an open position, entry details,
and manages signal cooldown to avoid spam.
"""
import json
import time
import uuid
from typing import Optional, Dict, Any

from config import config


class PositionTracker:
    """
    Manages position state and signal gating.

    Uses Redis for persistence across runs; falls back to in-memory
    dict if Redis is unavailable.
    """

    KEY_POSITION = f"tsd:position:{config.SYMBOL}"
    KEY_LAST_SIGNAL = f"tsd:last_signal:{config.SYMBOL}"
    KEY_SIGNAL_COUNT = f"tsd:signal_count:{config.SYMBOL}"

    def __init__(self, redis_client=None):
        self.redis = redis_client
        self._memory = {}  # fallback

    def _get(self, key: str) -> Optional[str]:
        if self.redis:
            try:
                val = self.redis.get(key)
                return val.decode() if val else None
            except Exception:
                pass
        return self._memory.get(key)

    def _set(self, key: str, value: str, ttl: int = 86400):
        if self.redis:
            try:
                self.redis.setex(key, ttl, value)
                return
            except Exception:
                pass
        self._memory[key] = value

    def _delete(self, key: str):
        if self.redis:
            try:
                self.redis.delete(key)
                return
            except Exception:
                pass
        self._memory.pop(key, None)

    def reset(self):
        """Clear replay position/cooldown state for a fresh harness run."""
        self._delete(self.KEY_POSITION)
        self._delete(self.KEY_LAST_SIGNAL)
        self._delete(self.KEY_SIGNAL_COUNT)

    # ---- Position State ----

    def has_position(self) -> bool:
        """Check if we currently hold a position."""
        data = self._get(self.KEY_POSITION)
        if data:
            try:
                pos = json.loads(data)
                return pos.get("active", False)
            except Exception:
                pass
        return False

    def get_position(self) -> Optional[Dict]:
        """Get current position details."""
        data = self._get(self.KEY_POSITION)
        if data:
            try:
                return json.loads(data)
            except Exception:
                pass
        return None

    def open_position(self, entry_price: float, direction: str,
                      stop: float, target: float, quantity: int,
                      entry_time: str) -> Dict:
        """Record a new position."""
        pos = {
            "position_id": f"pos_{uuid.uuid4().hex[:12]}",
            "active": True,
            "direction": direction,
            "entry_price": entry_price,
            "stop": stop,
            "target": target,
            "quantity": quantity,
            "entry_time": entry_time,
            "opened_at_epoch": time.time(),
        }
        self._set(self.KEY_POSITION, json.dumps(pos))
        return pos

    def close_position(self, exit_price: float = None, reason: str = "manual"):
        """Close an open position."""
        existing = self.get_position() or {}
        existing["active"] = False
        existing["exit_price"] = exit_price
        existing["exit_reason"] = reason
        existing["closed_at_epoch"] = time.time()
        self._set(self.KEY_POSITION, json.dumps(existing))

    # ---- Signal Gating (cooldown) ----

    def should_evaluate(self, current_timestamp) -> bool:
        """
        Should we evaluate a new signal now?

        Rules:
        - If we have a position, evaluate every candle (to check stop/target)
        - If no position, enforce cooldown between entry evaluation
        - Don't evaluate in the first few candles of the session
        """
        # Always evaluate if we have a position (need to manage it)
        if self.has_position():
            return True

        # Check cooldown
        last_ts = self._get(self.KEY_LAST_SIGNAL)
        if last_ts:
            try:
                last = float(last_ts)
                elapsed = current_timestamp.timestamp() - last
                if elapsed < config.MIN_MINUTES_BETWEEN_SIGNALS * 60:
                    return False
            except (ValueError, TypeError):
                pass

        return True

    def record_evaluation(self, current_timestamp):
        """Record that we evaluated a signal at this time."""
        self._set(self.KEY_LAST_SIGNAL, str(current_timestamp.timestamp()))

    def increment_signal_count(self) -> int:
        """Increment and return the signal evaluation counter."""
        count = 0
        raw = self._get(self.KEY_SIGNAL_COUNT)
        if raw:
            try:
                count = int(raw)
            except (ValueError, TypeError):
                pass
        count += 1
        self._set(self.KEY_SIGNAL_COUNT, str(count))
        return count

    def summary(self) -> Dict[str, Any]:
        """Return position and signal summary."""
        return {
            "has_position": self.has_position(),
            "position": self.get_position(),
            "signal_count": int(self._get(self.KEY_SIGNAL_COUNT) or 0),
        }
