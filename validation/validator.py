"""
Trade validator: applies POC trade validation rules to agent signals.
Rejects signals that don't meet 2:1 net R:R, capital cap, and session constraints.
"""
import math
from typing import Dict, Optional, Tuple
from datetime import datetime

from config import config


class TradeValidator:
    """
    Validates BUY/SELL signals against POC rules:
    - Must have entry, stop, target, direction
    - Quantity = floor(30000 / entry)
    - Net Target Profit >= 2 * Gross Risk
    - Must be actionable within capital cap
    - Must close by session end
    """

    REJECTION_REASONS = [
        "REJECTED_RISK_REWARD",
        "REJECTED_MISSING_LEVELS",
        "REJECTED_CHARGES",
        "REJECTED_NO_PRICE_ACTION_THESIS",
        "REJECTED_SESSION_END_CONSTRAINT",
        "REJECTED_PRICE_EXCEEDS_CAPITAL",
        "REJECTED_POSITION_OPEN",
    ]

    def __init__(self, capital_cap: float = None, total_charges: float = None):
        self.capital_cap = capital_cap or config.CAPITAL_CAP
        self.total_charges = total_charges or config.TOTAL_ORDER_CHARGES
        self.min_rr = config.MIN_REWARD_TO_RISK

    def validate(
        self,
        signal: Dict,
        decision_time: datetime,
        session_end: datetime,
    ) -> Tuple[bool, Optional[str], Optional[Dict]]:
        """
        Validate a BUY or SELL signal.

        Returns:
            (is_valid, rejection_reason, sizing_info)
        """
        action = signal.get("action", "HOLD")

        # HOLD always valid
        if action == "HOLD":
            return True, None, None

        # Must be BUY or SELL
        if action not in ("BUY", "SELL"):
            return False, "REJECTED_MISSING_LEVELS", None

        # Must have entry, stop, target
        entry = signal.get("entry")
        stop = signal.get("stop")
        target = signal.get("target")

        if entry is None or stop is None or target is None:
            return False, "REJECTED_MISSING_LEVELS", None

        entry = float(entry)
        stop = float(stop)
        target = float(target)

        # Validate basic logic
        if entry <= 0 or stop <= 0 or target <= 0:
            return False, "REJECTED_MISSING_LEVELS", None

        # Stop must be on the correct side
        if action == "BUY":
            if stop >= entry:
                return False, "REJECTED_MISSING_LEVELS", None
            if target <= entry:
                return False, "REJECTED_MISSING_LEVELS", None
        else:  # SELL
            if stop <= entry:
                return False, "REJECTED_MISSING_LEVELS", None
            if target >= entry:
                return False, "REJECTED_MISSING_LEVELS", None

        # ---- Sizing ----
        quantity = math.floor(self.capital_cap / entry)
        if quantity <= 0:
            return False, "REJECTED_PRICE_EXCEEDS_CAPITAL", {
                "entry": entry,
                "stop": stop,
                "target": target,
                "quantity": 0,
                "deployed_capital": 0,
                "error": f"Entry price {entry} exceeds capital cap {self.capital_cap}",
            }

        deployed_capital = quantity * entry

        # ---- P&L Math ----
        if action == "BUY":
            gross_risk = quantity * (entry - stop)
            gross_target_profit = quantity * (target - entry)
        else:  # SELL
            gross_risk = quantity * (stop - entry)
            gross_target_profit = quantity * (entry - target)

        if gross_risk <= 0:
            return False, "REJECTED_MISSING_LEVELS", None

        total_charges = self.total_charges
        net_target_profit = gross_target_profit - total_charges

        net_rr = net_target_profit / gross_risk
        gross_rr = abs(target - entry) / abs(entry - stop)

        sizing_info = {
            "entry": float(round(entry, 2)),
            "stop": float(round(stop, 2)),
            "target": float(round(target, 2)),
            "direction": action,
            "quantity": quantity,
            "deployed_capital": float(round(deployed_capital, 2)),
            "total_charges": float(round(total_charges, 2)),
            "gross_risk": float(round(gross_risk, 2)),
            "gross_target_profit": float(round(gross_target_profit, 2)),
            "net_target_profit": float(round(net_target_profit, 2)),
            "net_reward_to_risk": float(round(net_rr, 4)),
            "gross_reward_to_risk": float(round(gross_rr, 4)),
            "meets_2_to_1": net_target_profit >= (self.min_rr * gross_risk),
        }

        # ---- Checks ----

        # Check price-action thesis
        dart = signal.get("dart", {})
        if not dart.get("trigger") or dart["trigger"] in ("unclear", "", None):
            return False, "REJECTED_NO_PRICE_ACTION_THESIS", sizing_info

        # Check 2:1 net R:R
        if net_target_profit < (self.min_rr * gross_risk):
            # Check if gross R:R was >= 2 but charges dragged it down
            if gross_target_profit >= (self.min_rr * gross_risk):
                return False, "REJECTED_CHARGES", sizing_info
            return False, "REJECTED_RISK_REWARD", sizing_info

        # Check session end constraint
        # Is there enough time for the trade to resolve?
        # Rough check: at least 30 min (6 candles) left
        time_remaining_minutes = (session_end - decision_time).total_seconds() / 60
        if time_remaining_minutes < 30:
            return False, "REJECTED_SESSION_END_CONSTRAINT", sizing_info

        return True, None, sizing_info


def validate_signal(
    signal: Dict,
    decision_time: datetime,
    session_end: datetime,
) -> Dict:
    """
    Convenience function: validate and return enriched signal record.
    """
    validator = TradeValidator()
    is_valid, rejection, sizing = validator.validate(signal, decision_time, session_end)

    result = {
        "original_signal": signal,
        "is_valid": is_valid,
        "rejection_reason": rejection,
        "sizing": sizing,
        "action": signal.get("action", "HOLD"),
    }

    if not is_valid and signal.get("action") in ("BUY", "SELL"):
        result["action"] = rejection or "REJECTED_UNKNOWN"

    return result
