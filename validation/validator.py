"""
Trade validator for the agent harness.

Enforces the plan's state-aware action vocabulary, session constraints,
risk-based sizing capped by capital ceiling, and post-charge reward:risk.
"""
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from config import config
from core.charges import EquityCashCharges, compute_charges
from core.position_sizing import PositionSizingConfig, compute_position_size
from core.session_controller import MarketSessionController


class TradeValidator:
    """
    Validates signals against the Phase 1 action/state contract:
    - Flat CNC state: BUY / SKIP
    - Open position: HOLD / EXIT
    - SELL entry is reserved for future MIS support
    - Entry signals use risk-based sizing capped by capital ceiling
    - Net R:R is computed after deterministic charges
    """

    REJECTION_REASONS = [
        "REJECTED_RISK_REWARD",
        "REJECTED_MISSING_LEVELS",
        "REJECTED_CHARGES",
        "REJECTED_NO_PRICE_ACTION_THESIS",
        "REJECTED_SESSION_END_CONSTRAINT",
        "REJECTED_PRICE_EXCEEDS_CAPITAL",
        "REJECTED_POSITION_OPEN",
        "REJECTED_INVALID_ACTION_FOR_STATE",
        "REJECTED_SELL_REQUIRES_MIS",
        "REJECTED_SCHEMA",
        "REJECTED_RISK_BUDGET",
    ]

    def __init__(
        self,
        capital_cap: float = None,
        total_charges: float = None,
        starting_capital: float = 100000.0,
        risk_budget_pct: float = 0.01,
        product_type: str = "CNC",
    ):
        self.capital_cap = capital_cap or config.CAPITAL_CAP
        self.total_charges = total_charges or config.TOTAL_ORDER_CHARGES
        self.min_rr = config.MIN_REWARD_TO_RISK
        self.starting_capital = starting_capital
        self.risk_budget_pct = risk_budget_pct
        self.product_type = product_type
        self.session_controller = MarketSessionController()

    def validate(
        self,
        signal: Dict,
        decision_time: datetime,
        session_end: datetime,
        has_open_position: bool = False,
        tool_calls: List[Dict] = None,
    ) -> Tuple[bool, Optional[str], Optional[Dict]]:
        """
        Validate an agent signal.

        Returns:
            (is_valid, rejection_reason, sizing_info)
        """
        action = str(signal.get("action", "SKIP")).upper()

        if action not in ("BUY", "SELL", "SKIP", "HOLD", "EXIT"):
            return False, "REJECTED_SCHEMA", None

        # ---- Required tool policy checks ----
        if tool_calls is not None:
            called_tools = {tc.get("tool") for tc in tool_calls}
            if action == "BUY":
                required = {
                    "get_portfolio_state",
                    "get_session_phase",
                    "compute_session_vwap",
                    "detect_market_structure",
                    "score_confluence",
                    "calculate_trade_math"
                }
                missing = required - called_tools
                if missing:
                    return False, "REJECTED_SCHEMA", {
                        "error": f"Missing required tool calls: {list(missing)}"
                    }
                
                # Breakout tag check
                checklist = signal.get("checklist", {})
                market_regime = str(checklist.get("market_regime", "")).lower()
                structure_state = str(checklist.get("structure_state", "")).lower()
                if "breakout" in market_regime or "breakout" in structure_state or "bos" in structure_state:
                    if "compute_volume_profile" not in called_tools:
                        return False, "REJECTED_SCHEMA", {
                            "error": "Breakout trades require compute_volume_profile"
                        }
            elif action in ("HOLD", "EXIT"):
                if "get_open_position" not in called_tools:
                    return False, "REJECTED_SCHEMA", {
                        "error": "HOLD/EXIT requires get_open_position tool"
                    }

        # ---- State/action semantics ----
        if has_open_position:
            if action == "SKIP":
                return False, "REJECTED_INVALID_ACTION_FOR_STATE", {
                    "error": "SKIP is invalid while a position is open; use HOLD or EXIT"
                }
            if action in ("BUY", "SELL"):
                return False, "REJECTED_POSITION_OPEN", {
                    "error": "Cannot open a new entry while a position is open"
                }
        else:
            if action in ("HOLD", "EXIT"):
                return False, "REJECTED_INVALID_ACTION_FOR_STATE", {
                    "error": f"{action} is invalid with no open position; use SKIP when flat"
                }

        if action == "SKIP":
            if not signal.get("reason"):
                return False, "REJECTED_SCHEMA", {"error": "SKIP requires reason"}
            return True, None, None

        if action == "HOLD":
            if not signal.get("position_id"):
                return False, "REJECTED_SCHEMA", {"error": "HOLD requires position_id"}
            return True, None, None

        if action == "EXIT":
            missing = [
                field for field in ("position_id", "exit_reason", "suggested_exit_price")
                if signal.get(field) in (None, "")
            ]
            if missing:
                return False, "REJECTED_SCHEMA", {"error": f"EXIT missing fields: {missing}"}
            return True, None, None

        if action == "SELL" and self.product_type.upper() == "CNC":
            return False, "REJECTED_SELL_REQUIRES_MIS", {
                "error": "SELL entry is reserved for MIS; CNC flat state only supports BUY/SKIP"
            }

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

        sizing_config = PositionSizingConfig(
            starting_capital=self.starting_capital,
            risk_budget_pct=self.risk_budget_pct,
            max_capital_per_trade=self.capital_cap,
            min_net_reward_risk=self.min_rr,
        )
        size_without_charges = compute_position_size(
            entry, stop, target, action, sizing_config, total_charges=0
        )
        quantity_for_charge = max(size_without_charges.quantity, 1)
        charge_result = compute_charges(
            EquityCashCharges(),
            action,
            quantity_for_charge,
            entry,
            target,
        )
        sizing_result = compute_position_size(
            entry,
            stop,
            target,
            action,
            sizing_config,
            total_charges=charge_result.total_charges,
        )

        sizing_info = {
            "entry": float(round(entry, 2)),
            "stop": float(round(stop, 2)),
            "target": float(round(target, 2)),
            "direction": action,
            "quantity": sizing_result.quantity,
            "deployed_capital": sizing_result.deployed_capital,
            "total_charges": charge_result.total_charges,
            "charge_breakdown": charge_result.breakdown,
            "gross_risk": sizing_result.gross_risk,
            "gross_target_profit": sizing_result.gross_reward,
            "net_target_profit": sizing_result.net_reward,
            "net_reward_to_risk": sizing_result.net_reward_risk,
            "gross_reward_to_risk": sizing_result.gross_reward_risk,
            "risk_budget": sizing_config.risk_budget,
            "risk_budget_used_pct": sizing_result.risk_budget_used,
            "capital_ceiling_hit": sizing_result.capital_ceiling_hit,
            "risk_budget_hit": sizing_result.risk_budget_hit,
            "meets_2_to_1": sizing_result.net_reward_risk >= self.min_rr,
            "warnings": sizing_result.warnings,
        }
        sizing_info["breakeven_points_per_share"] = charge_result.breakeven_points

        # ---- Checks ----

        # Check price-action thesis
        dart = signal.get("dart", {})
        if not dart.get("trigger") or dart["trigger"] in ("unclear", "", None):
            return False, "REJECTED_NO_PRICE_ACTION_THESIS", sizing_info

        if not sizing_result.actionable:
            sizing_info["errors"] = sizing_result.errors
            if any("exceeds ceiling" in e or "qty_capital" in e for e in sizing_result.errors):
                return False, "REJECTED_PRICE_EXCEEDS_CAPITAL", sizing_info
            if any("risk budget" in e.lower() for e in sizing_result.errors):
                return False, "REJECTED_RISK_BUDGET", sizing_info
            if sizing_result.gross_reward_risk >= self.min_rr:
                return False, "REJECTED_CHARGES", sizing_info
            return False, "REJECTED_RISK_REWARD", sizing_info

        if not self.session_controller.can_open_new_position(decision_time):
            return False, "REJECTED_SESSION_END_CONSTRAINT", sizing_info
        expected_horizon = signal.get("expected_horizon_minutes")
        if expected_horizon is not None:
            minutes_to_squareoff = self.session_controller.minutes_to_squareoff(decision_time)
            if float(expected_horizon) > minutes_to_squareoff:
                return False, "REJECTED_SESSION_END_CONSTRAINT", sizing_info

        time_remaining_minutes = (session_end - decision_time).total_seconds() / 60
        if time_remaining_minutes < self.session_controller.config.minimum_minutes_for_new_trade:
            return False, "REJECTED_SESSION_END_CONSTRAINT", sizing_info

        return True, None, sizing_info


def validate_signal(
    signal: Dict,
    decision_time: datetime,
    session_end: datetime,
    has_open_position: bool = False,
    tool_calls: List[Dict] = None,
) -> Dict:
    """
    Convenience function: validate and return enriched signal record.
    """
    validator = TradeValidator()
    is_valid, rejection, sizing = validator.validate(
        signal,
        decision_time,
        session_end,
        has_open_position=has_open_position,
        tool_calls=tool_calls,
    )

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
