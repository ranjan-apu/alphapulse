"""
Signal journal: records every decision (both trades and no-trades).
Stores structured records with market state, agent output, and metadata.
"""
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

from config import config


class SignalJournal:
    """
    Append-only journal for all decisions made during the replay.
    """

    def __init__(self, output_path: Path = None, clear_existing: bool = True):
        self.output_path = output_path or config.JOURNAL_DIR / "signal_journal.jsonl"
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.records: List[Dict] = []
        # Clear existing journal to avoid duplicate entries across runs
        if clear_existing and self.output_path.exists():
            self.output_path.unlink()

    def record(
        self,
        decision_time: datetime,
        market_state_package: Dict,
        agent_result: Dict,
        validation_result: Dict,
        chart_paths: Dict[str, str],
    ) -> Dict:
        """
        Record a single decision. Returns the record dict.
        """
        signal = agent_result.get("final_signal", {})
        action = validation_result.get("action", signal.get("action", "HOLD"))

        record = {
            "timestamp": str(decision_time),
            "timestamp_epoch": decision_time.timestamp(),
            "instrument": config.INSTRUMENT_NAME,
            "symbol": config.SYMBOL,
            "current_price": market_state_package.get("current_price"),
            "action": action,
            "original_action": signal.get("action", "HOLD"),
            "confidence": signal.get("confidence"),
            "dart": signal.get("dart", {}),
            "entry": signal.get("entry"),
            "stop": signal.get("stop"),
            "target": signal.get("target"),
            "reward_risk": signal.get("rewardRisk"),
            "net_reward_risk_after_charges": signal.get("net_reward_risk_after_charges"),
            "quantity": signal.get("quantity"),
            "deployed_capital": signal.get("deployed_capital"),
            "reason": signal.get("reason", ""),
            "invalidation": signal.get("invalidation", ""),
            "is_valid": validation_result.get("is_valid"),
            "rejection_reason": validation_result.get("rejection_reason"),
            "sizing": validation_result.get("sizing"),
            "tool_calls": agent_result.get("tool_calls", []),
            "tool_call_count": len(agent_result.get("tool_calls", [])),
            "chart_paths": chart_paths,
            "market_state_snapshot": self._summarize_state(market_state_package),
        }

        self.records.append(record)
        self._append_to_file(record)
        return record

    def _summarize_state(self, state: Dict) -> Dict:
        """Create a compact summary of the market state for journal."""
        return {
            "trend_5m": state.get("trend_5m", ""),
            "trend_intraday": state.get("trend_intraday", state.get("trend_5m", "")),
            "intraday_timeframe": state.get("intraday_timeframe", ""),
            "trend_daily": state.get("trend_daily", ""),
            "trend_weekly": state.get("trend_weekly", ""),
            "pattern": state.get("pattern", ""),
            "price_location": state.get("price_location", ""),
            "indicators": state.get("indicators", {}),
            "levels": state.get("levels", {}),
        }

    def _append_to_file(self, record: Dict):
        """Append record as JSONL to the journal file."""
        with open(self.output_path, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def get_records(self) -> List[Dict]:
        return self.records

    def get_actionable_trades(self) -> List[Dict]:
        """Get only validated BUY/SELL signals."""
        return [r for r in self.records if r["action"] in ("BUY", "SELL")]

    def get_holds(self) -> List[Dict]:
        """Get HOLD signals."""
        return [r for r in self.records if r["action"] == "HOLD"]

    def get_rejected(self) -> List[Dict]:
        """Get rejected signals."""
        return [r for r in self.records if r["action"].startswith("REJECTED")]

    def summary(self) -> Dict:
        """Generate summary statistics."""
        total = len(self.records)
        buys = sum(1 for r in self.records if r["action"] == "BUY")
        sells = sum(1 for r in self.records if r["action"] == "SELL")
        holds = sum(1 for r in self.records if r["action"] == "HOLD")
        rejected = sum(1 for r in self.records if r["action"].startswith("REJECTED"))

        rejection_counts = {}
        for r in self.records:
            rr = r.get("rejection_reason")
            if rr:
                rejection_counts[rr] = rejection_counts.get(rr, 0) + 1

        avg_tool_calls = sum(r.get("tool_call_count", 0) for r in self.records) / max(total, 1)

        return {
            "total_decisions": total,
            "buy_signals": buys,
            "sell_signals": sells,
            "hold_signals": holds,
            "actionable_trades": buys + sells,
            "rejected_signals": rejected,
            "hold_rate": round(holds / max(total, 1), 3),
            "rejection_by_reason": rejection_counts,
            "avg_tool_calls_per_decision": round(avg_tool_calls, 2),
        }
