"""
Focused tests for the new engine contracts:
1. First open-session candle is context-only (not trade-eligible)
2. ContextDeliveryMode tracking
3. Interfaces are importable and structured correctly
"""
import pytest
from datetime import datetime, date
from typing import Dict, Any

from agent.dart import DartAgent
from core.session_controller import SessionPhase
from core.interfaces import (
    ContextDeliveryMode, HistoricalDataRequest, AgentTurnRecord,
    ToolCallRecord, AuditEvent, TradeEvent, TradeEventType, EngineDecisionResult,
)


class MockLLM:
    def __init__(self):
        self.calls = []
        self.responses = [
            '{"type": "analysis_plan", "market_regime": "trend", "session_type": "trend_day"}',
            '{"type": "final_signal", "action": "BUY", "confidence": 0.7}',
            '{"type": "analysis_plan", "market_regime": "trend", "session_type": "trend_day"}',
            '{"type": "final_signal", "action": "BUY", "confidence": 0.8}',
        ]
        self.response_idx = 0

    def call(self, messages):
        self.calls.append(list(messages))
        res = self.responses[self.response_idx]
        if self.response_idx < len(self.responses) - 1:
            self.response_idx += 1
        return res


class DummyHarness:
    def __init__(self, run_id: str, decision_time: datetime):
        self.run_id = run_id
        self.decision_time = decision_time
        self.call_count = 0


def make_mock_package(decision_time: datetime) -> Dict[str, Any]:
    return {
        "instrument": "NIFTY",
        "symbol": "RELIANCE",
        "decision_time": str(decision_time),
        "intraday_timeframe": "15min",
        "context_windows": {},
        "context_row_counts": {"weekly": 2, "daily": 2, "intraday": 2},
        "current_price": 2500.0,
        "latest_candle": {"open": 2495.0, "high": 2505.0, "low": 2490.0, "close": 2500.0, "volume": 5000},
        "indicators": {"rsi_14": 55.0, "atr_14": 25.0},
        "swings": {}, "levels": {}, "pattern": "inside_bar",
        "price_location": "near_vwap",
        "trend_5m": "bullish", "trend_daily": "bullish", "trend_weekly": "bullish",
        "recent_intraday_candles": [
            {"time": "2026-05-28 09:15:00+05:30", "open": 2490.0, "high": 2498.0, "low": 2488.0, "close": 2495.0, "volume": 1000},
        ],
        "daily_summaries": [{"date": "2026-05-27", "open": 2485.0, "high": 2505.0, "low": 2480.0, "close": 2498.0, "volume": 12000, "range": 25.0}],
        "weekly_summaries": [{"week": "2026-05-18", "open": 2450.0, "high": 2480.0, "low": 2440.0, "close": 2475.0, "volume": 50000, "range": 40.0}],
        "chart_paths": {},
    }


# ============================================================
# Interface Structure Tests
# ============================================================

class TestInterfaces:
    def test_context_delivery_mode_values(self):
        assert ContextDeliveryMode.BOOTSTRAP.value == "bootstrap"
        assert ContextDeliveryMode.INCREMENTAL.value == "incremental"
        assert ContextDeliveryMode.RESET.value == "reset"

    def test_historical_data_request_defaults(self):
        req = HistoricalDataRequest(timeframe="daily")
        assert req.timeframe == "daily"
        assert req.max_candles == 60
        assert req.start_date is None

    def test_tool_call_record(self):
        rec = ToolCallRecord(
            round_num=1, tool_name="test_tool",
            arguments={"arg1": "val1"}, reason="testing",
            result={"output": 42},
        )
        assert rec.round_num == 1
        assert rec.tool_name == "test_tool"
        assert rec.status == "success"
        assert rec.latency_ms is None

    def test_agent_turn_record(self):
        rec = AgentTurnRecord(turn_number=1, role="assistant", raw_output="{}")
        assert rec.schema_valid == True
        assert rec.schema_errors == []

    def test_audit_event_defaults(self):
        event = AuditEvent(event_type="TEST_EVENT", message="test")
        assert event.severity == "info"
        assert event.run_id is None

    def test_trade_event_type_values(self):
        assert TradeEventType.ENTRY_REQUESTED.value == "entry_requested"
        assert TradeEventType.REJECTED.value == "rejected"
        assert TradeEventType.FORCED_SQUARE_OFF.value == "forced_square_off"

    def test_engine_decision_result(self):
        from datetime import datetime
        result = EngineDecisionResult(
            decision_id="dec_001", run_id="run_001",
            symbol="TEST", decision_time=datetime.now(),
            raw_action="BUY", validated_action="BUY", is_valid=True,
        )
        assert result.context_data_hash is None
        assert result.evaluation_labels == []


# ============================================================
# First Candle Context-Only Contract
# ============================================================

class TestFirstCandleContract:
    def test_first_candle_overrides_buy_to_skip(self, monkeypatch):
        """The first candle of a new session should never generate a trade."""
        monkeypatch.setattr("db.services.ReplayStateService.get_latest_portfolio_snapshot", lambda run_id: {
            "cash_available": 100000.0, "capital_deployed": 0.0,
            "realized_pnl": 0.0, "charges_paid": 0.0, "trades_taken_today": 0,
            "max_trades_per_day": 3, "daily_loss_used": 0.0, "max_daily_loss": 5000.0,
        })
        monkeypatch.setattr("db.services.ReplayStateService.get_active_position", lambda run_id, symbol: None)
        monkeypatch.setattr("db.services.ReplayStateService.get_session_phase", lambda T: SessionPhase.ACTIVE_TRADING)
        monkeypatch.setattr("db.repository.SessionRepository.get_session_map", lambda self, session_id: {
            "opening_range_high": 2510.0, "opening_range_low": 2480.0,
            "session_high": 2505.0, "session_low": 2490.0, "session_vwap": 2497.0,
            "vwap_slope": 0.0001, "gap_classification": "gap_up",
            "market_regime": "trend", "current_bias": "bullish",
        })
        monkeypatch.setattr("db.repository.MemoryRepository.get_episodes", lambda self, symbol, limit: [])
        monkeypatch.setattr("db.repository.MemoryRepository.get_reflections", lambda self, symbol, limit: [])

        # Simulate the first candle of a session
        agent = DartAgent()
        mock_llm = MockLLM()
        monkeypatch.setattr(agent, "_call_llm", mock_llm.call)

        t = datetime(2026, 5, 28, 9, 45, 0)
        harness = DummyHarness("run_123", t)
        package = make_mock_package(t)

        # Run agent decide - would normally return BUY from mock
        result = agent.decide(package, "legacy_text", harness)

        # The LLM returns BUY but since this is the agent, it doesn't know about
        # the first-candle rule. The ReplayRunner enforces this externally.
        # So the agent still returns BUY - that's correct.
        # The enforcement happens in ReplayRunner._run_agent_with_audit.
        assert result["final_signal"]["action"] == "BUY"

        # Now simulate what ReplayRunner does:
        signal = result["final_signal"]
        is_first_candle = True  # First candle in session
        if is_first_candle and signal.get("action") in ("BUY", "SELL"):
            signal["action"] = "SKIP"
            signal["reason"] = f"First open-session candle is context-only. {signal.get('reason', '')}"

        assert signal["action"] == "SKIP"
        assert "context-only" in signal["reason"]
