"""
Unit tests for incremental context prompt optimization.
Verifies that:
1. The first prompt of a session contains the full historical candle tables.
2. Subsequent prompts of the same session omit the historical candle tables.
3. Newly closed candles are correctly identified and sent in an Incremental Market Update block.
4. Crossing session/date boundaries resets the history and sends the full context again.
"""
import pytest
from datetime import datetime, date
from typing import Dict, Any

from agent.dart import DartAgent
from core.session_controller import SessionPhase
from core.tools import ToolHarness

class MockLLM:
    def __init__(self):
        self.calls = []
        self.responses = [
            # Decision 1 (Round 0, Round 1)
            '{"type": "analysis_plan", "market_regime": "trend", "session_type": "trend_day"}',
            '{"type": "final_signal", "action": "SKIP"}',
            # Decision 2 (Round 0, Round 1)
            '{"type": "analysis_plan", "market_regime": "trend", "session_type": "trend_day"}',
            '{"type": "final_signal", "action": "SKIP"}',
            # Decision 3 (Round 0, Round 1)
            '{"type": "analysis_plan", "market_regime": "trend", "session_type": "trend_day"}',
            '{"type": "final_signal", "action": "SKIP"}'
        ]
        self.response_idx = 0

    def call(self, messages):
        # We copy messages to record the state at call time
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
        "context_windows": {
            "weekly": "last 3 months of completed weekly candles",
            "daily": "last 1 month of completed daily candles",
            "intraday": "last 3 trading sessions of completed 15min candles",
        },
        "context_row_counts": {
            "weekly": 2,
            "daily": 2,
            "intraday": 2,
        },
        "current_price": 2500.0,
        "latest_candle": {
            "open": 2495.0,
            "high": 2505.0,
            "low": 2490.0,
            "close": 2500.0,
            "volume": 5000,
        },
        "indicators": {"rsi_14": 55.0, "atr_14": 25.0},
        "swings": {},
        "levels": {},
        "pattern": "inside_bar",
        "price_location": "near_vwap",
        "trend_5m": "bullish",
        "trend_daily": "bullish",
        "trend_weekly": "bullish",
        "recent_intraday_candles": [
            {"time": "2026-05-28 09:15:00+05:30", "open": 2490.0, "high": 2498.0, "low": 2488.0, "close": 2495.0, "volume": 1000},
            {"time": "2026-05-28 09:30:00+05:30", "open": 2495.0, "high": 2502.0, "low": 2492.0, "close": 2499.0, "volume": 1200},
        ],
        "daily_summaries": [
            {"date": "2026-05-26", "open": 2470.0, "high": 2490.0, "low": 2465.0, "close": 2485.0, "volume": 10000, "range": 25.0},
            {"date": "2026-05-27", "open": 2485.0, "high": 2505.0, "low": 2480.0, "close": 2498.0, "volume": 12000, "range": 25.0},
        ],
        "weekly_summaries": [
            {"week": "2026-05-18", "open": 2450.0, "high": 2480.0, "low": 2440.0, "close": 2475.0, "volume": 50000, "range": 40.0},
            {"week": "2026-05-25", "open": 2475.0, "high": 2510.0, "low": 2470.0, "close": 2502.0, "volume": 60000, "range": 40.0},
        ],
        "chart_paths": {},
    }


def test_incremental_context_flow(monkeypatch):
    # Mock all DB dependencies
    monkeypatch.setattr("db.services.ReplayStateService.get_latest_portfolio_snapshot", lambda run_id: {
        "cash_available": 100000.0,
        "capital_deployed": 0.0,
        "realized_pnl": 0.0,
        "charges_paid": 0.0,
        "trades_taken_today": 0,
        "max_trades_per_day": 3,
        "daily_loss_used": 0.0,
        "max_daily_loss": 5000.0,
    })
    monkeypatch.setattr("db.services.ReplayStateService.get_active_position", lambda run_id, symbol: None)
    monkeypatch.setattr("db.services.ReplayStateService.get_session_phase", lambda T: SessionPhase.ACTIVE_TRADING)
    monkeypatch.setattr("db.repository.SessionRepository.get_session_map", lambda self, session_id: {
        "opening_range_high": 2510.0,
        "opening_range_low": 2480.0,
        "session_high": 2505.0,
        "session_low": 2490.0,
        "session_vwap": 2497.0,
        "vwap_slope": 0.0001,
        "gap_classification": "gap_up",
        "market_regime": "trend",
        "current_bias": "bullish",
    })
    monkeypatch.setattr("db.repository.MemoryRepository.get_episodes", lambda self, symbol, limit: [])
    monkeypatch.setattr("db.repository.MemoryRepository.get_reflections", lambda self, symbol, limit: [])

    # Initialize DartAgent and Mock LLM
    agent = DartAgent()
    mock_llm = MockLLM()
    monkeypatch.setattr(agent, "_call_llm", mock_llm.call)

    # --------------------------------------------------------
    # STEP 1: First decision of the session (T1 = 2026-05-28 09:45:00)
    # --------------------------------------------------------
    t1 = datetime(2026, 5, 28, 9, 45, 0)
    harness1 = DummyHarness("run_123", t1)
    package1 = make_mock_package(t1)

    result1 = agent.decide(package1, "legacy_text", harness1)
    assert result1["final_signal"]["action"] == "SKIP"

    # Verify that Decision 1 messages sent to LLM contains the full candle tables
    d1_round0_messages = mock_llm.calls[0]
    assert len(d1_round0_messages) == 3  # [system, full_user, step_prompt]
    user_msg_content = d1_round0_messages[1]["content"]
    if isinstance(user_msg_content, list):
        user_msg_content = user_msg_content[0]["text"]
    assert "LAST 2 WEEKLY CANDLES" in user_msg_content
    assert "LAST 2 DAILY CANDLES" in user_msg_content
    assert "LAST 2 15min CANDLES" in user_msg_content

    # Check that tracking timestamps were initialized
    assert agent.last_weekly_time == "2026-05-25"
    assert agent.last_daily_time == "2026-05-27"
    assert agent.last_intraday_time == "2026-05-28 09:30:00+05:30"
    assert agent.last_session_date == date(2026, 5, 28)
    assert len(agent.conversation_history) == 2  # [system, full_user]

    # --------------------------------------------------------
    # STEP 2: Second decision of the session, same day (T2 = 2026-05-28 10:00:00)
    # --------------------------------------------------------
    # Let's add 1 new intraday candle to the package
    t2 = datetime(2026, 5, 28, 10, 0, 0)
    harness2 = DummyHarness("run_123", t2)
    package2 = make_mock_package(t2)
    package2["recent_intraday_candles"].append({
        "time": "2026-05-28 09:45:00+05:30",
        "open": 2499.0,
        "high": 2504.0,
        "low": 2497.0,
        "close": 2502.0,
        "volume": 1500
    })
    # Set context row count to 3
    package2["context_row_counts"]["intraday"] = 3

    result2 = agent.decide(package2, "legacy_text", harness2)
    assert result2["final_signal"]["action"] == "SKIP"

    # Verify that Decision 2 messages contains the initial system/user message,
    # the Incremental Update, and the step prompt.
    d2_round0_messages = mock_llm.calls[2]  # Index 2 is the 3rd LLM call overall
    assert len(d2_round0_messages) == 4  # [system, full_user, incremental_update, step_prompt]

    # Check incremental update content
    incremental_msg = d2_round0_messages[2]
    assert incremental_msg["role"] == "user"
    incremental_content = incremental_msg["content"]
    if isinstance(incremental_content, list):
        incremental_content = incremental_content[0]["text"]
    assert "### INCREMENTAL MARKET UPDATE" in incremental_content
    assert "Newly Completed Intraday (15min) Candles" in incremental_content
    assert "Time: 09:45:00+05:30 | O=2499.00 H=2504.00 L=2497.00 C=2502.00 V=1500" in incremental_content

    # Check step prompt content (should NOT contain the candle tables)
    step_msg = d2_round0_messages[3]
    assert step_msg["role"] == "user"
    step_content = step_msg["content"]
    if isinstance(step_content, list):
        step_content = step_content[0]["text"]
    assert "### STEP DECISION POINT" in step_content
    assert "LAST 2 WEEKLY CANDLES" not in step_content
    assert "LAST 2 DAILY CANDLES" not in step_content

    # Verify tracking timestamps were updated
    assert agent.last_intraday_time == "2026-05-28 09:45:00+05:30"
    assert len(agent.conversation_history) == 3  # [system, full_user, incremental_update]

    # --------------------------------------------------------
    # STEP 3: Third decision, next day (T3 = 2026-05-29 09:45:00)
    # --------------------------------------------------------
    t3 = datetime(2026, 5, 29, 9, 45, 0)
    harness3 = DummyHarness("run_123", t3)
    package3 = make_mock_package(t3)
    
    # Update summaries to reflect next-day completed candles
    package3["daily_summaries"].append({
        "date": "2026-05-28", "open": 2498.0, "high": 2515.0, "low": 2488.0, "close": 2505.0, "volume": 15000, "range": 27.0
    })
    package3["recent_intraday_candles"] = [
        {"time": "2026-05-29 09:15:00+05:30", "open": 2505.0, "high": 2512.0, "low": 2500.0, "close": 2510.0, "volume": 1100},
        {"time": "2026-05-29 09:30:00+05:30", "open": 2510.0, "high": 2518.0, "low": 2508.0, "close": 2515.0, "volume": 1300},
    ]

    result3 = agent.decide(package3, "legacy_text", harness3)
    assert result3["final_signal"]["action"] == "SKIP"

    # Verify that crossing the day boundary reset the history and we got a full prompt again
    d3_round0_messages = mock_llm.calls[4]  # Index 4 is the 5th LLM call overall
    assert len(d3_round0_messages) == 3  # [system, full_user, step_prompt]
    
    user_msg_content_t3 = d3_round0_messages[1]["content"]
    if isinstance(user_msg_content_t3, list):
        user_msg_content_t3 = user_msg_content_t3[0]["text"]
    assert "LAST 2 WEEKLY CANDLES" in user_msg_content_t3
    assert "LAST 3 DAILY CANDLES" in user_msg_content_t3
    assert "LAST 2 15min CANDLES" in user_msg_content_t3
    assert "2026-05-28" in user_msg_content_t3  # The new daily candle is formatted
