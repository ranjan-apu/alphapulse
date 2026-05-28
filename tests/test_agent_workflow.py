"""
Regression tests for agent workflow wiring.
"""
import uuid

from agent.dart import DartAgent
from agent.memory import MemoryStore
from agent.schema import AnalysisPlan
from db.services import RunBootstrapService
from db.unit_of_work import UnitOfWork


def test_focused_memory_uses_analysis_plan_retrieval_query(monkeypatch):
    symbol = "RELIANCE"
    run_id = f"test_agent_memory_{uuid.uuid4().hex[:10]}"
    RunBootstrapService.create_or_resume_run(run_id=run_id, symbol=symbol)

    with UnitOfWork() as uow:
        uow.memory.save_episode({
            "episode_id": f"ep_{uuid.uuid4().hex[:10]}",
            "run_id": run_id,
            "symbol": symbol,
            "action": "BUY",
            "direction": "bullish",
            "market_regime": "trend",
            "session_type": "trend_day",
            "gap_type": "gap_up",
            "structure_state": "bullish_bos",
            "vwap_relation": "above_vwap",
            "profile_location": "value_high",
            "price_location": "near_prior_high",
            "time_bucket": "mid_session",
            "volatility_bucket": "normal",
            "setup_tags": ["breakout", "vwap_reclaim"],
            "outcome_net_r": 1.4,
            "outcome_label": "target_hit",
            "confidence": 0.8,
        })
        uow.memory.save_episode({
            "episode_id": f"ep_{uuid.uuid4().hex[:10]}",
            "run_id": run_id,
            "symbol": symbol,
            "action": "BUY",
            "direction": "bullish",
            "market_regime": "range",
            "session_type": "range_day",
            "gap_type": "no_gap",
            "structure_state": "range_bound",
            "vwap_relation": "at_vwap",
            "profile_location": "value_mid",
            "price_location": "range_mid",
            "time_bucket": "late_session",
            "volatility_bucket": "low",
            "setup_tags": ["range_trade"],
            "outcome_net_r": -0.5,
            "outcome_label": "stop_hit",
            "confidence": 0.5,
        })
        uow.memory.save_reflection({
            "reflection_id": f"refl_{uuid.uuid4().hex[:10]}",
            "run_id": run_id,
            "symbol": symbol,
            "lesson": "Breakout above VWAP worked when structure was bullish.",
            "tags": ["breakout"],
            "source_episode_ids": [],
            "direction": "BUY",
            "reflection_level": "HIGH",
            "confidence": 0.9,
            "num_supporting_episodes": 1,
        })

    original = MemoryStore.build_retrieval_query
    calls = []

    def spy(self, *args, **kwargs):
        calls.append(kwargs)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(MemoryStore, "build_retrieval_query", spy)

    plan = AnalysisPlan(
        direction_bias="bullish",
        setup_tags=["breakout"],
        market_regime="trend",
        session_type="trend_day",
        gap_type="gap_up",
        structure_state="bullish_bos",
        vwap_relation="above_vwap",
        profile_location="value_high",
        price_location="near_prior_high",
        time_bucket="mid_session",
        volatility_bucket="normal",
    )

    context = object.__new__(DartAgent)._build_focused_memory_context(plan, run_id, symbol)

    assert calls
    assert calls[0]["analysis_plan"] is plan
    assert "target_hit" in context
    assert "range_trade" not in context
    assert "Breakout above VWAP" in context
