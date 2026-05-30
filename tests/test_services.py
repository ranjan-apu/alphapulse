"""
Tests for Bootstrap, ReplayState, DecisionTransaction, and OutcomeFeedback Services.
"""
import pytest
from datetime import datetime, timezone, timedelta
import uuid
import pandas as pd
from core.charges import EquityCashCharges, compute_entry_charges, compute_exit_charges
from core.slippage import SlippageConfig
from db.services import (
    RunBootstrapService,
    ReplayStateService,
    DecisionTransactionService,
    OutcomeFeedbackService,
    SessionStateService,
    AuditService,
)
from db.unit_of_work import UnitOfWork
from db.repository import OrderRepository


def _unique_run(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def _seed_run(symbol: str = "RELIANCE") -> str:
    run_id = _unique_run("test_service_run")
    RunBootstrapService.create_or_resume_run(
        run_id=run_id,
        symbol=symbol,
        starting_capital=100000.0,
        max_capital_per_trade=30000.0,
    )
    return run_id


def _buy_signal():
    return {
        "action": "BUY",
        "confidence": 0.8,
        "entry": 2400.0,
        "stop": 2370.0,
        "target": 2460.0,
        "net_reward_risk": 2.0,
        "expected_horizon_minutes": 45,
        "dart": {
            "direction": "bullish",
            "area": "support",
            "risk": "low",
            "trigger": "breakout"
        },
        "checklist": {
            "market_regime": "trend",
            "session_type": "trend_day",
            "structure_state": "bullish_bos",
            "gap_type": "gap_up",
            "vwap_relation": "above_vwap",
            "profile_location": "value_high",
            "price_location": "near_prior_high",
            "time_bucket": "mid_session",
            "volatility_bucket": "normal",
        },
        "reason": "Thesis validation"
    }


def _buy_validation(quantity: int = 12):
    return {
        "action": "BUY",
        "is_valid": True,
        "sizing": {
            "quantity": quantity,
            "entry": 2400.0,
            "stop": 2370.0,
            "target": 2460.0
        }
    }

def test_run_services_integration():
    run_id = f"test_service_run_{int(datetime.now().timestamp())}"
    symbol = "RELIANCE"
    
    # 1. Bootstrap
    res_run = RunBootstrapService.create_or_resume_run(
        run_id=run_id,
        symbol=symbol,
        starting_capital=100000.0,
        max_capital_per_trade=30000.0,
        notes="Testing services layer"
    )
    assert res_run == run_id
    
    # Verify portfolio snapshot was seeded
    snap = ReplayStateService.get_latest_portfolio_snapshot(run_id)
    assert snap is not None
    assert snap["starting_capital"] == 100000.0
    assert snap["cash_available"] == 100000.0
    
    # 2. Snapshot set creation
    df = pd.DataFrame(
        {"open": [100.0], "high": [110.0], "low": [90.0], "close": [105.0], "volume": [1000]},
        index=pd.date_range("2026-05-28 09:15", periods=1, freq="15min")
    )
    set_id = RunBootstrapService.create_snapshot_set(symbol, df, df, df)
    assert set_id.startswith("set_")
    
    # 3. Replay state and evaluation timing
    T = datetime.now(timezone.utc)
    # Check if we should evaluate (first time is True)
    assert ReplayStateService.should_evaluate(run_id, symbol, T) is True
    
    # 4. Decision transaction processing
    signal = {
        "action": "BUY",
        "confidence": 0.8,
        "entry": 2400.0,
        "stop": 2370.0,
        "target": 2460.0,
        "net_reward_risk": 2.0,
        "expected_horizon_minutes": 45,
        "dart": {
            "direction": "bullish",
            "area": "support",
            "risk": "low",
            "trigger": "breakout"
        },
        "checklist": {
            "market_regime": "trend",
            "session_type": "trend_day",
            "structure_state": "bullish_bos"
        },
        "reason": "Thesis validation"
    }
    validation_result = {
        "action": "BUY",
        "is_valid": True,
        "sizing": {
            "quantity": 12,
            "entry": 2400.0,
            "stop": 2370.0,
            "target": 2460.0
        }
    }
    
    decision, decision_id = DecisionTransactionService.process_decision(
        run_id=run_id,
        symbol=symbol,
        T=T,
        current_price=2400.0,
        signal=signal,
        validation_result=validation_result
    )
    assert decision_id.startswith("dec_")
    assert decision["validated_action"] == "BUY"
    assert decision["position_id"] is not None
    
    # Check active position
    active_pos = ReplayStateService.get_active_position(run_id, symbol)
    assert active_pos is not None
    assert active_pos["position_id"] == decision["position_id"]
    assert active_pos["active"] is True
    assert active_pos["quantity"] == 12
    
    # Check should_evaluate immediately after decision (should be False if MIN_MINUTES_BETWEEN_SIGNALS not elapsed, unless has position)
    # But since we have a position, it should evaluate to manage it
    assert ReplayStateService.should_evaluate(run_id, symbol, T + timedelta(minutes=1)) is True
    
    # 5. Position exit transaction
    exit_validation = {
        "action": "EXIT",
        "is_valid": True,
        "stop_hit": False,
        "target_hit": True
    }
    
    exit_signal = {
        "action": "EXIT",
        "position_id": active_pos["position_id"],
        "exit_reason": "target_hit",
        "suggested_exit_price": 2460.0,
        "reason": "Target reached"
    }
    
    exit_decision, exit_dec_id = DecisionTransactionService.process_decision(
        run_id=run_id,
        symbol=symbol,
        T=T + timedelta(minutes=45),
        current_price=2460.0,
        signal=exit_signal,
        validation_result=exit_validation
    )
    
    # Active position should be closed
    closed_pos = ReplayStateService.get_active_position(run_id, symbol)
    assert closed_pos is None
    
    # 6. Feedback evaluation record
    OutcomeFeedbackService.record_feedback(
        decision_id=decision_id,
        outcome_label="win",
        net_r=2.0,
        mfe_pct=2.5,
        mae_pct=0.1,
        setup_tags=["breakout"]
    )
    
    # Verify calibration stats populated
    # Verify calibration stats populated
    with UnitOfWork() as uow:
        stats = uow.calibration.get_stats_for_run(run_id)
        assert len(stats) > 0
        assert stats[0]["wins"] == 1


def test_session_state_service():
    run_id = f"test_session_run_{int(datetime.now().timestamp())}"
    symbol = "TCS"
    
    # Setup DataFrames
    times = pd.date_range("2026-05-28 09:15:00", periods=5, freq="15min", tz="Asia/Kolkata")
    df_15m = pd.DataFrame({
        "open": [100.0, 101.0, 102.0, 103.0, 104.0],
        "high": [102.0, 103.0, 104.0, 105.0, 106.0],
        "low": [99.0, 100.0, 101.0, 102.0, 103.0],
        "close": [101.0, 102.0, 103.0, 104.0, 105.0],
        "volume": [1000, 1100, 1200, 1300, 1400]
    }, index=times)
    
    daily_times = pd.date_range("2026-05-25", periods=3, freq="D", tz="Asia/Kolkata")
    df_daily = pd.DataFrame({
        "open": [95.0, 97.0, 99.0],
        "high": [100.0, 102.0, 101.0],
        "low": [94.0, 96.0, 98.0],
        "close": [98.0, 99.0, 100.0],
        "volume": [10000, 11000, 12000]
    }, index=daily_times)
    
    # Bootstrap run first
    RunBootstrapService.create_or_resume_run(
        run_id=run_id,
        symbol=symbol,
        starting_capital=100000.0,
        max_capital_per_trade=30000.0
    )
    
    # 1. Init session
    T = times[0]
    session_id = SessionStateService.init_session_if_needed(run_id, symbol, T, df_daily, df_15m)
    assert session_id == f"sess_{run_id}_{T.strftime('%Y%m%d')}"
    
    # Verify DB entry
    with UnitOfWork() as uow:
        sess_map = uow.sessions.get_session_map(session_id)
        assert sess_map is not None
        assert sess_map["symbol"] == symbol
        assert sess_map["gap_classification"] is not None
        
        # Verify initial swing levels were added
        levels = uow.sessions.get_all_levels(session_id)
        assert len(levels) > 0
        
    # 2. Process candle update
    SessionStateService.process_candle_update(run_id, symbol, T, df_15m.loc[[T]], df_daily)
    
    # Verify update
    with UnitOfWork() as uow:
        sess_map = uow.sessions.get_session_map(session_id)
        assert sess_map["session_high"] == 102.0
        assert sess_map["session_low"] == 99.0


def test_buy_uses_entry_order_charges_only():
    symbol = "RELIANCE"
    run_id = _seed_run(symbol)
    T = datetime.now(timezone.utc)

    decision, _ = DecisionTransactionService.process_decision(
        run_id, symbol, T, 2400.0, _buy_signal(), _buy_validation(quantity=12)
    )

    with UnitOfWork() as uow:
        active_pos = uow.positions.get_active_position(run_id, symbol)
        orders = uow.orders.get_orders_for_position(decision["position_id"])

    assert active_pos is not None
    assert len(orders) == 1
    order = orders[0]
    assert order["order_type"] == "ENTRY"
    expected = compute_entry_charges(
        EquityCashCharges(), "BUY", 12, order["executed_price"]
    )
    assert order["charges_total"] == pytest.approx(expected["total"])
    assert active_pos["charges_entry"] == pytest.approx(expected["total"])
    assert order["charges_total"] < compute_entry_charges(
        EquityCashCharges(), "BUY", 12, 2460.0
    )["total"] + compute_exit_charges(EquityCashCharges(), "SELL", 12, 2460.0)["total"]


def test_hold_with_active_position_updates_unrealized_pnl():
    symbol = "RELIANCE"
    run_id = _seed_run(symbol)
    T = datetime.now(timezone.utc)
    DecisionTransactionService.process_decision(
        run_id, symbol, T, 2400.0, _buy_signal(), _buy_validation(quantity=10)
    )

    hold_decision, _ = DecisionTransactionService.process_decision(
        run_id,
        symbol,
        T + timedelta(minutes=15),
        2410.0,
        {"action": "HOLD", "confidence": 0.4, "reason": "still valid"},
        {"action": "HOLD", "is_valid": True},
    )

    with UnitOfWork() as uow:
        pos = uow.positions.get_active_position(run_id, symbol)
        snap = uow.portfolio.get_latest_snapshot(run_id)

    assert hold_decision["validated_action"] == "HOLD"
    assert pos["unrealized_pnl"] == pytest.approx(10 * (2410.0 - pos["executed_entry"]))
    assert snap["unrealized_pnl"] == pytest.approx(pos["unrealized_pnl"])


def test_skip_persists_decision_and_snapshot_without_order_or_position():
    symbol = "RELIANCE"
    run_id = _seed_run(symbol)
    T = datetime.now(timezone.utc)

    decision, decision_id = DecisionTransactionService.process_decision(
        run_id,
        symbol,
        T,
        2400.0,
        {"action": "SKIP", "confidence": 0.2, "reason": "no setup"},
        {"action": "SKIP", "is_valid": True},
    )

    with UnitOfWork() as uow:
        active_pos = uow.positions.get_active_position(run_id, symbol)
        snap = uow.portfolio.get_latest_snapshot(run_id)
        uow.cursor.execute("SELECT COUNT(*) FROM orders_simulated WHERE run_id = %s", (run_id,))
        order_count = uow.cursor.fetchone()[0]

    assert decision["validated_action"] == "SKIP"
    assert decision_id.startswith("dec_")
    assert active_pos is None
    assert order_count == 0
    assert snap["decision_id"] == decision_id


def test_exit_uses_exit_leg_charges_and_round_trip_reconciles():
    symbol = "RELIANCE"
    run_id = _seed_run(symbol)
    T = datetime.now(timezone.utc)
    buy_decision, _ = DecisionTransactionService.process_decision(
        run_id, symbol, T, 2400.0, _buy_signal(), _buy_validation(quantity=12)
    )

    DecisionTransactionService.process_decision(
        run_id,
        symbol,
        T + timedelta(minutes=45),
        2460.0,
        {"action": "EXIT", "position_id": buy_decision["position_id"], "reason": "Target reached"},
        {"action": "EXIT", "is_valid": True, "target_hit": True},
    )

    with UnitOfWork() as uow:
        pos = uow.positions.get_position(buy_decision["position_id"])
        orders = sorted(
            uow.orders.get_orders_for_position(buy_decision["position_id"]),
            key=lambda row: row["created_at"],
        )
        snap = uow.portfolio.get_latest_snapshot(run_id)

    assert pos["active"] is False
    assert len(orders) == 2
    entry_order, exit_order = orders
    assert exit_order["order_type"] == "TARGET"
    expected_exit = compute_exit_charges(
        EquityCashCharges(), "SELL", 12, exit_order["executed_price"]
    )
    assert exit_order["charges_total"] == pytest.approx(expected_exit["total"])
    assert pos["charges_total"] == pytest.approx(entry_order["charges_total"] + exit_order["charges_total"])
    assert snap["capital_deployed"] == pytest.approx(0.0)
    assert snap["charges_paid"] == pytest.approx(pos["charges_total"])
    gross = 12 * (exit_order["executed_price"] - entry_order["executed_price"])
    assert pos["realized_pnl"] == pytest.approx(gross - pos["charges_total"])
    assert snap["realized_pnl"] == pytest.approx(pos["realized_pnl"])


def test_forced_squareoff_applies_squareoff_slippage():
    symbol = "RELIANCE"
    run_id = _seed_run(symbol)
    T = datetime.now(timezone.utc)
    buy_decision, _ = DecisionTransactionService.process_decision(
        run_id, symbol, T, 2400.0, _buy_signal(), _buy_validation(quantity=5)
    )

    DecisionTransactionService.process_decision(
        run_id,
        symbol,
        T + timedelta(hours=6),
        2450.0,
        {"action": "EXIT", "position_id": buy_decision["position_id"], "reason": "squareoff"},
        {"action": "EXIT", "is_valid": True, "forced_exit": True},
    )

    with UnitOfWork() as uow:
        orders = uow.orders.get_orders_for_position(buy_decision["position_id"])
    exit_order = [order for order in orders if order["order_type"] == "FORCED_SQUAREOFF"][0]
    assert exit_order["executed_price"] == pytest.approx(2450.0 - SlippageConfig().force_squareoff_slippage)
    assert exit_order["slippage_points"] == pytest.approx(SlippageConfig().force_squareoff_slippage)


def test_feedback_preserves_checklist_and_writes_structured_outcome_memory_calibration():
    symbol = "RELIANCE"
    run_id = _seed_run(symbol)
    T = datetime.now(timezone.utc)
    signal = _buy_signal()
    _, decision_id = DecisionTransactionService.process_decision(
        run_id, symbol, T, 2400.0, signal, _buy_validation(quantity=8)
    )

    OutcomeFeedbackService.record_feedback(
        decision_id=decision_id,
        outcome_label="target_hit",
        net_r=1.8,
        mfe_pct=2.5,
        mae_pct=0.1,
        setup_tags=["breakout", "vwap_reclaim"],
    )

    with UnitOfWork() as uow:
        uow.cursor.execute("SELECT * FROM decisions WHERE decision_id = %s", (decision_id,))
        dec = dict(zip([desc[0] for desc in uow.cursor.description], uow.cursor.fetchone()))
        episodes = uow.memory.get_episodes(symbol, limit=1)
        stats = uow.calibration.get_stats_for_run(run_id)

    assert dec["checklist_json"]["market_regime"] == signal["checklist"]["market_regime"]
    assert dec["outcome_json"]["outcome_label"] == "target_hit"
    episode = episodes[0]
    for field in (
        "market_regime",
        "session_type",
        "gap_type",
        "structure_state",
        "vwap_relation",
        "profile_location",
        "price_location",
        "time_bucket",
        "volatility_bucket",
    ):
        assert episode[field]
    bucket_keys = {row["bucket_key"] for row in stats}
    assert "confidence_0.80" in bucket_keys
    assert "setup_breakout" in bucket_keys
    assert "setup_vwap_reclaim" in bucket_keys


def test_decision_transaction_rolls_back_when_order_persistence_fails(monkeypatch):
    symbol = "RELIANCE"
    run_id = _seed_run(symbol)
    T = datetime.now(timezone.utc)

    def fail_save_order(self, data):
        raise RuntimeError("forced order failure")

    monkeypatch.setattr(OrderRepository, "save_order", fail_save_order)

    with pytest.raises(RuntimeError, match="forced order failure"):
        DecisionTransactionService.process_decision(
            run_id, symbol, T, 2400.0, _buy_signal(), _buy_validation(quantity=3)
        )

    with UnitOfWork() as uow:
        uow.cursor.execute(
            "SELECT COUNT(*) FROM decisions WHERE run_id = %s AND decision_time = %s",
            (run_id, T),
        )
        decision_count = uow.cursor.fetchone()[0]
        uow.cursor.execute(
            "SELECT COUNT(*) FROM orders_simulated WHERE run_id = %s",
            (run_id,),
        )
        order_count = uow.cursor.fetchone()[0]
        uow.cursor.execute(
            "SELECT COUNT(*) FROM positions WHERE run_id = %s",
            (run_id,),
        )
        position_count = uow.cursor.fetchone()[0]
        uow.cursor.execute(
            "SELECT COUNT(*) FROM portfolio_snapshots WHERE run_id = %s AND decision_id IS NOT NULL",
            (run_id,),
        )
        decision_snapshot_count = uow.cursor.fetchone()[0]

    assert decision_count == 0
    assert order_count == 0
    assert position_count == 0
    assert decision_snapshot_count == 0


def test_audit_rows_can_be_linked_to_persisted_decision():
    symbol = "RELIANCE"
    run_id = _seed_run(symbol)
    T = datetime.now(timezone.utc)

    _, decision_id = DecisionTransactionService.process_decision(
        run_id, symbol, T, 2400.0, _buy_signal(), _buy_validation(quantity=3)
    )

    turn_id = AuditService.record_agent_turn(
        run_id=run_id,
        decision_id=decision_id,
        turn_number=0,
        role="assistant",
        raw_output='{"type": "analysis_plan"}',
        parsed_type="analysis_plan",
    )
    AuditService.record_tool_trace(
        run_id=run_id,
        decision_id=decision_id,
        turn_id=turn_id,
        round_num=1,
        tool_name="get_portfolio_state",
        arguments={},
        result={"cash_available": 100000.0},
    )
    AuditService.record_audit_event(
        run_id=run_id,
        decision_id=decision_id,
        event_type="TEST_AUDIT",
        message="linked audit event",
        symbol=symbol,
    )
    AuditService.record_trade_event(
        run_id=run_id,
        symbol=symbol,
        decision_id=decision_id,
        event_type="entry_requested",
        direction="BUY",
        price=2400.0,
        quantity=3,
    )

    with UnitOfWork() as uow:
        for table in ("agent_turn_records", "tool_call_traces", "audit_events", "trade_events"):
            uow.cursor.execute(
                f"SELECT COUNT(*) FROM {table} WHERE decision_id = %s",
                (decision_id,),
            )
            assert uow.cursor.fetchone()[0] == 1
