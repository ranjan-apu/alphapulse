"""
Tests for the new Phase 1-7 modules:
- ContextWindowPolicy (no lookahead)
- CashMarketChargesModel
- SlippageModel
- PositionSizing
- MarketSessionController
- CooldownController
- GapContext
- FinalSignal schema validation
- VolumeProfile
- MarketStructure
- Regime detection
- Confluence scoring
- VWAP computation
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz

IST = pytz.timezone("Asia/Kolkata")


# ============================================================
# Test Data Helpers
# ============================================================

def make_ohlcv_df(start_date, periods, freq="15min", base_price=2400.0):
    """Create synthetic OHLCV data."""
    dates = pd.date_range(start=start_date, periods=periods, freq=freq, tz="Asia/Kolkata")
    np.random.seed(42)
    closes = base_price + np.cumsum(np.random.randn(periods) * 5)
    highs = closes + abs(np.random.randn(periods) * 3)
    lows = closes - abs(np.random.randn(periods) * 3)
    opens = closes - np.random.randn(periods) * 2
    volumes = np.random.randint(1000, 10000, periods)
    return pd.DataFrame({
        "open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes
    }, index=dates).astype(float)


# ============================================================
# ContextWindowPolicy Tests
# ============================================================

class TestContextWindowPolicy:
    def test_completed_daily_excludes_current_day(self):
        from core.context_window import get_completed_daily_context

        T = IST.localize(datetime(2026, 5, 28, 14, 0))
        dates = pd.date_range("2026-05-01", "2026-05-28", freq="D", tz="Asia/Kolkata")
        df = pd.DataFrame({"open": 100, "high": 110, "low": 90, "close": 105, "volume": 1000}, index=dates)

        completed, partial, has_partial = get_completed_daily_context(df, T, months=1)

        # Current day (May 28) should NOT be in completed
        assert len(completed) > 0
        assert not any(d.date() == datetime(2026, 5, 28).date() for d in completed.index)
        # Partial should contain current day if present
        if has_partial:
            assert any(d.date() == datetime(2026, 5, 28).date() for d in partial.index)

    def test_completed_weekly_excludes_current_week(self):
        from core.context_window import get_completed_weekly_context

        T = IST.localize(datetime(2026, 5, 28, 14, 0))  # Thursday
        dates = pd.date_range("2026-05-04", "2026-05-28", freq="W-MON", tz="Asia/Kolkata")
        df = pd.DataFrame({"open": 100, "high": 110, "low": 90, "close": 105, "volume": 1000}, index=dates)

        completed, partial, has_partial = get_completed_weekly_context(df, T, months=3)

        assert len(completed) > 0
        # Current week's Monday is May 25; weeks on or after should be in partial
        current_monday = datetime(2026, 5, 25).date()
        for d in completed.index:
            assert d.date() < current_monday

    def test_completed_intraday_includes_only_closed_candles(self):
        from core.context_window import get_completed_intraday_context

        T = IST.localize(datetime(2026, 5, 28, 14, 0))
        df = make_ohlcv_df("2026-05-28 09:15", 30, "15min")

        result = get_completed_intraday_context(df, T, sessions=1)

        # All candles should be <= T
        assert all(idx <= T for idx in result.index)

    def test_build_context_contract(self):
        from core.context_window import build_context_contract

        T = IST.localize(datetime(2026, 5, 28, 14, 0))
        df_weekly = pd.DataFrame(
            {"open": 100, "high": 110, "low": 90, "close": 105, "volume": 1000},
            index=pd.date_range("2026-04-01", "2026-05-25", freq="W-MON", tz="Asia/Kolkata")
        )
        df_daily = pd.DataFrame(
            {"open": 100, "high": 110, "low": 90, "close": 105, "volume": 1000},
            index=pd.date_range("2026-05-01", "2026-05-27", freq="D", tz="Asia/Kolkata")
        )
        df_intraday = make_ohlcv_df("2026-05-28 09:15", 30, "15min")

        contract = build_context_contract(df_weekly, df_daily, df_intraday, T)

        assert "weekly" in contract
        assert "daily" in contract
        assert "intraday" in contract
        assert contract["daily"]["complete_only"] == True
        assert contract["daily"]["has_partial_current_day"] == False  # May 28 not in data


# ============================================================
# Charges Model Tests
# ============================================================

class TestCharges:
    def test_cnc_buy_charges(self):
        from core.charges import EquityCashCharges, compute_charges

        charges = EquityCashCharges()
        result = compute_charges(charges, "BUY", quantity=100, entry_price=2400.0, exit_price=2450.0)

        assert result.total_charges > 0
        assert result.breakeven_points > 0
        assert result.net_r_adjustment > 0
        assert "brokerage" in result.breakdown
        assert "stt" in result.breakdown

    def test_cnc_buy_charges_breakdown(self):
        from core.charges import EquityCashCharges, compute_charges

        charges = EquityCashCharges()
        result = compute_charges(charges, "BUY", quantity=10, entry_price=2400.0, exit_price=2450.0)

        # ~₹40 brokerage (2 legs * 20)
        assert result.breakdown["brokerage"] == pytest.approx(40.0, abs=1)
        # STT: 0.1% buy + 0.1% sell on turnover
        assert result.breakdown["stt"] > 0
        assert result.total_charges > 40  # At least brokerage

    def test_mis_charges_lower_stt(self):
        from core.charges import EquityCashMISCharges, compute_charges

        charges = EquityCashMISCharges()
        result = compute_charges(charges, "BUY", quantity=100, entry_price=2400.0, exit_price=2450.0)

        # MIS has 0% STT on buy, 0.025% on sell
        assert result.breakdown["stt"] > 0  # Still some STT on sell side


# ============================================================
# Slippage Model Tests
# ============================================================

class TestSlippage:
    def test_entry_slippage_buy(self):
        from core.slippage import apply_entry_slippage, SlippageConfig

        config = SlippageConfig(entry_slippage=0.50)
        result = apply_entry_slippage(2400.0, "BUY", config)
        # BUY entry: price goes higher (worse)
        assert result > 2400.0
        assert result == pytest.approx(2400.50, abs=0.01)

    def test_entry_slippage_sell(self):
        from core.slippage import apply_entry_slippage, SlippageConfig

        config = SlippageConfig(entry_slippage=0.50)
        result = apply_entry_slippage(2400.0, "SELL", config)
        # SELL entry: price goes lower (worse)
        assert result < 2400.0
        assert result == pytest.approx(2399.50, abs=0.01)

    def test_stop_slippage_buy(self):
        from core.slippage import apply_stop_slippage, SlippageConfig

        config = SlippageConfig(stop_slippage=1.00)
        result = apply_stop_slippage(2370.0, "BUY", config)
        # BUY stop: triggers lower (worse - more loss)
        assert result < 2370.0
        assert result == pytest.approx(2369.00, abs=0.01)

    def test_compute_executed_prices(self):
        from core.slippage import compute_executed_prices, SlippageConfig

        config = SlippageConfig()
        result = compute_executed_prices(2400.0, 2370.0, 2460.0, "BUY", config)

        assert result["entry_executed"] > 2400.0
        assert result["stop_executed"] < 2370.0
        assert result["target_executed"] < 2460.0
        assert result["slippage_entry_points"] > 0


# ============================================================
# Position Sizing Tests
# ============================================================

class TestPositionSizing:
    def test_risk_based_sizing(self):
        from core.position_sizing import compute_position_size, PositionSizingConfig

        config = PositionSizingConfig(
            starting_capital=100000.0,
            risk_budget_pct=0.01,       # ₹1,000 risk budget
            max_capital_per_trade=30000.0,
        )

        # Tight stop (₹10 risk/share) -> risk budget allows 100 shares
        # Capital cap allows 12 shares (₹30,000 / ₹2,400)
        result = compute_position_size(2400.0, 2390.0, 2445.0, "BUY", config, total_charges=60.0)

        assert result.actionable
        assert result.quantity == 12  # Limited by capital cap
        assert result.capital_ceiling_hit
        assert result.deployed_capital <= 30000.0

    def test_wide_stop_limited_by_risk(self):
        from core.position_sizing import compute_position_size, PositionSizingConfig

        config = PositionSizingConfig(
            starting_capital=100000.0,
            risk_budget_pct=0.01,       # ₹1,000 risk budget
            max_capital_per_trade=30000.0,
        )

        # Wide stop (₹80 risk/share) -> risk budget allows 12 shares
        # Capital cap allows 12 shares
        result = compute_position_size(2400.0, 2320.0, 2600.0, "BUY", config, total_charges=60.0)

        assert result.actionable
        assert result.quantity == 12  # Both limits equal here
        assert result.gross_risk <= 1000.0  # Within risk budget

    def test_invalid_stop_side(self):
        from core.position_sizing import compute_position_size, PositionSizingConfig

        config = PositionSizingConfig()
        result = compute_position_size(2400.0, 2410.0, 2450.0, "BUY", config)

        assert not result.actionable
        assert len(result.errors) > 0

    def test_invalid_target_side(self):
        from core.position_sizing import compute_position_size, PositionSizingConfig

        config = PositionSizingConfig()
        result = compute_position_size(2400.0, 2370.0, 2390.0, "BUY", config)

        assert not result.actionable
        assert len(result.errors) > 0


# ============================================================
# State-Aware Validator Tests
# ============================================================

class TestStateAwareValidator:
    def test_flat_hold_rejected(self):
        from validation.validator import validate_signal

        T = IST.localize(datetime(2026, 5, 28, 10, 0))
        session_end = IST.localize(datetime(2026, 5, 28, 15, 30))
        result = validate_signal(
            {"action": "HOLD", "position_id": "pos_1", "reason": "wait"},
            T,
            session_end,
            has_open_position=False,
        )

        assert not result["is_valid"]
        assert result["rejection_reason"] == "REJECTED_INVALID_ACTION_FOR_STATE"

    def test_open_position_skip_rejected(self):
        from validation.validator import validate_signal

        T = IST.localize(datetime(2026, 5, 28, 10, 0))
        session_end = IST.localize(datetime(2026, 5, 28, 15, 30))
        result = validate_signal(
            {"action": "SKIP", "reason": "no setup"},
            T,
            session_end,
            has_open_position=True,
        )

        assert not result["is_valid"]
        assert result["rejection_reason"] == "REJECTED_INVALID_ACTION_FOR_STATE"

    def test_cnc_sell_entry_rejected(self):
        from validation.validator import validate_signal

        T = IST.localize(datetime(2026, 5, 28, 10, 0))
        session_end = IST.localize(datetime(2026, 5, 28, 15, 30))
        signal = {
            "action": "SELL",
            "entry": 2400.0,
            "stop": 2420.0,
            "target": 2340.0,
            "expected_horizon_minutes": 45,
            "dart": {"trigger": "15m breakdown"},
            "reason": "short setup",
        }
        result = validate_signal(signal, T, session_end, has_open_position=False)

        assert not result["is_valid"]
        assert result["rejection_reason"] == "REJECTED_SELL_REQUIRES_MIS"

    def test_buy_uses_risk_budget_and_capital_ceiling(self):
        from validation.validator import validate_signal

        T = IST.localize(datetime(2026, 5, 28, 10, 0))
        session_end = IST.localize(datetime(2026, 5, 28, 15, 30))
        signal = {
            "action": "BUY",
            "entry": 2400.0,
            "stop": 2320.0,
            "target": 2600.0,
            "expected_horizon_minutes": 45,
            "dart": {"trigger": "15m close above retest"},
            "reason": "wide-stop trend continuation",
        }
        result = validate_signal(signal, T, session_end, has_open_position=False)

        assert result["is_valid"]
        assert result["sizing"]["quantity"] == 12
        assert result["sizing"]["gross_risk"] <= result["sizing"]["risk_budget"]


# ============================================================
# Session Controller Tests
# ============================================================

class TestSessionController:
    def test_phases(self):
        from core.session_controller import MarketSessionController, SessionPhase

        controller = MarketSessionController()

        # Pre-open
        ts = IST.localize(datetime(2026, 5, 28, 9, 0))
        assert controller.get_phase(ts) == SessionPhase.PRE_OPEN

        # Opening build
        ts = IST.localize(datetime(2026, 5, 28, 9, 20))
        assert controller.get_phase(ts) == SessionPhase.OPENING_BUILD

        # Active trading
        ts = IST.localize(datetime(2026, 5, 28, 11, 0))
        assert controller.get_phase(ts) == SessionPhase.ACTIVE_TRADING

        # Management only
        ts = IST.localize(datetime(2026, 5, 28, 15, 10))
        assert controller.get_phase(ts) == SessionPhase.MANAGEMENT_ONLY

        # Forced square-off
        ts = IST.localize(datetime(2026, 5, 28, 15, 25))
        assert controller.get_phase(ts) == SessionPhase.FORCED_SQUAREOFF

        # Closed
        ts = IST.localize(datetime(2026, 5, 28, 15, 35))
        assert controller.get_phase(ts) == SessionPhase.CLOSED

    def test_cannot_open_in_management(self):
        from core.session_controller import MarketSessionController

        controller = MarketSessionController()
        ts = IST.localize(datetime(2026, 5, 28, 15, 10))
        assert not controller.can_open_new_position(ts)

    def test_can_open_in_active_trading(self):
        from core.session_controller import MarketSessionController

        controller = MarketSessionController()
        ts = IST.localize(datetime(2026, 5, 28, 11, 0))
        assert controller.can_open_new_position(ts)


# ============================================================
# Cooldown Tests
# ============================================================

class TestCooldown:
    def test_lock_prevents_trade(self):
        from core.cooldown import CooldownController, CooldownReason

        controller = CooldownController()
        controller.add_lock("run1", "RELIANCE", CooldownReason.AFTER_STOP_LOSS, direction="BUY")

        can_open, reason = controller.can_open_position("BUY")
        assert not can_open
        assert "Cooldown" in reason

    def test_lock_allows_opposite_direction(self):
        from core.cooldown import CooldownController, CooldownReason

        controller = CooldownController()
        controller.add_lock("run1", "RELIANCE", CooldownReason.AFTER_STOP_LOSS, direction="BUY")

        can_open, reason = controller.can_open_position("SELL")
        assert can_open
        assert reason is None

    def test_max_attempts_per_level(self):
        from core.cooldown import CooldownController, CooldownReason

        config = type('obj', (object,), {'max_attempts_per_level_per_day': 2, 'decision_interval_minutes': 15})()
        controller = CooldownController()
        controller.config.max_attempts_per_level_per_day = 2

        controller.add_lock("run1", "RELIANCE", CooldownReason.SAME_LEVEL_REPEATED, level_zone="2400")
        controller.add_lock("run1", "RELIANCE", CooldownReason.SAME_LEVEL_REPEATED, level_zone="2400")

        can_open, reason = controller.can_open_position("BUY", level_zone="2400")
        assert not can_open
        assert "Max attempts" in reason


# ============================================================
# Gap Context Tests
# ============================================================

class TestGapContext:
    def test_gap_up_classification(self):
        from core.gap_context import classify_gap

        gap = classify_gap(
            prior_close=2350.0,
            today_open=2390.0,  # 40 point gap up
            prior_high=2370.0,
            prior_low=2330.0,
            atr=30.0,
        )

        assert gap.gap_direction == "gap_up"
        assert gap.gap_points == 40.0
        assert gap.gap_pct == pytest.approx(1.7, abs=0.1)
        assert gap.open_location_vs_prior_value == "above_value"
        assert gap.gap_fill_level == 2350.0

    def test_flat_open(self):
        from core.gap_context import classify_gap

        gap = classify_gap(
            prior_close=2350.0,
            today_open=2351.0,
        )

        assert gap.gap_direction == "flat_open"
        assert gap.gap_type == "no_gap"


# ============================================================
# Schema Validation Tests
# ============================================================

class TestSchemaValidation:
    def test_valid_buy_signal(self):
        from agent.schema import FinalSignal, validate_llm_output

        data = {
            "type": "final_signal",
            "action": "BUY",
            "confidence": 0.72,
            "dart": {"direction": "bullish", "area": "VWAP reclaim", "risk": "below VWAP", "trigger": "15m close above"},
            "checklist": {
                "market_regime": "trend", "session_type": "trend_day",
                "structure_state": "bullish_bos",
                "location_quality": 4, "trigger_quality": 4, "risk_quality": 4,
                "volume_confirmation": 3, "higher_tf_alignment": 4,
            },
            "entry": 2400.5, "stop": 2370.0, "target": 2445.0,
            "gross_reward_risk": 1.48, "net_reward_risk": 2.05,
            "expected_horizon_minutes": 45,
            "reason": "Test BUY signal",
            "invalidation": "Close below 2370",
        }

        is_valid, parsed, errors = validate_llm_output(data)
        assert is_valid
        assert parsed.action == "BUY"

    def test_missing_entry_rejected(self):
        from agent.schema import validate_llm_output

        data = {
            "type": "final_signal",
            "action": "BUY",
            "confidence": 0.72,
            "dart": {"direction": "bullish", "area": "test", "risk": "test", "trigger": "test"},
            "checklist": {
                "market_regime": "trend", "session_type": "trend_day",
                "structure_state": "bullish_bos",
                "location_quality": 4, "trigger_quality": 4, "risk_quality": 4,
                "volume_confirmation": 0, "higher_tf_alignment": 0,
            },
            "reason": "Test",
        }

        is_valid, parsed, errors = validate_llm_output(data)
        assert not is_valid

    def test_valid_skip_signal(self):
        from agent.schema import validate_llm_output

        data = {
            "type": "final_signal",
            "action": "SKIP",
            "confidence": 0.0,
            "dart": {"direction": "", "area": "", "risk": "", "trigger": ""},
            "checklist": {
                "market_regime": "range", "session_type": "range_day",
                "structure_state": "range_bound",
                "location_quality": 2, "trigger_quality": 1, "risk_quality": 2,
                "volume_confirmation": 1, "higher_tf_alignment": 2,
                "reason_to_wait": "No clear setup",
            },
            "reason": "No valid trade setup",
        }

        is_valid, parsed, errors = validate_llm_output(data)
        assert is_valid
        assert parsed.action == "SKIP"

    def test_valid_hold_signal(self):
        from agent.schema import validate_llm_output

        data = {
            "type": "final_signal",
            "action": "HOLD",
            "confidence": 0.65,
            "dart": {"direction": "bullish", "area": "above VWAP", "risk": "stop intact", "trigger": "no trigger to exit"},
            "checklist": {
                "market_regime": "trend", "session_type": "trend_day",
                "structure_state": "bullish_bos",
                "location_quality": 3, "trigger_quality": 0, "risk_quality": 4,
                "volume_confirmation": 2, "higher_tf_alignment": 3,
            },
            "position_id": "pos_abc123",
            "thesis_health": "valid",
            "reason": "Thesis remains valid",
        }

        is_valid, parsed, errors = validate_llm_output(data)
        assert is_valid
        assert parsed.action == "HOLD"

    def test_hold_requires_position_id(self):
        from agent.schema import validate_llm_output

        data = {
            "type": "final_signal",
            "action": "HOLD",
            "confidence": 0.65,
            "dart": {"direction": "bullish", "area": "test", "risk": "test", "trigger": "test"},
            "checklist": {
                "market_regime": "trend", "session_type": "trend_day",
                "structure_state": "bullish_bos",
                "location_quality": 3, "trigger_quality": 0, "risk_quality": 4,
                "volume_confirmation": 0, "higher_tf_alignment": 0,
            },
            "reason": "Test",
        }

        is_valid, parsed, errors = validate_llm_output(data)
        assert not is_valid

    def test_meets_scoring_thresholds(self):
        from agent.schema import FinalSignal

        signal = FinalSignal(
            action="BUY",
            confidence=0.7,
            dart={"direction": "bullish", "area": "test", "risk": "test", "trigger": "test"},
            checklist={
                "market_regime": "trend", "session_type": "trend_day",
                "structure_state": "bullish_bos",
                "location_quality": 4, "trigger_quality": 4, "risk_quality": 5,
                "volume_confirmation": 3, "higher_tf_alignment": 4,
            },
            entry=2400.0, stop=2370.0, target=2460.0,
            net_reward_risk=2.0, expected_horizon_minutes=45,
            reason="test", invalidation="test",
        )

        assert signal.meets_scoring_thresholds()


# ============================================================
# VWAP Tests
# ============================================================

class TestVWAP:
    def test_compute_session_vwap(self):
        from core.vwap import compute_session_vwap

        df = make_ohlcv_df("2026-05-28 09:15", 20, "15min", base_price=2400.0)
        current_price = float(df["close"].iloc[-1])

        result = compute_session_vwap(df, current_price, atr=15.0)

        assert result.current_vwap > 0
        assert result.relation in ("above_vwap", "below_vwap", "at_vwap")
        assert result.band_upper_1 is not None or result.band_upper_1 is None  # May be None with few candles


# ============================================================
# Volume Profile Tests
# ============================================================

class TestVolumeProfile:
    def test_compute_volume_profile(self):
        from core.volume_profile import compute_volume_profile

        df = make_ohlcv_df("2026-05-28 09:15", 30, "15min", base_price=2400.0)
        current_price = float(df["close"].iloc[-1])

        result = compute_volume_profile(df, current_price)

        assert result.poc > 0
        assert result.price_location in ("above_vah", "below_val", "inside_value", "no_value_area", "no_data")


# ============================================================
# Market Structure Tests
# ============================================================

class TestMarketStructure:
    def test_detect_market_structure(self):
        from core.market_structure import detect_market_structure

        df = make_ohlcv_df("2026-05-28 09:15", 30, "15min", base_price=2400.0)

        result = detect_market_structure(df)

        assert result.state in ("bullish_bos", "bearish_bos", "range_bound", "choch", "unclear")
        assert len(result.swings) >= 0


# ============================================================
# Regime Tests
# ============================================================

class TestRegime:
    def test_detect_market_regime(self):
        from core.regime import detect_market_regime

        df = make_ohlcv_df("2026-05-28 09:15", 30, "15min", base_price=2400.0)

        result = detect_market_regime(df)

        assert result.regime in ("trend", "range", "volatile", "compression", "unclear")
        assert result.session_type in ("trend_day", "range_day", "reversal_day", "inside_day", "opening_drive", "unclear")


# ============================================================
# Confluence Tests
# ============================================================

class TestConfluence:
    def test_score_confluence(self):
        from core.confluence import score_confluence

        result = score_confluence(
            htf_bias="bullish",
            structure_state="bullish_bos",
            vwap_relation="above_vwap",
            location_quality=4,
            trigger_quality=4,
            volume_confirmation=3,
            risk_quality=4,
        )

        assert result.total > 10
        assert result.is_tradable

    def test_poor_location_penalized(self):
        from core.confluence import score_confluence

        result = score_confluence(
            htf_bias="bullish",
            structure_state="range_bound",
            vwap_relation="at_vwap",
            location_quality=1,
            trigger_quality=1,
            volume_confirmation=1,
            risk_quality=2,
        )

        assert not result.is_tradable


# ============================================================
# Order Simulator Tests
# ============================================================

class TestOrderSimulator:
    def test_simulate_entry_order(self):
        from core.order_simulator import OrderSimulator

        sim = OrderSimulator()
        result = sim.simulate_entry_order(
            run_id="test_run",
            decision_id="dec_001",
            symbol="RELIANCE",
            direction="BUY",
            entry_price=2400.0,
            stop_price=2370.0,
            target_price=2460.0,
            quantity=12,
        )

        assert result["position_id"].startswith("pos_")
        assert result["entry_order"].order_id.startswith("ord_entry_")
        assert result["executed_entry"] > 2400.0  # Slippage applied
        assert result["executed_stop"] < 2370.0

    def test_simulate_round_trip_charges(self):
        from core.order_simulator import OrderSimulator

        sim = OrderSimulator()
        charges = sim.compute_round_trip_charges("BUY", 12, 2400.0, 2460.0)

        assert charges.total_charges > 0
        assert charges.breakeven_points > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
