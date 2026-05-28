"""
Integration tests for the new Phase 1-8 modules:
- DataSnapshot hashing & versioning
- StockMetadata population
- SessionLevels lifecycle transitions
- Calibration tracking
- Cooldown with run_id/symbol

All tests must pass in CI.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import hashlib


# ============================================================
# Data Snapshot Tests
# ============================================================

class TestDataSnapshot:
    def test_hash_deterministic(self):
        """Same data should produce same hash."""
        from core.data_snapshot import DataSnapshotManager

        dates = pd.date_range("2026-05-01", "2026-05-28", freq="D")
        df1 = pd.DataFrame({
            "open": np.ones(28)*100, "high": np.ones(28)*110,
            "low": np.ones(28)*90, "close": np.ones(28)*105,
            "volume": np.ones(28)*1000
        }, index=dates)

        df2 = df1.copy()

        hash1 = DataSnapshotManager._hash_ohlcv(df1)
        hash2 = DataSnapshotManager._hash_ohlcv(df2)

        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 is 64 hex chars

    def test_hash_different_data(self):
        """Different data should produce different hash."""
        from core.data_snapshot import DataSnapshotManager

        dates = pd.date_range("2026-05-01", "2026-05-28", freq="D")
        df1 = pd.DataFrame({
            "open": np.ones(28)*100, "high": np.ones(28)*110,
            "low": np.ones(28)*90, "close": np.ones(28)*105,
            "volume": np.ones(28)*1000
        }, index=dates)

        df2 = df1.copy()
        df2.iloc[0, df2.columns.get_loc("close")] = 106.0

        hash1 = DataSnapshotManager._hash_ohlcv(df1)
        hash2 = DataSnapshotManager._hash_ohlcv(df2)

        assert hash1 != hash2

    def test_verify_hash(self):
        """Hash verification should work."""
        from core.data_snapshot import DataSnapshotManager

        dates = pd.date_range("2026-05-01", "2026-05-28", freq="D")
        df = pd.DataFrame({
            "open": np.ones(28)*100, "high": np.ones(28)*110,
            "low": np.ones(28)*90, "close": np.ones(28)*105,
            "volume": np.ones(28)*1000
        }, index=dates)

        expected_hash = DataSnapshotManager._hash_ohlcv(df)
        assert DataSnapshotManager.verify_hash(df, expected_hash)
        assert not DataSnapshotManager.verify_hash(df, "bad_hash")

    def test_create_snapshot_set(self):
        """Create a complete snapshot set with all three timeframes."""
        from core.data_snapshot import DataSnapshotManager

        mgr = DataSnapshotManager(symbol="RELIANCE")

        df_weekly = pd.DataFrame(
            {"open": [100], "high": [110], "low": [90], "close": [105], "volume": [1000]},
            index=pd.date_range("2026-05-01", periods=1, freq="W-MON")
        )
        df_daily = pd.DataFrame(
            {"open": [100], "high": [110], "low": [90], "close": [105], "volume": [1000]},
            index=pd.date_range("2026-05-26", periods=1, freq="D")
        )
        df_15m = pd.DataFrame(
            {"open": [100], "high": [110], "low": [90], "close": [105], "volume": [1000]},
            index=pd.date_range("2026-05-28 09:15", periods=1, freq="15min")
        )

        snapshot_set = mgr.create_snapshot_set(df_weekly, df_daily, df_15m)

        assert snapshot_set.is_complete()
        assert "weekly" in snapshot_set.snapshots
        assert "daily" in snapshot_set.snapshots
        assert "intraday_15min" in snapshot_set.snapshots

    def test_compare_equivalent_sets(self):
        """Two sets from same data should be equivalent."""
        from core.data_snapshot import DataSnapshotManager

        mgr = DataSnapshotManager(symbol="RELIANCE")

        df_w = pd.DataFrame({"open": [100], "high": [110], "low": [90], "close": [105], "volume": [1000]},
                           index=pd.date_range("2026-05-01", periods=1, freq="W-MON"))
        df_d = pd.DataFrame({"open": [100], "high": [110], "low": [90], "close": [105], "volume": [1000]},
                           index=pd.date_range("2026-05-26", periods=1, freq="D"))
        df_i = pd.DataFrame({"open": [100], "high": [110], "low": [90], "close": [105], "volume": [1000]},
                           index=pd.date_range("2026-05-28 09:15", periods=1, freq="15min"))

        set1 = mgr.create_snapshot_set(df_w.copy(), df_d.copy(), df_i.copy())
        set2 = mgr.create_snapshot_set(df_w.copy(), df_d.copy(), df_i.copy())

        comparison = mgr.compare_sets(set1.set_id, set2.set_id)
        assert comparison["are_equivalent"]


# ============================================================
# Stock Metadata Tests
# ============================================================

class TestStockMetadata:
    def test_adjustment_factor_no_splits(self):
        from core.stock_metadata import StockMetadataManager

        mgr = StockMetadataManager()
        factor = mgr._compute_adjustment_factor([])
        assert factor == 1.0

    def test_adjustment_factor_with_split(self):
        from core.stock_metadata import StockMetadataManager

        splits = [{"date": "2025-01-01", "ratio": 2.0}]  # 2:1 split
        factor = mgr._compute_adjustment_factor(splits)
        assert factor == 2.0

    def test_adjustment_factor_multiple_splits(self):
        from core.stock_metadata import StockMetadataManager

        splits = [
            {"date": "2024-01-01", "ratio": 2.0},
            {"date": "2025-01-01", "ratio": 5.0},
        ]
        factor = mgr._compute_adjustment_factor(splits)
        assert factor == 10.0  # 2.0 * 5.0

    def test_adjust_prices(self):
        from core.stock_metadata import StockMetadataManager
        from core.stock_metadata import StockMetadata

        mgr = StockMetadataManager()
        meta = StockMetadata(symbol="TEST", adjustment_factor=2.0)
        mgr._metadata["TEST"] = meta

        df = pd.DataFrame({
            "open": [100.0], "high": [110.0], "low": [90.0], "close": [105.0],
            "volume": [1000]
        }, index=pd.date_range("2026-05-01", periods=1, freq="D"))

        adjusted = mgr.adjust_prices("TEST", df)
        assert adjusted["close"].iloc[0] == 210.0

    def test_has_earnings_nearby(self):
        from core.stock_metadata import StockMetadataManager
        from core.stock_metadata import StockMetadata

        mgr = StockMetadataManager()
        meta = StockMetadata(
            symbol="TEST",
            earnings_dates=[{"date": "2026-05-25", "quarter": "Q1 2026"}]
        )
        mgr._metadata["TEST"] = meta

        decision_date = date(2026, 5, 28)
        assert mgr.has_earnings_nearby("TEST", decision_date, days_window=5)
        assert not mgr.has_earnings_nearby("TEST", decision_date, days_window=2)

    def test_get_corporate_action_dates(self):
        from core.stock_metadata import StockMetadataManager
        from core.stock_metadata import StockMetadata

        mgr = StockMetadataManager()
        meta = StockMetadata(
            symbol="TEST",
            dividend_dates=[
                {"date": "2026-05-15", "amount": 5.0, "type": "interim"},
                {"date": "2026-06-15", "amount": 3.0, "type": "final"},
            ]
        )
        mgr._metadata["TEST"] = meta

        actions = mgr.get_corporate_action_dates(
            "TEST", "dividend",
            date(2026, 5, 1), date(2026, 5, 31)
        )
        assert len(actions) == 1
        assert actions[0]["amount"] == 5.0


# ============================================================
# Session Levels Lifecycle Tests
# ============================================================

class TestSessionLevels:
    def test_active_to_tested(self):
        from core.session_levels import (
            LevelLifecycleManager, SessionLevel, LevelState
        )
        from datetime import datetime

        mgr = LevelLifecycleManager(atr=10.0)
        level = SessionLevel(
            level_id="lvl_001", price=2400.0,
            level_type="resistance"
        )

        # Candle that approaches the level (within atr*0.3 = 3.0)
        candle = {"open": 2395, "high": 2398, "low": 2390, "close": 2397, "volume": 5000}
        candle_time = datetime(2026, 5, 28, 10, 0)

        events = mgr.process_candle(candle, candle_time, [level])
        assert level.state == LevelState.TESTED
        assert len(events) == 1
        assert events[0].event_type == "LEVEL_TESTED"

    def test_no_transition_when_far(self):
        from core.session_levels import (
            LevelLifecycleManager, SessionLevel, LevelState
        )
        from datetime import datetime

        mgr = LevelLifecycleManager(atr=10.0)
        level = SessionLevel(
            level_id="lvl_001", price=2500.0,  # Far away
            level_type="resistance"
        )

        candle = {"open": 2395, "high": 2398, "low": 2390, "close": 2397, "volume": 5000}
        candle_time = datetime(2026, 5, 28, 10, 0)

        events = mgr.process_candle(candle, candle_time, [level])
        assert level.state == LevelState.ACTIVE  # No change
        assert len(events) == 0

    def test_tested_to_broken_resistance(self):
        from core.session_levels import (
            LevelLifecycleManager, SessionLevel, LevelState
        )
        from datetime import datetime

        mgr = LevelLifecycleManager(atr=10.0)
        level = SessionLevel(
            level_id="lvl_001", price=2400.0,
            level_type="resistance", state=LevelState.TESTED
        )

        # Candle closes above level by atr*0.2 = 2.0
        candle = {"open": 2398, "high": 2410, "low": 2395, "close": 2405, "volume": 5000}
        candle_time = datetime(2026, 5, 28, 10, 0)

        events = mgr.process_candle(candle, candle_time, [level])
        assert level.state == LevelState.BROKEN
        assert any(e.event_type == "LEVEL_BROKEN" for e in events)

    def test_tested_to_broken_support(self):
        from core.session_levels import (
            LevelLifecycleManager, SessionLevel, LevelState
        )
        from datetime import datetime

        mgr = LevelLifecycleManager(atr=10.0)
        level = SessionLevel(
            level_id="lvl_001", price=2400.0,
            level_type="support", state=LevelState.TESTED
        )

        # Candle closes below level by atr*0.2 = 2.0
        candle = {"open": 2402, "high": 2405, "low": 2385, "close": 2390, "volume": 5000}
        candle_time = datetime(2026, 5, 28, 10, 0)

        events = mgr.process_candle(candle, candle_time, [level])
        assert level.state == LevelState.BROKEN
        assert any(e.event_type == "LEVEL_BROKEN" for e in events)

    def test_invalidated_after_no_reaction(self):
        from core.session_levels import (
            LevelLifecycleManager, SessionLevel, LevelState
        )
        from datetime import datetime

        mgr = LevelLifecycleManager(atr=10.0)
        level = SessionLevel(
            level_id="lvl_001", price=2500.0,  # Far away
            level_type="resistance", state=LevelState.ACTIVE
        )

        # 3 candles far from level, then a 4th
        for i in range(4):
            candle = {"open": 2395, "high": 2398, "low": 2390, "close": 2397, "volume": 5000}
            candle_time = datetime(2026, 5, 28, 10, i * 15)
            events = mgr.process_candle(candle, candle_time, [level])

        assert level.state == LevelState.INVALIDATED
        assert any(e.event_type == "LEVEL_INVALIDATED" for e in events)

    def test_expire_all_levels(self):
        from core.session_levels import (
            LevelLifecycleManager, SessionLevel, LevelState
        )
        from datetime import datetime

        mgr = LevelLifecycleManager(atr=10.0)
        levels = [
            SessionLevel("lvl_1", 2400.0, "resistance", LevelState.ACTIVE),
            SessionLevel("lvl_2", 2380.0, "support", LevelState.TESTED),
            SessionLevel("lvl_3", 2450.0, "swing_high", LevelState.BROKEN),
        ]

        events = mgr.expire_all_levels(levels, datetime(2026, 5, 28, 15, 30))
        assert len(events) == 3
        assert all(lvl.state == LevelState.EXPIRED for lvl in levels)


# ============================================================
# Calibration Tests
# ============================================================

class TestCalibration:
    def test_confidence_buckets(self):
        from core.calibration import CalibrationTracker

        tracker = CalibrationTracker()
        assert tracker._confidence_bucket(0.15) == "0.00-0.30"
        assert tracker._confidence_bucket(0.45) == "0.30-0.50"
        assert tracker._confidence_bucket(0.65) == "0.50-0.70"
        assert tracker._confidence_bucket(0.80) == "0.70-0.85"
        assert tracker._confidence_bucket(0.95) == "0.85-1.00"

    def test_record_and_retrieve(self):
        from core.calibration import CalibrationTracker

        tracker = CalibrationTracker(min_samples_for_hint=3)
        
        # Record trades
        for i in range(5):
            is_win = i < 3
            tracker.record_outcome(
                action="BUY",
                outcome={"outcome": "target_hit" if is_win else "stop_hit"},
                setup_tags=["vwap_reclaim"],
                market_regime="trend",
                session_type="trend_day",
                confidence=0.65,
                net_r=2.0 if is_win else -1.0,
                is_win=is_win,
                reflection_level="HIGH",
            )

        hints = tracker.get_calibration_hints(
            setup_tags=["vwap_reclaim"],
            regime="trend",
            confidence=0.65,
        )
        assert "vwap_reclaim" in hints.lower() or "trend" in hints.lower()

    def test_skip_and_hold_tracking(self):
        from core.calibration import CalibrationTracker

        tracker = CalibrationTracker()

        tracker.record_outcome(
            action="SKIP",
            outcome={"skip_quality": "missed_long_opportunity"},
            setup_tags=[], market_regime="", session_type="",
            confidence=0.0, net_r=0.0, is_win=False,
        )
        tracker.record_outcome(
            action="HOLD",
            outcome={"hold_quality": "good_hold_avoided_chop"},
            setup_tags=[], market_regime="", session_type="",
            confidence=0.5, net_r=0.0, is_win=False,
        )
        tracker.record_outcome(
            action="HOLD",
            outcome={"hold_quality": "bad_hold_should_exit"},
            setup_tags=[], market_regime="", session_type="",
            confidence=0.5, net_r=0.0, is_win=False,
        )

        assert tracker.total_skips == 1
        assert tracker.total_holds == 2
        assert tracker.good_holds == 1
        assert tracker.bad_holds == 1
        assert tracker.missed_opportunities == 1

    def test_insufficient_samples_no_hint(self):
        from core.calibration import CalibrationTracker

        tracker = CalibrationTracker(min_samples_for_hint=20)
        
        # Only 2 trades - insufficient
        for i in range(2):
            tracker.record_outcome(
                action="BUY", outcome={"outcome": "target_hit"},
                setup_tags=["breakout"], market_regime="trend",
                session_type="trend_day", confidence=0.7,
                net_r=2.0, is_win=True, reflection_level="HIGH",
            )

        hints = tracker.get_calibration_hints(setup_tags=["breakout"])
        assert hints == ""

    def test_experiment_metrics(self):
        from core.calibration import CalibrationTracker

        tracker = CalibrationTracker()
        
        for i in range(10):
            is_win = i < 6
            tracker.record_outcome(
                action="BUY", outcome={},
                setup_tags=["breakout"], market_regime="trend",
                session_type="trend_day", confidence=0.7,
                net_r=2.0 if is_win else -1.0,
                is_win=is_win, reflection_level="HIGH",
            )

        metrics = tracker.get_experiment_metrics()
        assert metrics["total_trades"] == 10
        assert metrics["total_wins"] == 6
        assert metrics["win_rate"] == 0.6
        assert "breakout" in metrics.get("setup_performance", {})


# ============================================================
# Cooldown with Run ID Tests
# ============================================================

class TestCooldownWithRunId:
    def test_record_loss_with_ids(self):
        from core.cooldown import CooldownController, CooldownReason

        controller = CooldownController()
        controller.config.cooldown_after_stop_candles = 1
        controller.config.decision_interval_minutes = 15

        controller.record_loss(
            run_id="run_001",
            symbol="RELIANCE",
            direction="BUY",
            level_zone="2400",
        )

        # Should have a lock
        can_open, reason = controller.can_open_position("BUY")
        assert not can_open
        assert "Cooldown" in reason

    def test_record_win_with_ids(self):
        from core.cooldown import CooldownController

        controller = CooldownController()
        controller.config.cooldown_after_target_candles = 1

        controller.record_win(
            run_id="run_001",
            symbol="RELIANCE",
            direction="BUY",
        )

        can_open, reason = controller.can_open_position("BUY")
        assert not can_open  # Has cooldown

    def test_record_exit_with_ids(self):
        from core.cooldown import CooldownController

        controller = CooldownController()
        controller.record_exit(
            run_id="run_001",
            symbol="RELIANCE",
            direction="BUY",
        )

        assert controller._trades_today == 1


# ============================================================
# Schema Validation Tests (Extended)
# ============================================================

class TestSchemaExtended:
    def test_buy_requires_complete_dart(self):
        from agent.schema import validate_llm_output

        data = {
            "type": "final_signal", "action": "BUY", "confidence": 0.7,
            "dart": {"direction": "", "area": "test", "risk": "", "trigger": ""},
            "checklist": {
                "market_regime": "trend", "session_type": "trend_day",
                "structure_state": "bullish_bos",
                "location_quality": 4, "trigger_quality": 4, "risk_quality": 4,
                "volume_confirmation": 3, "higher_tf_alignment": 4,
            },
            "entry": 2400.0, "stop": 2370.0, "target": 2460.0,
            "net_reward_risk": 2.0, "expected_horizon_minutes": 45,
            "reason": "test", "invalidation": "test",
        }

        is_valid, parsed, errors = validate_llm_output(data)
        assert not is_valid
        assert any("DART" in str(e) or "dart" in str(e).lower() for e in errors)

    def test_skip_requires_reason_to_wait(self):
        from agent.schema import validate_llm_output

        data = {
            "type": "final_signal", "action": "SKIP", "confidence": 0.0,
            "dart": {"direction": "", "area": "", "risk": "", "trigger": ""},
            "checklist": {
                "market_regime": "range", "session_type": "range_day",
                "structure_state": "range_bound",
                "location_quality": 2, "trigger_quality": 1, "risk_quality": 2,
                "volume_confirmation": 1, "higher_tf_alignment": 2,
                # Missing reason_to_wait
            },
            "reason": "No trade",
        }

        is_valid, parsed, errors = validate_llm_output(data)
        assert not is_valid
        assert any("reason_to_wait" in str(e) for e in errors)


# ============================================================
# Context Window Extended Tests
# ============================================================

class TestContextWindowExtended:
    def test_include_partial_weekly(self):
        from core.context_window import get_completed_weekly_context
        import pytz
        IST = pytz.timezone("Asia/Kolkata")

        T = IST.localize(datetime(2026, 5, 28, 14, 0))  # Thursday
        dates = pd.date_range("2026-04-01", "2026-05-28", freq="W-MON", tz="Asia/Kolkata")
        df = pd.DataFrame(
            {"open": 100, "high": 110, "low": 90, "close": 105, "volume": 1000},
            index=dates
        )

        # Without partial
        completed, partial, has_partial = get_completed_weekly_context(
            df, T, months=3, include_partial=False
        )
        assert has_partial == True

        # With partial - should include partial in completed
        completed_with, _, _ = get_completed_weekly_context(
            df, T, months=3, include_partial=True
        )
        assert len(completed_with) >= len(completed)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
