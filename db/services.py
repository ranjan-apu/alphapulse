"""
Services coordinating business operations and repository logic.
"""
import uuid
import json
import logging
from dataclasses import asdict
from datetime import datetime, date, timezone, time
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd

from config import config
from db.unit_of_work import UnitOfWork
from core.data_snapshot import DataSnapshotManager
from core.session_controller import MarketSessionController, SessionPhase
from core.order_simulator import OrderSimulator
from core.slippage import SlippageConfig
from core.summarizer import compute_atr, detect_swings
from core.gap_context import classify_gap
from core.vwap import compute_session_vwap
from core.volume_profile import compute_volume_profile
from core.session_levels import LevelLifecycleManager, SessionLevel, LevelState
from agent.reflection import ReflectionWriter

logger = logging.getLogger(__name__)

class RunBootstrapService:
    @staticmethod
    def create_or_resume_run(
        run_id: str,
        symbol: str,
        starting_capital: float = 100000.0,
        max_capital_per_trade: float = 30000.0,
        risk_budget_pct: float = 0.01,
        max_daily_loss: float = 3000.0,
        max_trades_per_day: int = 5,
        notes: str = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        data_snapshot_set_id: Optional[str] = None,
    ) -> str:
        """Seed or resume a run, creating experiment_run and seeding initial portfolio snapshot."""
        with UnitOfWork() as uow:
            run = uow.runs.get_experiment_run(run_id)
            if run:
                logger.info(f"Resuming existing run: {run_id}")
                return run_id

            logger.info(f"Creating new run: {run_id}")
            # Insert to experiment_runs
            run_data = {
                "run_id": run_id,
                "symbol": symbol,
                "instrument_type": config.INSTRUMENT_TYPE,
                "product_type": config.PRODUCT_TYPE,
                "decision_interval": config.DECISION_INTERVAL,
                "start_date": start_date or date.today(),
                "end_date": end_date or start_date or date.today(),
                "data_snapshot_set_id": data_snapshot_set_id,
                "starting_capital": starting_capital,
                "max_capital_per_trade": max_capital_per_trade,
                "risk_budget_per_trade": starting_capital * risk_budget_pct,
                "max_daily_loss": max_daily_loss,
                "max_trades_per_day": max_trades_per_day,
                "agent_version": "dart-pa-v2",
                "prompt_version": "pa-checklist-v1",
                "toolset_version": "structure-vwap-profile-v1",
                "status": "running",
                "notes": notes,
                "config_snapshot": {
                    "STATE_BACKEND": config.STATE_BACKEND,
                    "AGENT_WORKFLOW": config.AGENT_WORKFLOW,
                    "PRODUCT_TYPE": config.PRODUCT_TYPE
                }
            }
            uow.runs.save_experiment_run(run_data)

            # Seed first portfolio snapshot
            snapshot_id = f"snap_init_{run_id}"
            portfolio_data = {
                "snapshot_id": snapshot_id,
                "run_id": run_id,
                "timestamp": datetime.now(timezone.utc),
                "starting_capital": starting_capital,
                "cash_available": starting_capital,
                "capital_deployed": 0.0,
                "capital_reserved": 0.0,
                "realized_pnl": 0.0,
                "unrealized_pnl": 0.0,
                "charges_paid": 0.0,
                "max_capital_per_trade": max_capital_per_trade,
                "max_daily_loss": max_daily_loss,
                "daily_loss_used": 0.0,
                "trades_taken_today": 0,
                "max_trades_per_day": max_trades_per_day,
            }
            uow.portfolio.save_portfolio_snapshot(portfolio_data)
            return run_id

    @staticmethod
    def create_snapshot_set(
        symbol: str,
        df_weekly: pd.DataFrame,
        df_daily: pd.DataFrame,
        df_15m: pd.DataFrame,
        source: str = "yahoo_finance",
        run_id: Optional[str] = None,
    ) -> str:
        """Create and hash a complete data snapshot set."""
        set_id = f"set_{symbol}_{int(datetime.now().timestamp())}"

        with UnitOfWork() as uow:
            uow.snapshots.save_snapshot_set({
                "set_id": set_id,
                "symbol": symbol,
                "source": source,
                "adjusted_for_splits": True,
                "adjusted_for_dividends": True,
                "notes": "Replay data snapshot set"
            })

            # Save timeframe snapshots
            for tf, df, label in [("weekly", df_weekly, "weekly"), ("daily", df_daily, "daily"), ("intraday_15min", df_15m, "intraday_15min")]:
                uow.snapshots.save_snapshot({
                    "snapshot_id": f"snap_{tf}_{set_id}",
                    "set_id": set_id,
                    "timeframe": label,
                    "period_start": df.index[0].date(),
                    "period_end": df.index[-1].date(),
                    "candle_count": len(df),
                    "first_candle": df.index[0],
                    "last_candle": df.index[-1],
                    "data_hash": DataSnapshotManager._hash_ohlcv(df),
                    "yfinance_period": "custom"
                })
            if run_id:
                uow.runs.update_experiment_run(run_id, {"data_snapshot_set_id": set_id})
        return set_id


class ReplayStateService:
    @staticmethod
    def get_latest_portfolio_snapshot(run_id: str) -> Optional[Dict[str, Any]]:
        with UnitOfWork() as uow:
            return uow.portfolio.get_latest_snapshot(run_id)

    @staticmethod
    def get_active_position(run_id: str, symbol: str) -> Optional[Dict[str, Any]]:
        with UnitOfWork() as uow:
            return uow.positions.get_active_position(run_id, symbol)

    @staticmethod
    def get_session_phase(T: datetime) -> SessionPhase:
        controller = MarketSessionController()
        return controller.get_phase(T)

    @staticmethod
    def get_cooldown_locks(run_id: str, symbol: str, T: datetime) -> List[Dict[str, Any]]:
        with UnitOfWork() as uow:
            return uow.locks.get_active_locks(run_id, symbol, T)

    @staticmethod
    def should_evaluate(run_id: str, symbol: str, T: datetime) -> bool:
        """
        Check if we should evaluate signals at T.
        Enforces min time interval between entry decisions, and cooldown trade locks.
        """
        active_pos = ReplayStateService.get_active_position(run_id, symbol)
        if active_pos:
            return True # Always check to manage open position

        # Enforce minutes between decisions
        with UnitOfWork() as uow:
            uow.cursor.execute(
                "SELECT decision_time FROM decisions WHERE run_id = %s AND symbol = %s ORDER BY decision_time DESC LIMIT 1",
                (run_id, symbol)
            )
            row = uow.cursor.fetchone()
            if row:
                last_time = row[0]
                elapsed = (T - last_time).total_seconds()
                if elapsed < config.MIN_MINUTES_BETWEEN_SIGNALS * 60:
                    return False

            # Check cooldown Locks in trade_locks table
            active_locks = uow.locks.get_active_locks(run_id, symbol, T)
            if active_locks:
                return False

        return True


class DecisionTransactionService:
    @staticmethod
    def process_decision(
        run_id: str,
        symbol: str,
        T: datetime,
        current_price: float,
        signal: Dict[str, Any],
        validation_result: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], str]:
        """
        Coordinate decision, orders, positions, and portfolio state updates inside ONE transaction.
        """
        decision_id = f"dec_{uuid.uuid4().hex[:12]}"
        
        with UnitOfWork() as uow:
            # 1. Get latest portfolio state
            portfolio = uow.portfolio.get_latest_snapshot(run_id)
            if not portfolio:
                raise ValueError(f"No portfolio state found for run: {run_id}")

            # Reconcile realized P&L / charges / capital
            cash_available = portfolio["cash_available"]
            capital_deployed = portfolio["capital_deployed"]
            realized_pnl = portfolio["realized_pnl"]
            charges_paid = portfolio["charges_paid"]
            trades_taken_today = portfolio["trades_taken_today"]

            action = validation_result.get("action", "SKIP")
            is_valid = validation_result.get("is_valid", False)
            rejection_reason = validation_result.get("rejection_reason")
            active_pos = uow.positions.get_active_position(run_id, symbol)
            
            position_id = active_pos["position_id"] if active_pos else None

            # Initialize variables to record changes
            deployed_diff = 0.0
            cash_diff = 0.0
            pnl_realized_diff = 0.0
            charges_diff = 0.0

            position_to_save = None
            position_to_update = None
            order_to_save = None
            lock_to_save = None

            # CNC BUY Execution
            if is_valid and action == "BUY" and not active_pos:
                sizing = validation_result.get("sizing", {})
                qty = int(sizing.get("quantity", 0))
                req_entry = float(sizing.get("entry", current_price))
                req_stop = float(sizing.get("stop", 0))
                req_target = float(sizing.get("target", 0))

                entry_result = OrderSimulator(
                    slippage_config=SlippageConfig()
                ).simulate_entry_order(
                    run_id=run_id,
                    decision_id=decision_id,
                    symbol=symbol,
                    direction="BUY",
                    entry_price=req_entry,
                    stop_price=req_stop,
                    target_price=req_target,
                    quantity=qty,
                    product_type=config.PRODUCT_TYPE,
                )

                exec_entry = float(entry_result["executed_entry"])
                charges_entry = float(entry_result["charges_entry"].get("total", 0.0))
                position_id = entry_result["position_id"]

                deployed_capital = qty * exec_entry

                order_data = asdict(entry_result["entry_order"])
                order_data.update({
                    "filled_at": T,
                    "created_at": T,
                    "breakeven_adjustment": charges_entry / qty if qty else 0.0,
                })
                order_to_save = order_data

                # Open Position
                position_data = {
                    "position_id": position_id,
                    "run_id": run_id,
                    "symbol": symbol,
                    "instrument_type": config.INSTRUMENT_TYPE,
                    "product_type": config.PRODUCT_TYPE,
                    "active": True,
                    "direction": "BUY",
                    "entry": req_entry,
                    "executed_entry": exec_entry,
                    "stop": req_stop,
                    "target": req_target,
                    "quantity": qty,
                    "entry_time": T,
                    "last_price": current_price,
                    "unrealized_pnl": qty * (current_price - exec_entry),
                    "status": "OPEN",
                    "entry_order_id": order_data["order_id"],
                    "slippage_entry": entry_result["slippage_entry_points"],
                    "charges_entry": charges_entry,
                    "charges_total": charges_entry
                }
                position_to_save = position_data
                
                cash_diff = -(deployed_capital + charges_entry)
                deployed_diff = deployed_capital
                charges_diff = charges_entry
                trades_taken_today += 1
                portfolio["unrealized_pnl"] = position_data["unrealized_pnl"]

            # EXIT / Forced exit / Stop or target hit
            elif active_pos and (action == "EXIT" or validation_result.get("forced_exit") or validation_result.get("stop_hit") or validation_result.get("target_hit")):
                qty = active_pos["quantity"]
                direction = active_pos["direction"]
                exec_entry = active_pos["executed_entry"]
                
                exit_price_req = current_price
                exit_reason = "EXIT"
                if validation_result.get("stop_hit"):
                    exit_price_req = active_pos["stop"]
                    exit_reason = "STOP_LOSS"
                elif validation_result.get("target_hit"):
                    exit_price_req = active_pos["target"]
                    exit_reason = "TARGET"
                elif validation_result.get("forced_exit"):
                    exit_reason = "FORCED_SQUAREOFF"

                exit_result = OrderSimulator(
                    slippage_config=SlippageConfig()
                ).simulate_exit_order(
                    run_id=run_id,
                    position_id=position_id,
                    symbol=symbol,
                    direction=direction,
                    exit_price=float(exit_price_req),
                    quantity=qty,
                    order_type=exit_reason,
                    product_type=config.PRODUCT_TYPE,
                )
                exec_exit = float(exit_result["executed_exit"])
                charges_exit = float(exit_result["charges_exit"].get("total", 0.0))
                charges_entry = float(active_pos.get("charges_entry") or 0.0)
                
                # P&L calculations
                gross_pnl = qty * (exec_exit - exec_entry) if direction == "BUY" else qty * (exec_entry - exec_exit)
                net_pnl = gross_pnl - charges_entry - charges_exit

                order_data = asdict(exit_result["exit_order"])
                order_data.update({
                    "decision_id": decision_id,
                    "filled_at": T,
                    "created_at": T,
                    "breakeven_adjustment": charges_exit / qty if qty else 0.0,
                })
                order_to_save = order_data
                risk_amount = qty * abs(exec_entry - active_pos["stop"])

                # Close position
                position_to_update = (position_id, {
                    "active": False,
                    "exit_time": T,
                    "exit_price": exec_exit,
                    "exit_reason": exit_reason,
                    "last_price": current_price,
                    "unrealized_pnl": 0.0,
                    "realized_pnl": net_pnl,
                    "r_multiple_realized": net_pnl / risk_amount if risk_amount else 0.0,
                    "status": "CLOSED" if exit_reason != "FORCED_SQUAREOFF" else "SQUARED_OFF",
                    "exit_order_id": order_data["order_id"],
                    "slippage_exit": exit_result["slippage_exit_points"],
                    "charges_exit": charges_exit,
                    "charges_total": charges_entry + charges_exit
                })

                cash_diff = (qty * exec_exit) - charges_exit
                deployed_diff = -(qty * exec_entry)
                pnl_realized_diff = net_pnl
                charges_diff = charges_exit
                portfolio["unrealized_pnl"] = 0.0

                # Cooldown Locks
                lock_duration_candles = 2
                if exit_reason == "TARGET":
                    lock_duration_candles = 1
                
                expires_at = T + pd.Timedelta(minutes=lock_duration_candles * 15)
                lock_data = {
                    "lock_id": f"lock_{uuid.uuid4().hex[:8]}",
                    "run_id": run_id,
                    "symbol": symbol,
                    "direction": direction,
                    "reason": f"AFTER_{exit_reason}",
                    "expires_at": expires_at,
                    "created_at": T
                }
                lock_to_save = lock_data

            # Hold / Skip / Rejections: no position capital changes
            elif active_pos and action == "HOLD":
                # Re-evaluate live unrealized P&L
                qty = active_pos["quantity"]
                direction = active_pos["direction"]
                exec_entry = active_pos["executed_entry"]
                risk_amount = qty * abs(exec_entry - active_pos["stop"])
                unrealized = qty * (current_price - exec_entry) if direction == "BUY" else qty * (exec_entry - current_price)
                position_to_update = (position_id, {
                    "last_price": current_price,
                    "unrealized_pnl": unrealized,
                    "r_multiple_live": unrealized / risk_amount if risk_amount else 0.0
                })
                # Set portfolio snapshot fields to match
                portfolio["unrealized_pnl"] = unrealized

            # 4. Save Portfolio Snapshot (After)
            after_snap_id = f"snap_after_{decision_id}"
            portfolio_after = {
                "snapshot_id": after_snap_id,
                "run_id": run_id,
                "decision_id": decision_id,
                "timestamp": T,
                "starting_capital": portfolio["starting_capital"],
                "cash_available": cash_available + cash_diff,
                "capital_deployed": capital_deployed + deployed_diff,
                "capital_reserved": portfolio["capital_reserved"],
                "realized_pnl": realized_pnl + pnl_realized_diff,
                "unrealized_pnl": portfolio.get("unrealized_pnl", 0.0),
                "charges_paid": charges_paid + charges_diff,
                "max_capital_per_trade": portfolio["max_capital_per_trade"],
                "max_daily_loss": portfolio["max_daily_loss"],
                "daily_loss_used": portfolio["daily_loss_used"] + (0.0 if pnl_realized_diff >= 0 else abs(pnl_realized_diff)),
                "trades_taken_today": trades_taken_today,
                "max_trades_per_day": portfolio["max_trades_per_day"],
            }

            # 5. Save Decision
            checklist = signal.get("checklist", {})
            dart = signal.get("dart", {})
            
            decision_data = {
                "decision_id": decision_id,
                "run_id": run_id,
                "symbol": symbol,
                "decision_time": T,
                "current_price": current_price,
                "portfolio_snapshot_before": portfolio["snapshot_id"],
                "portfolio_snapshot_after": after_snap_id,
                "raw_action": signal.get("action"),
                "validated_action": action,
                "confidence": signal.get("confidence", 0.0),
                "entry": signal.get("entry"),
                "stop": signal.get("stop"),
                "target": signal.get("target"),
                "net_reward_risk": signal.get("net_reward_risk"),
                "expected_horizon_minutes": signal.get("expected_horizon_minutes"),
                "dart_direction": dart.get("direction"),
                "dart_area": dart.get("area"),
                "dart_risk": dart.get("risk"),
                "dart_trigger": dart.get("trigger"),
                "checklist_json": checklist,
                "reason": signal.get("reason", ""),
                "invalidation": signal.get("invalidation"),
                "thesis_health": signal.get("thesis_health", "not_applicable"),
                "exit_reason": signal.get("exit_reason"),
                "suggested_exit_price": signal.get("suggested_exit_price"),
                "position_id": position_id,
                "is_valid": is_valid,
                "rejection_reason": rejection_reason,
                "tool_calls_json": signal.get("tool_calls", []),
                "memory_references": signal.get("memory_references", []),
                "reflection_ids": signal.get("reflection_ids", []),
                "agent_version": "dart-pa-v2"
            }

            # TOPOLOGICAL INSERTIONS TO AVOID FOREIGN KEY ISSUES
            if position_to_save:
                uow.positions.save_position(position_to_save)
            if position_to_update:
                uow.positions.update_position(position_to_update[0], position_to_update[1])
            if lock_to_save:
                uow.locks.save_lock(lock_to_save)

            uow.portfolio.save_portfolio_snapshot(portfolio_after)
            uow.decisions.save_decision(decision_data)

            if order_to_save:
                uow.orders.save_order(order_to_save)

        return decision_data, decision_id


class OutcomeFeedbackService:
    @staticmethod
    def _upsert_calibration_bucket(
        uow: UnitOfWork,
        run_id: str,
        bucket_key: str,
        bucket_type: str,
        net_r: float,
        is_win: bool,
    ) -> None:
        uow.cursor.execute(
            "SELECT * FROM calibration_stats WHERE run_id = %s AND bucket_key = %s",
            (run_id, bucket_key)
        )
        cal_row = uow.cursor.fetchone()
        if cal_row:
            cal_cols = [desc[0] for desc in uow.cursor.description]
            cal = dict(zip(cal_cols, cal_row))
            new_total = cal["total_trades"] + 1
            new_wins = cal["wins"] + (1 if is_win else 0)
            new_losses = cal["losses"] + (0 if is_win else 1)
            new_sum = cal["sum_net_r"] + net_r
            uow.calibration.update_calibration_stats(cal["stat_id"], {
                "total_trades": new_total,
                "wins": new_wins,
                "losses": new_losses,
                "win_rate": new_wins / new_total,
                "sum_net_r": new_sum,
                "avg_net_r": new_sum / new_total
            })
        else:
            uow.calibration.save_calibration_stats({
                "stat_id": f"cal_{uuid.uuid4().hex[:12]}",
                "run_id": run_id,
                "bucket_key": bucket_key,
                "bucket_type": bucket_type,
                "total_trades": 1,
                "wins": 1 if is_win else 0,
                "losses": 0 if is_win else 1,
                "win_rate": 1.0 if is_win else 0.0,
                "avg_net_r": net_r,
                "sum_net_r": net_r
            })

    @staticmethod
    def record_feedback(
        decision_id: str,
        outcome_label: str,
        net_r: float,
        mfe_pct: float,
        mae_pct: float,
        setup_tags: List[str] = None
    ) -> None:
        """Record evaluation outcome into decisions, memory episodes, reflections, and calibration."""
        with UnitOfWork() as uow:
            # 1. Update decision outcomes
            outcome_json = {
                "outcome_label": outcome_label,
                "net_r": net_r,
                "mfe_pct": mfe_pct,
                "mae_pct": mae_pct,
            }
            uow.decisions.update_decision_outcome(decision_id, {
                "outcome_json": outcome_json,
                "outcome_label": outcome_label,
                "outcome_net_r": net_r,
                "mfe_pct": mfe_pct,
                "mae_pct": mae_pct,
            })

            # Fetch the decision details
            uow.cursor.execute("SELECT * FROM decisions WHERE decision_id = %s", (decision_id,))
            row = uow.cursor.fetchone()
            if not row:
                return
            columns = [desc[0] for desc in uow.cursor.description]
            dec = dict(zip(columns, row))
            checklist = dec.get("checklist_json") or {}
            if isinstance(checklist, str):
                checklist = json.loads(checklist)
            setup_tags = setup_tags or []

            # 2. Save Memory Episode
            episode_id = f"ep_{uuid.uuid4().hex[:12]}"
            episode_data = {
                "episode_id": episode_id,
                "run_id": dec["run_id"],
                "symbol": dec["symbol"],
                "decision_id": decision_id,
                "action": dec["validated_action"],
                "direction": dec["dart_direction"],
                "market_regime": checklist.get("market_regime", "unclear"),
                "session_type": checklist.get("session_type", "unclear"),
                "gap_type": checklist.get("gap_type", "unknown"),
                "structure_state": checklist.get("structure_state", "unclear"),
                "vwap_relation": checklist.get("vwap_relation", "unknown"),
                "vwap_distance_atr": checklist.get("vwap_distance_atr"),
                "profile_location": checklist.get("profile_location", "unknown"),
                "price_location": checklist.get("price_location", "unknown"),
                "time_bucket": checklist.get("time_bucket", "unknown"),
                "volatility_bucket": checklist.get("volatility_bucket", "unknown"),
                "setup_tags": setup_tags,
                "outcome_net_r": net_r,
                "outcome_label": outcome_label,
                "mfe_pct": mfe_pct,
                "mae_pct": mae_pct,
                "confidence": dec["confidence"],
                "thesis_json": {
                    "dart_direction": dec.get("dart_direction"),
                    "dart_area": dec.get("dart_area"),
                    "dart_risk": dec.get("dart_risk"),
                    "dart_trigger": dec.get("dart_trigger"),
                    "reason": dec.get("reason"),
                    "invalidation": dec.get("invalidation"),
                },
            }
            uow.memory.save_episode(episode_data)

            # 3. Save Reflection (if confidence gate is passed)
            reflection = ReflectionWriter().write_reflection(
                signal={
                    "symbol": dec["symbol"],
                    "action": dec["validated_action"],
                    "confidence": dec["confidence"],
                    "entry": dec.get("entry"),
                    "stop": dec.get("stop"),
                    "target": dec.get("target"),
                    "invalidation": dec.get("invalidation"),
                    "dart": {
                        "direction": dec.get("dart_direction"),
                        "area": dec.get("dart_area"),
                        "risk": dec.get("dart_risk"),
                        "trigger": dec.get("dart_trigger"),
                    },
                },
                market_state={
                    "market_regime": episode_data["market_regime"],
                    "session_type": episode_data["session_type"],
                    "structure_state": episode_data["structure_state"],
                },
                outcome={
                    "outcome": outcome_label,
                    "net_r_multiple": net_r,
                    "mfe_pct": mfe_pct,
                    "mae_pct": mae_pct,
                },
                setup_tags=setup_tags,
            )
            if reflection and reflection.reflection_level in ("HIGH", "MEDIUM"):
                uow.memory.save_reflection({
                    "reflection_id": reflection.reflection_id,
                    "run_id": dec["run_id"],
                    "symbol": dec["symbol"],
                    "lesson": reflection.lesson,
                    "tags": reflection.tags,
                    "source_episode_ids": [episode_id],
                    "direction": reflection.direction,
                    "reflection_level": reflection.reflection_level,
                    "confidence": reflection.confidence,
                    "num_supporting_episodes": 1
                })

            # 4. Update Calibration Stats
            bucket_key = f"confidence_{dec['confidence']:.2f}"
            is_win = outcome_label == "win" or net_r > 0
            OutcomeFeedbackService._upsert_calibration_bucket(
                uow, dec["run_id"], bucket_key, "confidence", net_r, is_win
            )
            for tag in setup_tags:
                OutcomeFeedbackService._upsert_calibration_bucket(
                    uow, dec["run_id"], f"setup_{tag}", "setup_tag", net_r, is_win
                )


class SessionStateService:
    @staticmethod
    def init_session_if_needed(
        run_id: str,
        symbol: str,
        T: datetime,
        df_daily: pd.DataFrame,
        df_15m: pd.DataFrame
    ) -> str:
        session_id = f"sess_{run_id}_{T.strftime('%Y%m%d')}"
        with UnitOfWork() as uow:
            session_map = uow.sessions.get_session_map(session_id)
            if session_map:
                return session_id

            # Create new session map
            # Find prior day close, high, low from df_daily
            prior_days = df_daily[df_daily.index.date < T.date()]
            if not prior_days.empty:
                prior_day = prior_days.iloc[-1]
                prior_close = float(prior_day["close"])
                prior_high = float(prior_day["high"])
                prior_low = float(prior_day["low"])
            else:
                prior_close = 0.0
                prior_high = 0.0
                prior_low = 0.0

            # Today's open from df_15m
            today_candles = df_15m[df_15m.index.date == T.date()]
            if not today_candles.empty:
                today_open = float(today_candles.iloc[0]["open"])
            else:
                today_open = prior_close

            # Calculate ATR from df_daily
            atr_series = compute_atr(df_daily)
            atr = float(atr_series.loc[prior_days.index[-1]]) if not prior_days.empty and prior_days.index[-1] in atr_series.index else prior_close * 0.005
            
            # Classify gap
            gap = classify_gap(
                prior_close=prior_close,
                today_open=today_open,
                prior_high=prior_high,
                prior_low=prior_low,
                atr=atr
            )

            # Insert session map
            uow.sessions.save_session_map({
                "session_id": session_id,
                "run_id": run_id,
                "symbol": symbol,
                "session_date": T.date(),
                "opening_range_high": None,
                "opening_range_low": None,
                "session_high": today_open,
                "session_low": today_open,
                "session_vwap": today_open,
                "vwap_slope": 0.0,
                "current_poc": today_open,
                "current_vah": today_open,
                "current_val": today_open,
                "gap_classification": gap.gap_type,
                "gap_points": gap.gap_points,
                "gap_pct": gap.gap_pct,
                "market_regime": "normal",
                "current_bias": "neutral",
                "session_type": "normal",
                "cooldown_active": False
            })
            
            # Identify initial levels: prior high/low, prior close
            # Save level events and levels
            if prior_high > 0:
                level_id = f"lvl_{session_id}_prior_high"
                uow.sessions.save_level({
                    "level_id": level_id,
                    "session_id": session_id,
                    "run_id": run_id,
                    "price": prior_high,
                    "level_type": "swing_high",
                    "state": "ACTIVE",
                    "strength": 1
                })
                uow.sessions.save_event({
                    "event_id": f"evt_{level_id}_id",
                    "session_id": session_id,
                    "run_id": run_id,
                    "event_time": T,
                    "event_type": "LEVEL_IDENTIFIED",
                    "event_data": {
                        "level_id": level_id,
                        "level_price": prior_high,
                        "level_type": "swing_high"
                    }
                })
            if prior_low > 0:
                level_id = f"lvl_{session_id}_prior_low"
                uow.sessions.save_level({
                    "level_id": level_id,
                    "session_id": session_id,
                    "run_id": run_id,
                    "price": prior_low,
                    "level_type": "swing_low",
                    "state": "ACTIVE",
                    "strength": 1
                })
                uow.sessions.save_event({
                    "event_id": f"evt_{level_id}_id",
                    "session_id": session_id,
                    "run_id": run_id,
                    "event_time": T,
                    "event_type": "LEVEL_IDENTIFIED",
                    "event_data": {
                        "level_id": level_id,
                        "level_price": prior_low,
                        "level_type": "swing_low"
                    }
                })
            return session_id

    @staticmethod
    def process_candle_update(
        run_id: str,
        symbol: str,
        T: datetime,
        df_15m_up_to_T: pd.DataFrame,
        df_daily: pd.DataFrame
    ) -> None:
        session_id = f"sess_{run_id}_{T.strftime('%Y%m%d')}"
        
        # 1. Compute today's session high, low, opening range, vwap, volume profile
        today_data = df_15m_up_to_T[df_15m_up_to_T.index.date == T.date()]
        if today_data.empty:
            return
            
        current_candle = today_data.iloc[-1]
        current_price = float(current_candle["close"])
        
        session_high = float(today_data["high"].max())
        session_low = float(today_data["low"].min())
        
        # Opening range: 09:15 to 09:30 IST (first 15 minutes, which is usually the first candle)
        # Or let's say the first candle's high/low
        or_candles = today_data[today_data.index.time <= time(9, 30)]
        opening_range_high = float(or_candles["high"].max()) if not or_candles.empty else None
        opening_range_low = float(or_candles["low"].min()) if not or_candles.empty else None
        
        # Calculate daily ATR for levels thresholding
        prior_days = df_daily[df_daily.index.date < T.date()]
        atr_series = compute_atr(df_daily)
        atr = float(atr_series.loc[prior_days.index[-1]]) if not prior_days.empty and prior_days.index[-1] in atr_series.index else current_price * 0.005

        # Compute VWAP
        vwap_res = compute_session_vwap(today_data, current_price, atr=atr)
        
        # Compute volume profile
        vp_res = compute_volume_profile(today_data, current_price)
        
        with UnitOfWork() as uow:
            # Update session map in DB
            uow.sessions.update_session_map(session_id, {
                "opening_range_high": opening_range_high,
                "opening_range_low": opening_range_low,
                "session_high": session_high,
                "session_low": session_low,
                "session_vwap": vwap_res.current_vwap,
                "vwap_slope": vwap_res.vwap_slope,
                "current_poc": vp_res.poc,
                "current_vah": vp_res.vah,
                "current_val": vp_res.val,
                "updated_at": datetime.now(timezone.utc)
            })
            
            # Load active levels for this session map from DB
            db_levels = uow.sessions.get_all_levels(session_id)
            levels = []
            for dbl in db_levels:
                levels.append(SessionLevel(
                    level_id=dbl["level_id"],
                    price=dbl["price"],
                    level_type=dbl["level_type"],
                    state=LevelState(dbl["state"]),
                    strength=dbl["strength"]
                ))
            
            # Run lifecycle manager
            mgr = LevelLifecycleManager(atr=atr)
            candle_dict = {
                "open": float(current_candle["open"]),
                "high": float(current_candle["high"]),
                "low": float(current_candle["low"]),
                "close": float(current_candle["close"]),
                "volume": int(current_candle["volume"])
            }
            events = mgr.process_candle(candle_dict, T, levels)
            
            # Save events and update levels in DB
            for event in events:
                uow.sessions.save_event({
                    "event_id": event.event_id,
                    "session_id": session_id,
                    "run_id": run_id,
                    "event_time": event.event_time,
                    "event_type": event.event_type,
                    "event_data": event.event_data
                })
                
            for level in levels:
                uow.sessions.update_level(level.level_id, {
                    "state": level.state.value,
                    "level_type": level.level_type,
                    "updated_at": datetime.now(timezone.utc)
                })
                
            # Detect swings dynamically
            swings = detect_swings(today_data, lookback=3)
            for sh in swings["swing_highs"]:
                sh_time = pd.Timestamp(sh["time"])
                level_id = f"lvl_{session_id}_sh_{sh_time.strftime('%H%M%S')}"
                if not any(dbl["level_id"] == level_id for dbl in db_levels):
                    uow.sessions.save_level({
                        "level_id": level_id,
                        "session_id": session_id,
                        "run_id": run_id,
                        "price": sh["price"],
                        "level_type": "swing_high",
                        "state": "ACTIVE",
                        "strength": 1
                    })
                    uow.sessions.save_event({
                        "event_id": f"evt_{level_id}_id",
                        "session_id": session_id,
                        "run_id": run_id,
                        "event_time": T,
                        "event_type": "LEVEL_IDENTIFIED",
                        "event_data": {
                            "level_id": level_id,
                            "level_price": sh["price"],
                            "level_type": "swing_high"
                        }
                    })
            for sl in swings["swing_lows"]:
                sl_time = pd.Timestamp(sl["time"])
                level_id = f"lvl_{session_id}_sl_{sl_time.strftime('%H%M%S')}"
                if not any(dbl["level_id"] == level_id for dbl in db_levels):
                    uow.sessions.save_level({
                        "level_id": level_id,
                        "session_id": session_id,
                        "run_id": run_id,
                        "price": sl["price"],
                        "level_type": "swing_low",
                        "state": "ACTIVE",
                        "strength": 1
                    })
                    uow.sessions.save_event({
                        "event_id": f"evt_{level_id}_id",
                        "session_id": session_id,
                        "run_id": run_id,
                        "event_time": T,
                        "event_type": "LEVEL_IDENTIFIED",
                        "event_data": {
                            "level_id": level_id,
                            "level_price": sl["price"],
                            "level_type": "swing_low"
                        }
                    })
                
            # If session is close to end, expire all levels
            controller = MarketSessionController()
            if T.time() >= controller.config.session_end:
                expire_events = mgr.expire_all_levels(levels, T)
                for event in expire_events:
                    uow.sessions.save_event({
                        "event_id": event.event_id,
                        "session_id": session_id,
                        "run_id": run_id,
                        "event_time": event.event_time,
                        "event_type": event.event_type,
                        "event_data": event.event_data
                    })
                for level in levels:
                    uow.sessions.update_level(level.level_id, {
                        "state": level.state.value,
                        "updated_at": datetime.now(timezone.utc)
                    })
            
