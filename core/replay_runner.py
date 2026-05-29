"""
Replay runner: orchestrates the walk-forward replay loop.

Extracted from main.py monolith into a single-responsibility module.
Handles session bootstrap, context delivery, agent orchestration,
validation, persistence, auditing, and evaluation.
"""
import json
import hashlib
import time
from datetime import datetime
from typing import Dict, Any, List

import pandas as pd

from config import config
from data.collector import resample_to_timeframe
from core.clock import WalkForwardClock
from core.context import build_market_state_package, format_market_state_for_prompt
from core.tools import ToolHarness
from core.charts import generate_all_charts
from core.session_controller import MarketSessionController
from agent.dart import DartAgent
from validation.validator import validate_signal
from journal.signal import SignalJournal
from journal.evaluator import FeedbackEvaluator
from observability.langfuse_integration import create_tracer
from db.services import (
    RunBootstrapService, ReplayStateService,
    DecisionTransactionService, OutcomeFeedbackService,
    SessionStateService, AuditService,
)
from core.interfaces import ContextDeliveryMode, TradeEventType


class ReplayRunner:
    """
    Orchestrates the walk-forward replay loop.

    Responsibilities:
    - Session bootstrap and clock iteration
    - Context delivery (bootstrap vs incremental)
    - Agent orchestration with structured auditing
    - Validation and risk policy enforcement
    - Persistence and audit logging
    - Evaluation and reporting
    - Failure mode enforcement (DB, LLM, tool, persistence, evaluation)
    """

    def __init__(
        self,
        data_5m: pd.DataFrame,
        data_daily: pd.DataFrame,
        data_weekly: pd.DataFrame,
        redis_avail: bool = False,
        redis_client=None,
        replay_date=None,
    ):
        self.df_5m = data_5m
        self.df_daily = data_daily
        self.df_weekly = data_weekly
        self.redis_avail = redis_avail
        self.redis_client = redis_client
        self.replay_date = replay_date

        self.clock = None
        self.agent = DartAgent()
        self.journal = SignalJournal()
        self.evaluator = FeedbackEvaluator(self.df_5m)
        self.tracer = create_tracer()

        self.run_id = None
        self.decision_ids: Dict[str, str] = {}
        self._context_delivery_mode = ContextDeliveryMode.RESET
        self._session_first_candle_seen: Dict[str, bool] = {}

    def _check_db_persistence(self, uow) -> None:
        """Check that the DB transaction is healthy before committing."""
        try:
            uow.cursor.execute("SELECT 1")
        except Exception as e:
            raise RuntimeError(f"Persistence failure: DB connection lost - {e}")

    def _mark_run_incomplete(self, reason: str) -> None:
        """Mark the run as incomplete due to evaluation or persistence failure."""
        from db.unit_of_work import UnitOfWork
        try:
            with UnitOfWork() as uow:
                uow.runs.update_experiment_run(self.run_id, {
                    "status": "incomplete",
                    "notes": f"Incomplete: {reason}",
                })
        except Exception:
            pass

    def run(
        self,
        max_steps: int = None,
    ) -> Dict[str, Any]:
        """Execute the full replay loop. Returns result dict with journal, evaluator, summary, metrics."""
        df_5m = resample_to_timeframe(self.df_5m, config.DECISION_INTERVAL)
        df_daily = self.df_daily
        df_weekly = self.df_weekly
        self.clock = WalkForwardClock(df_5m)

        # Agent workflow check
        agent_workflow = config.AGENT_WORKFLOW.lower()
        if agent_workflow == "graph":
            print(f"\n  [WARNING] AGENT_WORKFLOW=graph is not wired into replay loop.")
            print(f"             Falling back to 'direct' workflow (DartAgent).")

        total_steps = self.clock.total_steps()
        if self.replay_date:
            total_steps = sum(
                1 for point in self.clock.iterate()
                if point["decision_time"].date() == self.replay_date
            )

        print(f"\n{'─' * 60}")
        print(f"  WALK-FORWARD REPLAY")
        print(f"{'─' * 60}")
        print(f"  Total eligible decision points: {total_steps}")
        if self.replay_date:
            print(f"  Replay date: {self.replay_date}")
        print(f"  {config.DECISION_INTERVAL} data range: {df_5m.index[0]} to {df_5m.index[-1]}")

        # ---- Initialize Run & Snapshot Set ----
        self.run_id = f"run_{config.SYMBOL}_{int(datetime.now().timestamp())}"
        try:
            RunBootstrapService.create_or_resume_run(
                run_id=self.run_id,
                symbol=config.SYMBOL,
                starting_capital=config.STARTING_CAPITAL,
                max_capital_per_trade=config.MAX_CAPITAL_PER_TRADE,
                risk_budget_pct=config.RISK_BUDGET_PCT,
                max_daily_loss=config.MAX_DAILY_LOSS,
                max_trades_per_day=config.MAX_TRADES_PER_DAY,
                notes="Replay walk-forward run",
                start_date=df_5m.index.min().date(),
                end_date=df_5m.index.max().date(),
            )
            RunBootstrapService.create_snapshot_set(
                symbol=config.SYMBOL,
                df_weekly=df_weekly,
                df_daily=df_daily,
                df_15m=df_5m,
                run_id=self.run_id,
            )
        except Exception as e:
            print(f"\n  FATAL: Could not initialize run in Postgres: {e}")
            AuditService.record_audit_event(
                run_id=None,
                event_type="BOOT_FAILURE",
                message=f"Postgres unavailable at run init: {e}",
                severity="fatal",
                symbol=config.SYMBOL,
            )
            raise

        # ---- REPLAY LOOP ----
        step_count = 0
        start_time = time.time()

        print(f"\n{'─' * 60}")
        print("  STARTING REPLAY")
        print(f"{'─' * 60}\n")

        for decision_point in self.clock.iterate():
            T = decision_point["decision_time"]
            session_start = decision_point["session_start"]
            session_end = decision_point["session_end"]

            if self.replay_date and T.date() != self.replay_date:
                continue

            # ---- Session bootstrap and candle update ----
            try:
                SessionStateService.init_session_if_needed(self.run_id, config.SYMBOL, T, df_daily, df_5m)
                SessionStateService.process_candle_update(self.run_id, config.SYMBOL, T, decision_point["data_up_to_T"], df_daily)
            except Exception as e:
                AuditService.record_audit_event(
                    run_id=self.run_id,
                    event_type="SESSION_INIT_FAILURE",
                    message=f"Session state error at {T}: {e}",
                    severity="error",
                    symbol=config.SYMBOL,
                )
                print(f"\n  [Session error: {e}] - skipping")
                continue

            # ---- Context delivery mode tracking ----
            session_key = str(T.date())
            is_first_candle = not self._session_first_candle_seen.get(session_key, False)
            self._session_first_candle_seen[session_key] = True

            # First candle of the session is context-only (not trade-eligible)
            first_candle_context_only = is_first_candle

            # Determine context delivery mode
            if is_first_candle:
                self._context_delivery_mode = ContextDeliveryMode.BOOTSTRAP
            elif self._context_delivery_mode == ContextDeliveryMode.BOOTSTRAP:
                self._context_delivery_mode = ContextDeliveryMode.INCREMENTAL

            if not ReplayStateService.should_evaluate(self.run_id, config.SYMBOL, T):
                continue

            # ---- Stop/target check for active positions ----
            active_pos = ReplayStateService.get_active_position(self.run_id, config.SYMBOL)
            if active_pos:
                try:
                    self._handle_position_exit(T, decision_point, active_pos)
                except Exception as e:
                    AuditService.record_audit_event(
                        run_id=self.run_id,
                        event_type="POSITION_EXIT_FAILURE",
                        message=f"Position exit error at {T}: {e}",
                        severity="error",
                        symbol=config.SYMBOL,
                    )
                active_pos = ReplayStateService.get_active_position(self.run_id, config.SYMBOL)

            if max_steps and step_count >= max_steps:
                print(f"\n  Reached max steps limit ({max_steps}). Stopping.")
                break
            step_count += 1

            has_pos = active_pos is not None

            if first_candle_context_only:
                print(f"\n[{step_count}/{max_steps or total_steps}] {T} "
                      f"| First candle - context only, no trade eligible", end="", flush=True)

            print(f"\n[{step_count}/{max_steps or total_steps}] {T} "
                  f"| Price: {decision_point['candle_T']['close']:.2f}", end="", flush=True)

            # ---- Langfuse root span ----
            trace_name = f"decision_{T.strftime('%Y%m%d_%H%M')}"
            root_span = self.tracer.create_root_span(
                name=trace_name,
                input_data={
                    "step": step_count,
                    "decision_time": str(T),
                    "instrument": config.SYMBOL,
                    "current_price": float(decision_point['candle_T']['close']),
                    "context_delivery_mode": self._context_delivery_mode.value,
                },
                metadata={
                    "step": step_count,
                    "decision_time": str(T),
                    "instrument": config.SYMBOL,
                    "context_delivery_mode": self._context_delivery_mode.value,
                },
            )

            # ---- Build Market State Package ----
            try:
                package = build_market_state_package(
                    T=T,
                    data_5m=df_5m,
                    data_daily=df_daily,
                    data_weekly=df_weekly,
                    chart_paths={},
                )
                package["session_start"] = str(session_start)
                package["session_end"] = str(session_end)
            except Exception as e:
                AuditService.record_audit_event(
                    run_id=self.run_id,
                    event_type="MARKET_STATE_FAILURE",
                    message=f"Market state build error at {T}: {e}",
                    severity="error",
                    symbol=config.SYMBOL,
                )
                print(f"\n  [Error building market state: {e}] - skipping")
                continue

            # ---- Charts ----
            initial_charts = {}
            try:
                initial_charts = generate_all_charts(df_5m, df_daily, df_weekly, T, None, None, None)
                package["chart_paths"] = initial_charts
            except Exception as e:
                print(f" [Chart warning: {e}]", end="")

            # ---- Format market state ----
            market_state_text = format_market_state_for_prompt(package)

            # ---- Tool Harness ----
            harness = ToolHarness(df_5m, df_daily, df_weekly, T, run_id=self.run_id)

            # ---- Run DART Agent with structured auditing ----
            agent_result = self._run_agent_with_audit(package, market_state_text, harness, has_pos)

            signal = agent_result.get("final_signal", {})

            # Enforce first candle context-only: override to SKIP/HOLD
            if first_candle_context_only and signal.get("action") in ("BUY", "SELL"):
                original_action = signal.get("action")
                signal["action"] = "SKIP"
                signal["reason"] = f"First open-session candle is context-only. {signal.get('reason', '')}"
                agent_result["final_signal"] = signal
                agent_result["raw_responses"] = agent_result.get("raw_responses", [])
                AuditService.record_audit_event(
                    run_id=self.run_id,
                    event_type="FIRST_CANDLE_OVERRIDE",
                    message=f"Trade prevented on first open-session candle at {T}",
                    severity="info",
                    symbol=config.SYMBOL,
                    details={"original_action": original_action, "decision_time": str(T)},
                )

            if has_pos and signal.get("action") in ("HOLD", "EXIT") and not signal.get("position_id"):
                current_pos = active_pos or {}
                signal["position_id"] = current_pos.get("position_id") or "open_position"

            # ---- Update charts with signal levels ----
            entry = signal.get("entry")
            stop = signal.get("stop")
            target = signal.get("target")
            final_charts = initial_charts
            raw_action = signal.get("action", "HOLD" if has_pos else "SKIP")
            if raw_action in ("BUY", "SELL") and entry and stop and target:
                try:
                    final_charts = generate_all_charts(df_5m, df_daily, df_weekly, T, entry, stop, target)
                except Exception:
                    pass

            # ---- Validate Signal ----
            validation_result = validate_signal(
                signal, T, session_end,
                has_open_position=has_pos,
                tool_calls=agent_result.get("tool_calls", [])
            )
            signal["tool_calls"] = agent_result.get("tool_calls", [])
            signal["_raw_llm_responses"] = agent_result.get("raw_responses", [])
            signal["_context_data_hash"] = self._hash_context_package(package)

            # ---- Process Decision in Postgres (with rollback on failure) ----
            try:
                dec_data, dec_id = DecisionTransactionService.process_decision(
                    run_id=self.run_id,
                    symbol=config.SYMBOL,
                    T=T,
                    current_price=float(decision_point['candle_T']['close']),
                    signal=signal,
                    validation_result=validation_result,
                )
                self.decision_ids[str(T)] = dec_id
                self._persist_agent_audit(dec_id, agent_result)

                # Record trade events for entry/exit lifecycle
                if validation_result.get("is_valid"):
                    vaction = validation_result.get("action")
                    if vaction == "BUY":
                        AuditService.record_trade_event(
                            run_id=self.run_id, symbol=config.SYMBOL,
                            event_type=TradeEventType.ENTRY_REQUESTED.value,
                            decision_id=dec_id, direction=vaction,
                            price=signal.get("entry"), quantity=signal.get("quantity"),
                            reason=signal.get("reason", ""),
                        )
                    elif vaction == "EXIT":
                        AuditService.record_trade_event(
                            run_id=self.run_id, symbol=config.SYMBOL,
                            event_type=TradeEventType.EXIT_REQUESTED.value,
                            decision_id=dec_id, direction="EXIT",
                            price=signal.get("suggested_exit_price"),
                            reason=signal.get("exit_reason", "manual_exit"),
                        )
                elif validation_result.get("rejection_reason"):
                    AuditService.record_trade_event(
                        run_id=self.run_id, symbol=config.SYMBOL,
                        event_type=TradeEventType.REJECTED.value,
                        decision_id=dec_id,
                        reason=validation_result["rejection_reason"],
                        details={"signal_action": signal.get("action")},
                    )

            except Exception as e:
                AuditService.record_audit_event(
                    run_id=self.run_id,
                    event_type="PERSISTENCE_FAILURE",
                    message=f"Decision persistence failed at {T}: {e}",
                    severity="fatal",
                    symbol=config.SYMBOL,
                    details={"step": step_count},
                )
                print(f"\n  [FATAL] Persistence error: {e}. Halting run.")
                self._mark_run_incomplete(f"Persistence failure at step {step_count}: {e}")
                raise

            is_valid = validation_result.get("is_valid", False)
            final_action = validation_result.get("action", signal.get("action", "HOLD"))
            sizing = validation_result.get("sizing", {})

            if is_valid and final_action in ("BUY", "SELL") and not has_pos:
                print(f" [OPEN {final_action} qty={sizing['quantity']}]", end="", flush=True)

            if validation_result.get("rejection_reason"):
                print(f" [{validation_result['rejection_reason']}]", end="", flush=True)
            else:
                print(f" [{final_action}]", end="", flush=True)

            # ---- Journal ----
            record = self.journal.record(
                decision_time=T,
                market_state_package=package,
                agent_result=agent_result,
                validation_result=validation_result,
                chart_paths=final_charts,
            )

            # ---- Langfuse tracing ----
            self._trace_to_langfuse(root_span, agent_result, signal, market_state_text, validation_result)

            # ---- Redis caching ----
            if self.redis_avail and self.redis_client:
                try:
                    cache_key = f"decision:{config.SYMBOL}:{T.strftime('%Y%m%d_%H%M')}"
                    self.redis_client.setex(cache_key, 3600, json.dumps({
                        "action": final_action,
                        "price": package.get("current_price"),
                        "confidence": signal.get("confidence"),
                        "has_position": has_pos,
                    }, default=str))
                except Exception:
                    pass

        # ---- REPLAY COMPLETE ----
        elapsed = time.time() - start_time
        print(f"\n\n{'─' * 60}")
        print(f"  REPLAY COMPLETE")
        print(f"{'─' * 60}")
        print(f"  Steps executed: {step_count}")
        print(f"  Elapsed: {elapsed:.1f}s ({elapsed/step_count:.1f}s per step)" if step_count > 0 else "")

        # ---- Close remaining positions ----
        self._square_off_remaining(df_5m)

        # ---- Flush Langfuse ----
        self.tracer.flush()

        # ---- Journal Summary ----
        summary = self.journal.summary()
        print(f"\n{'─' * 60}")
        print(f"  JOURNAL SUMMARY")
        print(f"{'─' * 60}")
        for k, v in summary.items():
            print(f"  {k}: {v}")

        summary_path = config.JOURNAL_DIR / "journal_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\n  Journal: {self.journal.output_path}")
        print(f"  Summary: {summary_path}")

        # ---- Evaluate Outcomes ----
        try:
            eval_results = self.evaluator.evaluate_all(self.journal.get_records())
            metrics = self.evaluator.compute_metrics()
        except Exception as e:
            AuditService.record_audit_event(
                run_id=self.run_id,
                event_type="EVALUATION_FAILURE",
                message=f"Evaluation failed: {e}",
                severity="error",
                symbol=config.SYMBOL,
            )
            print(f"\n  [Evaluation error: {e}] - marking run incomplete")
            self._mark_run_incomplete(f"Evaluation failure: {e}")
            metrics = {"total_evaluated": 0, "error": str(e)}
            eval_results = []

        print(f"\n{'─' * 60}")
        print(f"  EVALUATING OUTCOMES")
        print(f"{'─' * 60}")
        for k, v in metrics.items():
            print(f"    {k}: {v}")

        # Persist evaluation
        self._persist_evaluation(eval_results)

        eval_path = self.evaluator.save_results()
        print(f"\n  Evaluation results: {eval_path}")

        # ---- Final Report ----
        print(f"\n{'=' * 60}")
        print(f"  POC COMPLETE")
        print(f"{'=' * 60}")
        print(f"  Total decisions:   {summary['total_decisions']}")
        print(f"  Actionable trades: {summary['actionable_trades']}")
        print(f"  Rejected signals:  {summary['rejected_signals']}")
        print(f"  HOLD rate:         {summary['hold_rate']:.2%}")
        if "win_rate" in metrics:
            print(f"  Win rate:          {metrics['win_rate']:.2%}")
        if "avg_net_r_multiple" in metrics:
            print(f"  Avg Net R:         {metrics['avg_net_r_multiple']:.2f}R")
        if "total_net_pnl" in metrics:
            print(f"  Total Net P&L:     ₹{metrics['total_net_pnl']:,.2f}")

        return {
            "journal": self.journal,
            "evaluator": self.evaluator,
            "summary": summary,
            "metrics": metrics,
        }

    def _handle_position_exit(self, T: datetime, decision_point: Dict, active_pos: Dict) -> None:
        """Handle deterministic stop/target/square-off exits."""
        candle_T = decision_point["candle_T"]
        low = float(candle_T["low"])
        high = float(candle_T["high"])
        stop = active_pos["stop"]
        target = active_pos["target"]
        direction = active_pos["direction"]

        stop_hit = False
        target_hit = False

        if direction == "BUY":
            if low <= stop:
                stop_hit = True
            if high >= target:
                target_hit = True
        else:
            if high >= stop:
                stop_hit = True
            if low <= target:
                target_hit = True

        if stop_hit and target_hit:
            stop_hit = True
            target_hit = False

        controller = MarketSessionController()
        is_squareoff_time = T.time() >= controller.config.force_squareoff_time

        if stop_hit or target_hit or is_squareoff_time:
            exit_reason = "stop_hit" if stop_hit else ("target_hit" if target_hit else "forced_squareoff")
            exit_signal = {
                "action": "EXIT",
                "position_id": active_pos["position_id"],
                "exit_reason": exit_reason,
                "suggested_exit_price": stop if stop_hit else (target if target_hit else float(candle_T["close"])),
                "reason": f"Deterministic exit: stop_hit={stop_hit}, target_hit={target_hit}, forced_squareoff={is_squareoff_time}"
            }
            exit_validation = {
                "action": "EXIT",
                "is_valid": True,
                "stop_hit": stop_hit,
                "target_hit": target_hit,
                "forced_exit": is_squareoff_time and not stop_hit and not target_hit
            }
            dec_data, dec_id = DecisionTransactionService.process_decision(
                run_id=self.run_id,
                symbol=config.SYMBOL,
                T=T,
                current_price=float(candle_T["close"]),
                signal=exit_signal,
                validation_result=exit_validation,
            )
            self.decision_ids[f"{T}_exit"] = dec_id
            print(f" [EXITED via {exit_reason}]", end="", flush=True)

            trade_event_type = TradeEventType.STOP_HIT if stop_hit else (
                TradeEventType.TARGET_HIT if target_hit else TradeEventType.FORCED_SQUARE_OFF
            )
            AuditService.record_trade_event(
                run_id=self.run_id, symbol=config.SYMBOL,
                event_type=trade_event_type.value,
                decision_id=dec_id, position_id=active_pos["position_id"],
                direction=direction, price=float(candle_T["close"]),
                reason=exit_reason,
            )

    def _hash_context_package(self, package: Dict[str, Any]) -> str:
        """Hash the exact package fields available at decision time."""
        payload = json.dumps(package, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _persist_agent_audit(self, decision_id: str, agent_result: Dict[str, Any]) -> None:
        """Persist LLM turns and tool calls once the decision row exists."""
        turn_ids = {}
        try:
            for idx, raw_output in enumerate(agent_result.get("raw_responses", [])):
                parsed_type = None
                try:
                    parsed = json.loads(raw_output) if isinstance(raw_output, str) else None
                    if isinstance(parsed, dict):
                        parsed_type = parsed.get("type")
                except Exception:
                    parsed_type = None

                turn_ids[idx] = AuditService.record_agent_turn(
                    run_id=self.run_id,
                    decision_id=decision_id,
                    turn_number=idx,
                    role="assistant",
                    raw_output=str(raw_output),
                    parsed_type=parsed_type,
                    schema_valid=raw_output is not None,
                    schema_errors=[] if raw_output is not None else ["empty_response"],
                )

            for tc in agent_result.get("tool_calls", []):
                result = tc.get("result") or {}
                status = "error" if isinstance(result, dict) and result.get("error") else "success"
                AuditService.record_tool_trace(
                    run_id=self.run_id,
                    decision_id=decision_id,
                    turn_id=turn_ids.get(tc.get("round")),
                    round_num=tc.get("round", 0),
                    tool_name=tc.get("tool", ""),
                    arguments=tc.get("arguments", {}),
                    result=result,
                    status=status,
                    error_message=result.get("error") if isinstance(result, dict) else None,
                    latency_ms=tc.get("latency_ms"),
                )
                if status == "error":
                    AuditService.record_audit_event(
                        run_id=self.run_id,
                        decision_id=decision_id,
                        event_type="TOOL_FAILURE",
                        message=f"Tool '{tc.get('tool')}' failed: {result.get('error')}",
                        severity="warning",
                        symbol=config.SYMBOL,
                        details={"tool": tc.get("tool"), "arguments": tc.get("arguments", {})},
                    )
        except Exception as e:
            AuditService.record_audit_event(
                run_id=self.run_id,
                decision_id=decision_id,
                event_type="AGENT_AUDIT_PERSIST_FAILURE",
                message=f"Failed to persist agent audit rows: {e}",
                severity="error",
                symbol=config.SYMBOL,
            )

    def _square_off_remaining(self, df_5m: pd.DataFrame) -> None:
        """Force square-off any remaining open positions at replay end."""
        active_pos = ReplayStateService.get_active_position(self.run_id, config.SYMBOL)
        print(f"  Position state: {'OPEN' if active_pos else 'CLOSED'}")
        if active_pos:
            square_off_df = df_5m
            if self.replay_date:
                square_off_df = df_5m[df_5m.index.date == self.replay_date]
            last_price = float(square_off_df["close"].iloc[-1])

            exit_signal = {
                "action": "EXIT",
                "position_id": active_pos["position_id"],
                "exit_reason": "replay_end_square_off",
                "suggested_exit_price": last_price,
                "reason": "Deterministic forced square-off at replay end"
            }
            exit_validation = {
                "action": "EXIT",
                "is_valid": True,
                "forced_exit": True
            }
            DecisionTransactionService.process_decision(
                run_id=self.run_id,
                symbol=config.SYMBOL,
                T=df_5m.index[-1],
                current_price=last_price,
                signal=exit_signal,
                validation_result=exit_validation,
            )
            print(f"  Forced square-off at replay end @ {last_price:.2f}")

    def _run_agent_with_audit(
        self,
        package: Dict,
        market_state_text: str,
        harness: ToolHarness,
        has_pos: bool,
    ) -> Dict:
        """Run the DART agent and record structured audit data."""
        try:
            agent_result = self.agent.decide(package, market_state_text, harness)
            return agent_result
        except Exception as e:
            AuditService.record_audit_event(
                run_id=self.run_id,
                event_type="AGENT_FAILURE",
                message=f"Agent error: {e}",
                severity="error",
                symbol=config.SYMBOL,
                details={"has_position": has_pos},
            )
            print(f"\n  [Agent error: {e}]")
            current_pos = {"position_id": None}
            try:
                pos = ReplayStateService.get_active_position(self.run_id, config.SYMBOL)
                if pos:
                    current_pos = pos
            except Exception:
                pass

            fallback = {
                "raw_responses": [f"Agent error: {e}"],
                "tool_calls": [],
                "final_signal": {
                    "type": "final_signal",
                    "action": "HOLD" if has_pos else "SKIP",
                    "confidence": 0.0,
                    "dart": {"direction": "", "area": "", "risk": "", "trigger": ""},
                    "checklist": {
                        "market_regime": "unclear", "session_type": "unclear",
                        "structure_state": "unclear", "location_quality": 0,
                        "trigger_quality": 0, "risk_quality": 0,
                        "volume_confirmation": 0, "reason_to_wait": "LLM failed; flat-state fallback is SKIP",
                    },
                    "entry": None, "stop": None, "target": None,
                    "position_id": current_pos.get("position_id") if has_pos else None,
                    "thesis_health": "valid" if has_pos else "not_applicable",
                    "reason": f"Agent error: {e}",
                    "invalidation": None,
                },
            }
            return fallback

    def _trace_to_langfuse(
        self,
        root_span,
        agent_result: Dict,
        signal: Dict,
        market_state_text: str,
        validation_result: Dict,
    ) -> None:
        """Log decision trace to Langfuse."""
        if not root_span:
            return

        for i, raw_resp in enumerate(agent_result.get("raw_responses", [])):
            approx_prompt_tokens = len(market_state_text) // 4
            approx_resp_tokens = len(str(raw_resp)) // 4
            self.tracer.add_generation(
                parent_span=root_span,
                name=f"llm_call_{i+1}",
                model=config.MODEL_NAME,
                input_data=market_state_text[:2000],
                output_data=str(raw_resp)[:2000],
                usage={
                    "input": approx_prompt_tokens,
                    "output": approx_resp_tokens,
                    "total": approx_prompt_tokens + approx_resp_tokens,
                },
                metadata={
                    "call_number": i + 1,
                    "total_calls": len(agent_result.get("raw_responses", [])),
                },
            )

        for tc in agent_result.get("tool_calls", []):
            self.tracer.add_span(
                parent_span=root_span,
                name=f"tool_{tc['tool']}",
                input_data={"tool": tc["tool"], "arguments": tc.get("arguments")},
                output_data=tc.get("result"),
                metadata={"reason": tc.get("reason", "")},
            )

        self.tracer.add_span(
            parent_span=root_span,
            name="validation",
            input_data={
                "action": signal.get("action"),
                "entry": signal.get("entry"),
                "stop": signal.get("stop"),
                "target": signal.get("target"),
            },
            output_data={
                "is_valid": validation_result.get("is_valid"),
                "action": validation_result.get("action"),
                "rejection_reason": validation_result.get("rejection_reason"),
            },
        )

        self.tracer.end_root_span(
            root_span,
            output_data={
                "action": validation_result.get("action", signal.get("action")),
                "confidence": signal.get("confidence"),
                "is_valid": validation_result.get("is_valid"),
                "tool_calls_made": len(agent_result.get("tool_calls", [])),
            },
        )

    def _persist_evaluation(self, eval_results: List[Dict]) -> None:
        """Persist evaluation outcomes to Postgres."""
        for eval_res in eval_results:
            t_str = eval_res["decision_time"]
            dec_id = None
            eval_dt = pd.Timestamp(t_str)
            action = eval_res.get("action")

            if action == "EXIT":
                for k, d_id in self.decision_ids.items():
                    if k.endswith("_exit") and pd.Timestamp(k[:-5]) == eval_dt:
                        dec_id = d_id
                        break

            if not dec_id:
                for k, d_id in self.decision_ids.items():
                    if not k.endswith("_exit") and pd.Timestamp(k) == eval_dt:
                        dec_id = d_id
                        break

            if dec_id:
                outcome_label = "ambiguous"
                outcome = eval_res.get("outcome")
                if outcome == "target_hit" or outcome == "target_first":
                    outcome_label = "win"
                elif outcome == "stop_hit" or outcome == "stop_first":
                    outcome_label = "loss"
                elif outcome == "square_off_at_close":
                    net_pnl = eval_res.get("net_pnl", 0)
                    outcome_label = "win" if net_pnl > 0 else ("loss" if net_pnl < 0 else "scratch")
                elif outcome == "no_trade":
                    outcome_label = "no_trade"

                setup_tags = []
                action = eval_res.get("action")
                if action in ("BUY", "SELL"):
                    setup_tags.append(action)
                hold_qual = eval_res.get("hold_quality")
                if hold_qual:
                    setup_tags.append(hold_qual)

                try:
                    OutcomeFeedbackService.record_feedback(
                        decision_id=dec_id,
                        outcome_label=outcome_label,
                        net_r=eval_res.get("net_r_multiple", 0.0) or eval_res.get("r_multiple", 0.0) or 0.0,
                        mfe_pct=eval_res.get("max_favorable_excursion_pct", 0.0) or 0.0,
                        mae_pct=eval_res.get("max_adverse_excursion_pct", 0.0) or 0.0,
                        setup_tags=setup_tags,
                    )
                except Exception as e:
                    AuditService.record_audit_event(
                        run_id=self.run_id,
                        event_type="OUTCOME_PERSIST_FAILURE",
                        message=f"Failed to persist outcome for decision {dec_id}: {e}",
                        severity="error",
                        symbol=config.SYMBOL,
                    )
