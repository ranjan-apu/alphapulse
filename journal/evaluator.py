"""
Feedback evaluator: scores each recorded signal against future candles.
Inspects T+15m, T+30m, max favorable/adverse excursion,
stop/target touch, and computes win rate and average R.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import json
from pathlib import Path

from config import config


class FeedbackEvaluator:
    """
    Evaluates recorded signals by looking forward through the same trading session.
    Leakage rule: evaluator runs AFTER the decision is recorded.
    """

    def __init__(self, data_5m: pd.DataFrame):
        self.data_5m = data_5m.sort_index()
        self.results: List[Dict] = []

    def evaluate_signal(
        self,
        record: Dict,
        decision_time: datetime,
        session_end: datetime,
    ) -> Dict:
        """
        Evaluate a single signal against future candles.

        Returns evaluation dict with:
        - T+15m price and outcome
        - T+30m price and outcome
        - Max favorable/adverse excursion
        - Stop/target touch
        - Result in R multiples (if actionable)
        """
        action = record.get("action")
        entry = record.get("entry")
        stop = record.get("stop")
        target = record.get("target")
        sizing = record.get("sizing", {})
        current_price = record.get("current_price")

        eval_result = {
            "decision_time": str(decision_time),
            "action": action,
            "entry": entry,
            "stop": stop,
            "target": target,
        }

        # Get future candles
        future = self.data_5m[self.data_5m.index > decision_time]
        # Only same-session candles
        future = future[future.index <= session_end]

        if len(future) == 0:
            eval_result["evaluation"] = "no_future_data"
            self.results.append(eval_result)
            return eval_result

        # T+15m (3 candles later)
        t_15 = decision_time + timedelta(minutes=15)
        t_30 = decision_time + timedelta(minutes=30)

        # Find closest candle to T+15m
        future_15m = future[future.index <= t_15]
        if len(future_15m) > 0:
            candle_15m = future_15m.iloc[-1]
            price_15m = float(candle_15m["close"])
            eval_result["t_plus_15m_price"] = price_15m
            eval_result["t_plus_15m_change_pct"] = (
                round((price_15m - current_price) / current_price * 100, 2)
                if current_price and current_price > 0 else None
            )
        else:
            eval_result["t_plus_15m_price"] = None
            eval_result["t_plus_15m_change_pct"] = None

        # T+30m
        future_30m = future[future.index <= t_30]
        if len(future_30m) > 0:
            candle_30m = future_30m.iloc[-1]
            price_30m = float(candle_30m["close"])
            eval_result["t_plus_30m_price"] = price_30m
            eval_result["t_plus_30m_change_pct"] = (
                round((price_30m - current_price) / current_price * 100, 2)
                if current_price and current_price > 0 else None
            )
        else:
            eval_result["t_plus_30m_price"] = None
            eval_result["t_plus_30m_change_pct"] = None

        # Max favorable and adverse excursion
        future_highs = future["high"].values
        future_lows = future["low"].values
        future_closes = future["close"].values

        max_high = float(np.max(future_highs)) if len(future_highs) > 0 else current_price
        min_low = float(np.min(future_lows)) if len(future_lows) > 0 else current_price

        eval_result["max_favorable_excursion_pct"] = (
            round((max_high - current_price) / current_price * 100, 2)
            if current_price and current_price > 0 else None
        )
        eval_result["max_adverse_excursion_pct"] = (
            round((current_price - min_low) / current_price * 100, 2)
            if current_price and current_price > 0 else None
        )

        # Stop/target touch tracking (for actionable trades)
        if action in ("BUY", "SELL") and entry and stop and target:
            eval_result["stop_touched"] = False
            eval_result["target_touched"] = False
            eval_result["stop_touch_time"] = None
            eval_result["target_touch_time"] = None

            stop_touched_at = None
            target_touched_at = None

            for idx, row in future.iterrows():
                high = float(row["high"])
                low = float(row["low"])

                if action == "BUY":
                    if not eval_result["stop_touched"] and low <= stop:
                        eval_result["stop_touched"] = True
                        eval_result["stop_touch_time"] = str(idx)
                        stop_touched_at = idx
                    if not eval_result["target_touched"] and high >= target:
                        eval_result["target_touched"] = True
                        eval_result["target_touch_time"] = str(idx)
                        target_touched_at = idx
                else:  # SELL
                    if not eval_result["stop_touched"] and high >= stop:
                        eval_result["stop_touched"] = True
                        eval_result["stop_touch_time"] = str(idx)
                        stop_touched_at = idx
                    if not eval_result["target_touched"] and low <= target:
                        eval_result["target_touched"] = True
                        eval_result["target_touch_time"] = str(idx)
                        target_touched_at = idx

                # If both touched in same candle, use the one that happened first
                # (stop first = loss, target first = win)
                # We use a simple rule: if both in same candle, evaluate based on
                # which one was more "extreme" or use close proximity

            # Determine outcome
            if eval_result["stop_touched"] and eval_result["target_touched"]:
                if stop_touched_at and target_touched_at:
                    if stop_touched_at < target_touched_at:
                        eval_result["outcome"] = "stop_first"
                    elif target_touched_at < stop_touched_at:
                        eval_result["outcome"] = "target_first"
                    else:
                        eval_result["outcome"] = "simultaneous_touch"
                else:
                    eval_result["outcome"] = "both_touched"
            elif eval_result["stop_touched"]:
                eval_result["outcome"] = "stop_hit"
            elif eval_result["target_touched"]:
                eval_result["outcome"] = "target_hit"
            else:
                # Neither touched - square off at session end
                final_close = float(future_closes[-1]) if len(future_closes) > 0 else current_price
                eval_result["outcome"] = "square_off_at_close"
                eval_result["square_off_price"] = final_close

            # Compute R-multiple result
            if sizing and sizing.get("quantity", 0) > 0:
                gross_risk = sizing.get("gross_risk", 0)
                qty = sizing.get("quantity", 0)

                if eval_result["outcome"] == "target_hit":
                    gross_pnl = sizing.get("gross_target_profit", 0)
                    net_pnl = gross_pnl - sizing.get("total_charges", 0)
                    eval_result["gross_pnl"] = gross_pnl
                    eval_result["net_pnl"] = net_pnl
                    eval_result["r_multiple"] = round(gross_pnl / gross_risk, 2) if gross_risk > 0 else 0
                    eval_result["net_r_multiple"] = round(net_pnl / gross_risk, 2) if gross_risk > 0 else 0

                elif eval_result["outcome"] == "stop_hit":
                    gross_pnl = -gross_risk
                    net_pnl = gross_pnl - sizing.get("total_charges", 0)
                    eval_result["gross_pnl"] = gross_pnl
                    eval_result["net_pnl"] = net_pnl
                    eval_result["r_multiple"] = -1.0
                    eval_result["net_r_multiple"] = round(net_pnl / gross_risk, 2) if gross_risk > 0 else 0

                elif eval_result["outcome"] == "square_off_at_close":
                    square_off = eval_result.get("square_off_price", current_price)
                    if action == "BUY":
                        gross_pnl = qty * (square_off - entry)
                    else:
                        gross_pnl = qty * (entry - square_off)
                    net_pnl = gross_pnl - sizing.get("total_charges", 0)
                    eval_result["gross_pnl"] = float(round(gross_pnl, 2))
                    eval_result["net_pnl"] = float(round(net_pnl, 2))
                    eval_result["r_multiple"] = round(gross_pnl / gross_risk, 2) if gross_risk > 0 else 0
                    eval_result["net_r_multiple"] = round(net_pnl / gross_risk, 2) if gross_risk > 0 else 0

                else:
                    eval_result["gross_pnl"] = 0
                    eval_result["net_pnl"] = 0
                    eval_result["r_multiple"] = 0
                    eval_result["net_r_multiple"] = 0

        else:
            # HOLD evaluation
            eval_result["outcome"] = "no_trade"
            # Was HOLD a good decision?
            if current_price:
                t15 = eval_result.get("t_plus_15m_change_pct")
                t30 = eval_result.get("t_plus_30m_change_pct")
                mfe = eval_result.get("max_favorable_excursion_pct", 0)
                mae = eval_result.get("max_adverse_excursion_pct", 0)

                # A "good hold" means the market was choppy (small range) or
                # would have hit a stop quickly
                if t15 is not None and abs(t15) < 0.2 and (mae and mae < 0.3):
                    eval_result["hold_quality"] = "good_hold_avoided_chop"
                elif mfe and mfe > 0.5 and (mae and mae < 0.3):
                    eval_result["hold_quality"] = "missed_opportunity"
                else:
                    eval_result["hold_quality"] = "neutral"

        self.results.append(eval_result)
        return eval_result

    def evaluate_all(
        self,
        journal_records: List[Dict],
    ) -> List[Dict]:
        """Evaluate all journal records."""
        session_end_cache = {}

        for record in journal_records:
            dt = pd.Timestamp(record["timestamp"])
            if dt.tzinfo is None:
                dt = dt.tz_localize("Asia/Kolkata")

            # Get session end
            date_key = dt.date()
            if date_key not in session_end_cache:
                session_end = dt.replace(
                    hour=config.SESSION_END_HOUR,
                    minute=config.SESSION_END_MINUTE,
                    second=0, microsecond=0
                )
                session_end_cache[date_key] = session_end
            else:
                session_end = session_end_cache[date_key]

            self.evaluate_signal(record, dt, session_end)

        return self.results

    def compute_metrics(self) -> Dict:
        """Compute aggregate metrics from evaluation results."""
        trades = [r for r in self.results if r["action"] in ("BUY", "SELL")]
        holds = [r for r in self.results if r["action"] == "HOLD"]

        metrics = {
            "total_evaluated": len(self.results),
            "total_trades": len(trades),
            "total_holds": len(holds),
        }

        if trades:
            wins = [r for r in trades if r.get("outcome") == "target_hit"]
            losses = [r for r in trades if r.get("outcome") == "stop_hit"]
            sq_offs = [r for r in trades if r.get("outcome") == "square_off_at_close"]

            metrics["wins"] = len(wins)
            metrics["losses"] = len(losses)
            metrics["square_offs"] = len(sq_offs)
            metrics["win_rate"] = round(len(wins) / len(trades), 3) if trades else 0

            # Average R
            r_values = [r.get("r_multiple", 0) for r in trades if r.get("r_multiple") is not None]
            net_r_values = [r.get("net_r_multiple", 0) for r in trades if r.get("net_r_multiple") is not None]

            metrics["avg_r_multiple"] = round(float(np.mean(r_values)), 3) if r_values else 0
            metrics["avg_net_r_multiple"] = round(float(np.mean(net_r_values)), 3) if net_r_values else 0

            if r_values:
                metrics["sum_r_multiple"] = round(float(np.sum(r_values)), 2)
            if net_r_values:
                metrics["sum_net_r_multiple"] = round(float(np.sum(net_r_values)), 2)

            # P&L
            gross_pnls = [r.get("gross_pnl", 0) for r in trades if r.get("gross_pnl") is not None]
            net_pnls = [r.get("net_pnl", 0) for r in trades if r.get("net_pnl") is not None]
            metrics["total_gross_pnl"] = round(float(np.sum(gross_pnls)), 2) if gross_pnls else 0
            metrics["total_net_pnl"] = round(float(np.sum(net_pnls)), 2) if net_pnls else 0
            profitable = [p for p in net_pnls if p > 0]
            losing = [p for p in net_pnls if p < 0]
            metrics["profitable_trades"] = len(profitable)
            metrics["losing_trades"] = len(losing)
            metrics["flat_trades"] = len([p for p in net_pnls if p == 0])
            metrics["total_net_profit"] = round(float(np.sum(profitable)), 2) if profitable else 0
            metrics["total_net_loss"] = round(float(np.sum(losing)), 2) if losing else 0
            metrics["profit_factor"] = (
                round(abs(metrics["total_net_profit"] / metrics["total_net_loss"]), 3)
                if metrics["total_net_loss"] < 0 else None
            )

            # Outcomes
            outcome_counts = {}
            for r in trades:
                outcome = r.get("outcome", "unknown")
                outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
            metrics["outcome_distribution"] = outcome_counts

        if holds:
            good_holds = sum(1 for h in holds if h.get("hold_quality") == "good_hold_avoided_chop")
            missed = sum(1 for h in holds if h.get("hold_quality") == "missed_opportunity")
            metrics["hold_good_avoids"] = good_holds
            metrics["hold_missed_opportunities"] = missed

        return metrics

    def save_results(self, output_path: Path = None):
        """Save evaluation results to JSON."""
        output_path = output_path or config.JOURNAL_DIR / "evaluation_results.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        results = {
            "metrics": self.compute_metrics(),
            "details": self.results,
        }

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        return output_path
