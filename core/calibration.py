"""
Agent calibration and experiment tracking (Section 6.9 + Phase 8).

Tracks whether agent confidence is meaningful:
- Accuracy by confidence bucket
- Average net R by confidence bucket
- Win rate by setup tag
- Average adverse excursion by setup type
- False breakout frequency
- SKIP missed-opportunity rate
- HOLD quality

Calibration hints only shown when statistics are reliable
(min_trades_per_bucket threshold gating).

Implements the fallback policy from Section 9.4.5:
  HIGH confidence episodes >= min_trades → use HIGH only
  HIGH+MID >= min_trades → use combined with 0.5x MID weight
  HIGH+MID < min_trades → use LOW with 0.25x weight
  total < 5 → no hint
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


@dataclass
class CalibrationBucket:
    """Statistics for a single calibration bucket."""
    bucket_key: str         # e.g., 'confidence_0.60-0.70', 'setup_breakout'
    bucket_type: str        # 'confidence', 'setup_tag', 'regime', 'session_type'
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    sum_net_r: float = 0.0
    sum_mfe: float = 0.0    # Max favorable excursion sum
    sum_mae: float = 0.0    # Max adverse excursion sum
    sum_gross_r: float = 0.0

    # For reflection-quality weighted stats
    high_confidence_wins: int = 0
    high_confidence_losses: int = 0
    high_confidence_sum_r: float = 0.0
    medium_confidence_wins: int = 0
    medium_confidence_losses: int = 0
    medium_confidence_sum_r: float = 0.0
    low_confidence_wins: int = 0
    low_confidence_losses: int = 0
    low_confidence_sum_r: float = 0.0

    min_samples_for_hint: int = 20
    absolute_min_samples: int = 5

    @property
    def win_rate(self) -> Optional[float]:
        if self.total_trades == 0:
            return None
        return self.wins / self.total_trades

    @property
    def avg_net_r(self) -> Optional[float]:
        if self.total_trades == 0:
            return None
        return self.sum_net_r / self.total_trades

    @property
    def avg_mfe(self) -> Optional[float]:
        if self.total_trades == 0:
            return None
        return self.sum_mfe / self.total_trades

    @property
    def avg_mae(self) -> Optional[float]:
        if self.total_trades == 0:
            return None
        return self.sum_mae / self.total_trades

    def can_show_hint(self) -> Tuple[bool, str, float]:
        """
        Determine if a calibration hint can be shown, following Section 9.4.5.

        Returns:
            (can_show, confidence_label, sample_size)
        """
        high_count = self.high_confidence_wins + self.high_confidence_losses
        medium_count = self.medium_confidence_wins + self.medium_confidence_losses
        low_count = self.low_confidence_wins + self.low_confidence_losses

        # HIGH only
        if high_count >= self.min_samples_for_hint:
            return True, "high-confidence estimate", high_count

        # HIGH + MEDIUM combined
        combined_hm = high_count + medium_count
        if combined_hm >= self.min_samples_for_hint:
            return True, "medium-confidence estimate", combined_hm

        # HIGH + MEDIUM + LOW (low weight)
        total = high_count + medium_count + low_count
        if total >= self.absolute_min_samples and total < self.min_samples_for_hint:
            return True, "low-confidence estimate, small sample", total

        if total < self.absolute_min_samples:
            return False, "insufficient samples", total

        return False, "", 0

    def get_weighted_stats(self) -> dict:
        """
        Compute weighted statistics based on reflection confidence.

        Weights: HIGH=1.0, MEDIUM=0.5, LOW=0.25
        """
        weighted_wins = (
            self.high_confidence_wins * 1.0
            + self.medium_confidence_wins * 0.5
            + self.low_confidence_wins * 0.25
        )
        weighted_losses = (
            self.high_confidence_losses * 1.0
            + self.medium_confidence_losses * 0.5
            + self.low_confidence_losses * 0.25
        )
        weighted_sum_r = (
            self.high_confidence_sum_r * 1.0
            + self.medium_confidence_sum_r * 0.5
            + self.low_confidence_sum_r * 0.25
        )
        weighted_n = weighted_wins + weighted_losses

        return {
            "weighted_win_rate": weighted_wins / weighted_n if weighted_n > 0 else None,
            "weighted_avg_net_r": weighted_sum_r / weighted_n if weighted_n > 0 else None,
            "effective_samples": round(weighted_n, 1),
        }


class CalibrationTracker:
    """
    Tracks performance statistics across multiple dimensions:
    - Confidence buckets
    - Setup tags
    - Market regimes
    - Session types

    Provides calibration hints to the agent when statistics are reliable.
    """

    def __init__(self, min_samples_for_hint: int = 20):
        self._buckets: Dict[str, CalibrationBucket] = {}
        self.min_samples_for_hint = min_samples_for_hint

        # Performance tracking
        self.total_skips: int = 0
        self.total_holds: int = 0
        self.good_holds: int = 0
        self.bad_holds: int = 0
        self.missed_opportunities: int = 0

    def record_outcome(
        self,
        action: str,
        outcome: dict,
        setup_tags: List[str],
        market_regime: str,
        session_type: str,
        confidence: float,
        net_r: float,
        is_win: bool,
        reflection_level: str = "HIGH",
        mfe_pct: Optional[float] = None,
        mae_pct: Optional[float] = None,
    ):
        """
        Record a trade outcome for calibration tracking.

        Args:
            action: BUY, SELL, SKIP, HOLD, EXIT
            outcome: Outcome dict from evaluator
            setup_tags: Tags for the setup type
            market_regime: Market regime at decision time
            session_type: Session type at decision time
            confidence: Agent's confidence (0-1)
            net_r: Net R multiple realized
            is_win: Whether the trade was profitable
            reflection_level: HIGH/MEDIUM/LOW from reflection writer
            mfe_pct: Max favorable excursion %
            mae_pct: Max adverse excursion %
        """
        if action in ("SKIP",):
            self.total_skips += 1
            skip_quality = outcome.get("skip_quality", "")
            if skip_quality in ("missed_long_opportunity", "missed_short_opportunity"):
                self.missed_opportunities += 1
            return

        if action == "HOLD":
            self.total_holds += 1
            hold_quality = outcome.get("hold_quality", "")
            if hold_quality == "good_hold_avoided_chop":
                self.good_holds += 1
            elif hold_quality == "bad_hold_should_exit":
                self.bad_holds += 1
            return

        if action not in ("BUY", "SELL"):
            return

        # Confidence bucket (0.0-0.3, 0.3-0.5, 0.5-0.7, 0.7-0.85, 0.85-1.0)
        conf_bucket = self._confidence_bucket(confidence)
        self._update_bucket(
            f"confidence_{conf_bucket}", "confidence",
            is_win, net_r, reflection_level, mfe_pct or 0, mae_pct or 0, confidence
        )

        # Setup tag buckets
        for tag in setup_tags:
            self._update_bucket(
                f"setup_{tag}", "setup_tag",
                is_win, net_r, reflection_level, mfe_pct or 0, mae_pct or 0, confidence
            )

        # Regime bucket
        if market_regime:
            self._update_bucket(
                f"regime_{market_regime}", "regime",
                is_win, net_r, reflection_level, mfe_pct or 0, mae_pct or 0, confidence
            )

        # Session type bucket
        if session_type:
            self._update_bucket(
                f"session_{session_type}", "session_type",
                is_win, net_r, reflection_level, mfe_pct or 0, mae_pct or 0, confidence
            )

    def _update_bucket(
        self,
        bucket_key: str,
        bucket_type: str,
        is_win: bool,
        net_r: float,
        reflection_level: str,
        mfe: float,
        mae: float,
        confidence: float,
    ):
        """Update a single calibration bucket."""
        if bucket_key not in self._buckets:
            self._buckets[bucket_key] = CalibrationBucket(
                bucket_key=bucket_key,
                bucket_type=bucket_type,
                min_samples_for_hint=self.min_samples_for_hint,
            )

        bucket = self._buckets[bucket_key]
        bucket.total_trades += 1
        bucket.sum_net_r += net_r
        bucket.sum_mfe += mfe
        bucket.sum_mae += mae
        bucket.sum_gross_r += abs(net_r)  # Gross R is absolute value of net R in this context

        if is_win:
            bucket.wins += 1
        else:
            bucket.losses += 1

        # Reflection-level tracking
        if reflection_level == "HIGH":
            if is_win:
                bucket.high_confidence_wins += 1
            else:
                bucket.high_confidence_losses += 1
            bucket.high_confidence_sum_r += net_r
        elif reflection_level == "MEDIUM":
            if is_win:
                bucket.medium_confidence_wins += 1
            else:
                bucket.medium_confidence_losses += 1
            bucket.medium_confidence_sum_r += net_r
        elif reflection_level == "LOW":
            if is_win:
                bucket.low_confidence_wins += 1
            else:
                bucket.low_confidence_losses += 1
            bucket.low_confidence_sum_r += net_r

    @staticmethod
    def _confidence_bucket(confidence: float) -> str:
        """Map confidence value to a bucket label."""
        if confidence < 0.3:
            return "0.00-0.30"
        elif confidence < 0.5:
            return "0.30-0.50"
        elif confidence < 0.7:
            return "0.50-0.70"
        elif confidence < 0.85:
            return "0.70-0.85"
        else:
            return "0.85-1.00"

    def get_calibration_hints(
        self,
        setup_tags: Optional[List[str]] = None,
        regime: Optional[str] = None,
        confidence: Optional[float] = None,
    ) -> str:
        """
        Generate calibration hints for the agent prompt.

        Only shows hints for buckets with sufficient samples.
        """
        hints = []

        # Confidence bucket hint
        if confidence is not None:
            conf_key = f"confidence_{self._confidence_bucket(confidence)}"
            hint = self._format_bucket_hint(conf_key)
            if hint:
                hints.append(hint)

        # Setup tag hints
        if setup_tags:
            for tag in setup_tags[:3]:
                tag_key = f"setup_{tag}"
                hint = self._format_bucket_hint(tag_key)
                if hint:
                    hints.append(hint)

        # Regime hint
        if regime:
            regime_key = f"regime_{regime}"
            hint = self._format_bucket_hint(regime_key)
            if hint:
                hints.append(hint)

        if not hints:
            return ""

        return "Recent calibration:\n" + "\n".join(f"  {h}" for h in hints)

    def _format_bucket_hint(self, bucket_key: str) -> Optional[str]:
        """Format a single bucket as a calibration hint string."""
        bucket = self._buckets.get(bucket_key)
        if not bucket:
            return None

        can_show, label, n = bucket.can_show_hint()
        if not can_show:
            return None

        stats = bucket.get_weighted_stats()
        wr = stats.get("weighted_win_rate")
        ar = stats.get("weighted_avg_net_r")
        es = stats.get("effective_samples", 0)

        if wr is None or ar is None:
            return None

        return (
            f"{bucket_key}: {es:.0f} trades, "
            f"{wr:.0%} win rate, {ar:+.2f} avg net R "
            f"[{label}, N={n}]"
        )

    def get_experiment_metrics(self) -> dict:
        """
        Compute full experiment metrics for evaluation.

        Returns metrics dict suitable for experiment_runs.metrics JSON field.
        """
        all_trades = [
            b for b in self._buckets.values()
            if b.bucket_type == "confidence"
        ]
        total_trades = sum(b.total_trades for b in all_trades)
        total_wins = sum(b.wins for b in all_trades)
        total_net_r = sum(b.sum_net_r for b in all_trades)

        # By setup tag
        setup_stats = {}
        for key, bucket in self._buckets.items():
            if bucket.bucket_type == "setup_tag" and bucket.total_trades >= 5:
                setup_stats[key.replace("setup_", "")] = {
                    "trades": bucket.total_trades,
                    "win_rate": bucket.win_rate,
                    "avg_net_r": bucket.avg_net_r,
                    "avg_mfe": bucket.avg_mfe,
                    "avg_mae": bucket.avg_mae,
                }

        return {
            "total_trades": total_trades,
            "total_wins": total_wins,
            "win_rate": total_wins / max(total_trades, 1),
            "avg_net_r": total_net_r / max(total_trades, 1),
            "total_skips": self.total_skips,
            "total_holds": self.total_holds,
            "good_holds": self.good_holds,
            "bad_holds": self.bad_holds,
            "missed_opportunities": self.missed_opportunities,
            "setup_performance": setup_stats,
            "confidence_buckets": {
                key.replace("confidence_", ""): {
                    "trades": b.total_trades,
                    "win_rate": b.win_rate,
                    "avg_net_r": b.avg_net_r,
                }
                for key, b in self._buckets.items()
                if b.bucket_type == "confidence" and b.total_trades > 0
            },
        }

    def to_postgres_records(self, run_id: str) -> List[dict]:
        """Serialize calibration stats for Postgres storage."""
        records = []
        import uuid

        for key, bucket in self._buckets.items():
            records.append({
                "stat_id": f"cal_{uuid.uuid4().hex[:12]}",
                "run_id": run_id,
                "bucket_key": bucket.bucket_key,
                "bucket_type": bucket.bucket_type,
                "total_trades": bucket.total_trades,
                "wins": bucket.wins,
                "losses": bucket.losses,
                "win_rate": bucket.win_rate,
                "avg_net_r": bucket.avg_net_r,
                "sum_net_r": bucket.sum_net_r,
                "avg_mfe": bucket.avg_mfe,
                "avg_mae": bucket.avg_mae,
                "min_samples_for_hint": bucket.min_samples_for_hint,
                "last_updated": "NOW()",
            })

        return records
