"""
Data snapshot versioning for experiment reproducibility.

Each experiment uses three timeframes (weekly, daily, intraday).
A snapshot set groups these three snapshots. SHA-256 hashing ensures
that if raw data is ever re-sourced or re-adjusted, the hash mismatch
flags comparisons as invalid.

Corporate actions rule (Section 1.1):
- Historical context uses adjusted close (for charting/indicators)
- Trade execution uses raw unadjusted prices
- Both adjusted and unadjusted DataFrames are stored separately
"""
import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np


@dataclass
class DataSnapshot:
    """A single timeframe snapshot with content hash."""
    snapshot_id: str
    set_id: str
    timeframe: str          # 'weekly', 'daily', 'intraday_15min'
    period_start: date
    period_end: date
    candle_count: int
    first_candle: datetime
    last_candle: datetime
    data_hash: str          # SHA-256 of sorted OHLCV rows
    yfinance_period: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now().astimezone())


@dataclass
class DataSnapshotSet:
    """A set of three timeframe snapshots for one experiment run."""
    set_id: str
    symbol: str
    source: str             # 'yahoo_finance', 'broker_backfill', etc.
    adjusted_for_splits: bool = False
    adjusted_for_dividends: bool = False
    snapshots: Dict[str, DataSnapshot] = field(default_factory=dict)
    notes: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now().astimezone())

    def is_complete(self) -> bool:
        """Check if all three timeframes are present."""
        return all(
            tf in self.snapshots
            for tf in ("weekly", "daily", "intraday_15min")
        )

    def get_manifest(self) -> dict:
        """Get a manifest for experiment run metadata."""
        return {
            "set_id": self.set_id,
            "symbol": self.symbol,
            "source": self.source,
            "adjusted_for_splits": self.adjusted_for_splits,
            "adjusted_for_dividends": self.adjusted_for_dividends,
            "snapshot_hashes": {
                tf: sn.data_hash for tf, sn in self.snapshots.items()
            },
            "timeframes_complete": self.is_complete(),
        }


class DataSnapshotManager:
    """
    Creates and validates data snapshots for experiment reproducibility.

    Usage:
        mgr = DataSnapshotManager(symbol="RELIANCE")
        mgr.create_snapshot_set(df_weekly, df_daily, df_5m_resampled)
        manifest = mgr.get_manifest()
    """

    def __init__(self, symbol: str, source: str = "yahoo_finance"):
        self.symbol = symbol
        self.source = source
        self._sets: Dict[str, DataSnapshotSet] = {}

    def create_snapshot_set(
        self,
        df_weekly: pd.DataFrame,
        df_daily: pd.DataFrame,
        df_intraday_15min: pd.DataFrame,
        adjusted_for_splits: bool = False,
        adjusted_for_dividends: bool = False,
        notes: str = "",
    ) -> DataSnapshotSet:
        """
        Create a snapshot set from three DataFrames.

        Args:
            df_weekly: Weekly OHLCV data
            df_daily: Daily OHLCV data
            df_intraday_15min: 15-minute intraday OHLCV data
            adjusted_for_splits: Whether data is split-adjusted
            adjusted_for_dividends: Whether data is dividend-adjusted

        Returns:
            DataSnapshotSet with all three snapshots hashed
        """
        set_id = f"dss_{uuid.uuid4().hex[:12]}"
        snapshot_set = DataSnapshotSet(
            set_id=set_id,
            symbol=self.symbol,
            source=self.source,
            adjusted_for_splits=adjusted_for_splits,
            adjusted_for_dividends=adjusted_for_dividends,
            notes=notes,
        )

        # Create weekly snapshot
        if not df_weekly.empty:
            snap = self._create_snapshot(
                set_id, df_weekly, "weekly",
                yfinance_period=f"{len(df_weekly)}wk"
            )
            snapshot_set.snapshots["weekly"] = snap

        # Create daily snapshot
        if not df_daily.empty:
            snap = self._create_snapshot(
                set_id, df_daily, "daily",
                yfinance_period=f"{len(df_daily)}d"
            )
            snapshot_set.snapshots["daily"] = snap

        # Create intraday 15min snapshot
        if not df_intraday_15min.empty:
            snap = self._create_snapshot(
                set_id, df_intraday_15min, "intraday_15min",
                yfinance_period=f"{len(df_intraday_15min)}bars"
            )
            snapshot_set.snapshots["intraday_15min"] = snap

        self._sets[set_id] = snapshot_set
        return snapshot_set

    def _create_snapshot(
        self,
        set_id: str,
        df: pd.DataFrame,
        timeframe: str,
        yfinance_period: Optional[str] = None,
    ) -> DataSnapshot:
        """Create a single snapshot with SHA-256 hash."""
        snapshot_id = f"sn_{uuid.uuid4().hex[:12]}"

        # Ensure sorted index
        df = df.sort_index()

        # Compute SHA-256 hash of all OHLCV rows
        data_hash = self._hash_ohlcv(df)

        # Get period boundaries
        period_start = df.index[0].date() if hasattr(df.index[0], 'date') else df.index[0]
        period_end = df.index[-1].date() if hasattr(df.index[-1], 'date') else df.index[-1]

        return DataSnapshot(
            snapshot_id=snapshot_id,
            set_id=set_id,
            timeframe=timeframe,
            period_start=period_start,
            period_end=period_end,
            candle_count=len(df),
            first_candle=df.index[0],
            last_candle=df.index[-1],
            data_hash=data_hash,
            yfinance_period=yfinance_period,
        )

    @staticmethod
    def _hash_ohlcv(df: pd.DataFrame) -> str:
        """
        Compute SHA-256 hash of OHLCV data.

        Uses sorted OHLCV values to ensure deterministic hashing.
        The hash covers: open, high, low, close, volume for every row.
        """
        # Sort by index for determinism
        df_sorted = df.sort_index()

        # Build a canonical string representation
        rows = []
        for idx, row in df_sorted.iterrows():
            rows.append(
                f"{idx}|{row['open']:.12g}|{row['high']:.12g}|"
                f"{row['low']:.12g}|{row['close']:.12g}|{row['volume']:.0f}"
            )

        canonical = "\n".join(rows)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @staticmethod
    def verify_hash(df: pd.DataFrame, expected_hash: str) -> bool:
        """Verify a DataFrame matches an expected hash."""
        actual = DataSnapshotManager._hash_ohlcv(df)
        return actual == expected_hash

    def validate_snapshot_set(self, set_id: str) -> Tuple[bool, List[str]]:
        """
        Validate that a snapshot set is complete and consistent.

        Returns:
            (is_valid, list_of_issues)
        """
        if set_id not in self._sets:
            return False, [f"Snapshot set {set_id} not found"]

        snapshot_set = self._sets[set_id]
        issues = []

        if not snapshot_set.is_complete():
            missing = {"weekly", "daily", "intraday_15min"} - set(snapshot_set.snapshots.keys())
            issues.append(f"Missing timeframes: {missing}")

        # Check all snapshots have same symbol (implicitly via set)

        return len(issues) == 0, issues

    def get_manifest(self, set_id: str) -> Optional[dict]:
        """Get the manifest for a snapshot set."""
        if set_id not in self._sets:
            return None
        return self._sets[set_id].get_manifest()

    def compare_sets(self, set_id_a: str, set_id_b: str) -> dict:
        """
        Compare two snapshot sets for equivalence.

        Returns dict with timeframe-by-timeframe comparison.
        """
        result = {"are_equivalent": True, "timeframes": {}}

        set_a = self._sets.get(set_id_a)
        set_b = self._sets.get(set_id_b)

        if not set_a or not set_b:
            result["are_equivalent"] = False
            result["error"] = "One or both sets not found"
            return result

        for tf in ("weekly", "daily", "intraday_15min"):
            snap_a = set_a.snapshots.get(tf)
            snap_b = set_b.snapshots.get(tf)

            if snap_a and snap_b:
                match = snap_a.data_hash == snap_b.data_hash
                result["timeframes"][tf] = {
                    "match": match,
                    "hash_a": snap_a.data_hash[:16] + "...",
                    "hash_b": snap_b.data_hash[:16] + "...",
                    "candles_a": snap_a.candle_count,
                    "candles_b": snap_b.candle_count,
                }
                if not match:
                    result["are_equivalent"] = False
            elif snap_a or snap_b:
                result["timeframes"][tf] = {
                    "match": False,
                    "reason": "Missing in one set",
                }
                result["are_equivalent"] = False

        return result

    def to_dict(self, set_id: str) -> dict:
        """Serialize a snapshot set for Postgres storage."""
        if set_id not in self._sets:
            return {}

        ss = self._sets[set_id]
        data = {
            "set_id": ss.set_id,
            "symbol": ss.symbol,
            "source": ss.source,
            "adjusted_for_splits": ss.adjusted_for_splits,
            "adjusted_for_dividends": ss.adjusted_for_dividends,
            "notes": ss.notes,
            "created_at": ss.created_at.isoformat(),
            "snapshots": {},
        }

        for tf, snap in ss.snapshots.items():
            data["snapshots"][tf] = {
                "snapshot_id": snap.snapshot_id,
                "set_id": snap.set_id,
                "timeframe": snap.timeframe,
                "period_start": str(snap.period_start),
                "period_end": str(snap.period_end),
                "candle_count": snap.candle_count,
                "first_candle": snap.first_candle.isoformat(),
                "last_candle": snap.last_candle.isoformat(),
                "data_hash": snap.data_hash,
                "yfinance_period": snap.yfinance_period,
                "created_at": snap.created_at.isoformat(),
            }

        return data
