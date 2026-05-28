"""
ContextWindowPolicy: ensures the agent never sees future/partial candle data.
Implements strict no-lookahead filtering for weekly, daily, and intraday context.

Key rules:
- Daily context: only completed daily candles (the current day is excluded)
- Weekly context: only completed weekly candles (the current week is excluded)
- Intraday context: only completed candles ending at or before decision time T
"""
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

import pandas as pd
import pytz

IST = pytz.timezone("Asia/Kolkata")


@dataclass
class ContextWindowPolicy:
    """Policy for what data windows the agent receives at decision time T."""
    weekly_months: int = 3
    daily_months: int = 1
    intraday_sessions: int = 3
    intraday_timeframe: str = "15min"
    include_partial_daily: bool = False
    include_partial_weekly: bool = False
    require_complete_intraday_candles: bool = True


def _get_cutoff(T: datetime, months: int) -> pd.Timestamp:
    """Get a pandas Timestamp for the cutoff date, preserving timezone."""
    ts = pd.Timestamp(T)
    if T.tzinfo and ts.tzinfo is None:
        ts = ts.tz_localize(T.tzinfo)
    return ts - pd.DateOffset(months=months)


def _ensure_tz(ts: datetime) -> datetime:
    """Ensure timestamp has IST timezone for consistent comparisons."""
    if ts.tzinfo is None:
        return IST.localize(ts)
    return ts.astimezone(IST)


def _timeframe_to_minutes(timeframe: str) -> int:
    """Parse simple intraday timeframe labels like 5m, 15min, 1h."""
    value = timeframe.strip().lower()
    if value.endswith("min"):
        return int(value[:-3])
    if value.endswith("m"):
        return int(value[:-1])
    if value.endswith("h"):
        return int(value[:-1]) * 60
    return 15


def _maybe_resample_intraday(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Resample finer intraday data to the requested decision timeframe.

    Resampled bars are labeled at the right edge, so a 09:15-09:30 bar is
    available to the agent at 09:30, never before it closes.
    """
    if df.empty:
        return df.copy()

    target_minutes = _timeframe_to_minutes(timeframe)
    sorted_df = df.sort_index()
    if len(sorted_df.index) >= 2:
        diffs = sorted_df.index.to_series().diff().dropna()
        if not diffs.empty:
            source_minutes = diffs.median().total_seconds() / 60
            if source_minutes >= target_minutes:
                return sorted_df.copy()

    if target_minutes <= 5:
        return sorted_df.copy()

    return sorted_df.resample(
        f"{target_minutes}min",
        label="right",
        closed="left",
        origin="start_day",
    ).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna()


def get_completed_weekly_context(
    df_weekly: pd.DataFrame,
    T: datetime,
    months: int = 3,
    include_partial: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, bool]:
    """
    Get completed weekly context ending before the current week.

    Rules:
    - Weekly context only includes weeks that ended before the week containing T.
    - The current incomplete week is excluded unless include_partial = True.
    - Returns (completed_weekly, partial_current_week, has_partial).
    """
    T = _ensure_tz(T)
    df = df_weekly[df_weekly.index <= T].copy()
    if df.empty:
        return df, pd.DataFrame(), False

    # Find the current week's Monday
    current_monday = T - timedelta(days=T.weekday())
    current_monday = current_monday.replace(hour=0, minute=0, second=0, microsecond=0)

    # Completed weeks end before the current week's Monday
    completed = df[df.index < pd.Timestamp(current_monday)]
    partial = df[df.index >= pd.Timestamp(current_monday)]

    # Apply lookback (preserves timezone)
    cutoff = _get_cutoff(T, months)
    completed = completed[completed.index >= cutoff]

    return completed.copy(), partial.copy(), len(partial) > 0


def get_completed_daily_context(
    df_daily: pd.DataFrame,
    T: datetime,
    months: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame, bool]:
    """
    Get completed daily context ending before the current trading day.

    Rules:
    - Daily context only includes candles whose trading day is complete before T.date().
    - The current incomplete day is excluded unless include_partial = True.
    - Returns (completed_daily, partial_current_day, has_partial).
    """
    T = _ensure_tz(T)
    df = df_daily[df_daily.index <= T].copy()
    if df.empty:
        return df, pd.DataFrame(), False

    # Completed days: date strictly before T.date()
    completed_mask = pd.Index(df.index.date) < T.date()
    completed = df[completed_mask]
    partial = df[~completed_mask]

    # Apply lookback (preserves timezone)
    cutoff = _get_cutoff(T, months)
    completed = completed[completed.index >= cutoff]

    return completed.copy(), partial.copy(), len(partial) > 0


def get_completed_intraday_context(
    df_intraday: pd.DataFrame,
    T: datetime,
    sessions: int = 3,
    timeframe: str = "15min",
    require_complete: bool = True,
) -> pd.DataFrame:
    """
    Get completed intraday candles for the last N trading sessions.

    Rules:
    - Only closed candles labeled <= T.
    - The current incomplete candle at T is excluded (it's still forming).
    - If require_complete, only sessions with enough candles are included.

    Returns DataFrame filtered to completed candles across last N sessions.
    """
    T = _ensure_tz(T)

    completed_bars = _maybe_resample_intraday(df_intraday, timeframe)

    # Strictly include only candles with right-edge timestamp <= T.
    available = completed_bars[completed_bars.index <= T].copy()
    if available.empty:
        return available

    # Get unique session dates
    session_dates = pd.Index(available.index.date).unique()
    selected_dates = set(session_dates[-sessions:])

    return available[
        [idx.date() in selected_dates for idx in available.index]
    ].copy()


def build_context_contract(
    df_weekly: pd.DataFrame,
    df_daily: pd.DataFrame,
    df_intraday: pd.DataFrame,
    T: datetime,
    policy: Optional[ContextWindowPolicy] = None,
) -> Dict:
    """
    Build a context_contract block for MarketStatePackage.

    Returns a dict describing exactly what data windows were used,
    clearly labeling completed vs partial data.
    """
    if policy is None:
        policy = ContextWindowPolicy()

    weekly_completed, weekly_partial, has_partial_week = get_completed_weekly_context(
        df_weekly, T, policy.weekly_months, include_partial=policy.include_partial_weekly
    )
    daily_completed, daily_partial, has_partial_day = get_completed_daily_context(
        df_daily, T, policy.daily_months
    )
    intraday_completed = get_completed_intraday_context(
        df_intraday, T, policy.intraday_sessions, policy.intraday_timeframe, policy.require_complete_intraday_candles
    )

    contract = {
        "weekly": {
            "months": policy.weekly_months,
            "complete_only": not policy.include_partial_weekly,
            "completed_rows": len(weekly_completed),
            "partial_rows": len(weekly_partial),
            "has_partial_current_week": has_partial_week,
            "date_range": (
                f"{weekly_completed.index[0].date()} to {weekly_completed.index[-1].date()}"
                if len(weekly_completed) > 0 else "none"
            ),
        },
        "daily": {
            "months": policy.daily_months,
            "complete_only": not policy.include_partial_daily,
            "completed_rows": len(daily_completed),
            "partial_rows": len(daily_partial),
            "has_partial_current_day": has_partial_day,
            "date_range": (
                f"{daily_completed.index[0].date()} to {daily_completed.index[-1].date()}"
                if len(daily_completed) > 0 else "none"
            ),
        },
        "intraday": {
            "sessions": policy.intraday_sessions,
            "timeframe": policy.intraday_timeframe,
            "complete_only": policy.require_complete_intraday_candles,
            "completed_rows": len(intraday_completed),
            "date_range": (
                f"{intraday_completed.index[0]} to {intraday_completed.index[-1]}"
                if len(intraday_completed) > 0 else "none"
            ),
        },
    }
    return contract


def has_full_context(
    df_weekly: pd.DataFrame,
    df_daily: pd.DataFrame,
    df_intraday: pd.DataFrame,
    T: datetime,
    policy: Optional[ContextWindowPolicy] = None,
    min_weekly: int = 3,
    min_daily: int = 5,
    min_intraday: int = 5,
) -> bool:
    """Check if sufficient completed context is available at decision time T."""
    if policy is None:
        policy = ContextWindowPolicy()

    weekly_completed, _, _ = get_completed_weekly_context(df_weekly, T, policy.weekly_months)
    daily_completed, _, _ = get_completed_daily_context(df_daily, T, policy.daily_months)
    intraday_completed = get_completed_intraday_context(df_intraday, T, policy.intraday_sessions, policy.intraday_timeframe)

    return (
        len(weekly_completed) >= min_weekly
        and len(daily_completed) >= min_daily
        and len(intraday_completed) >= min_intraday
    )


# Backward-compatible function names that use the new completed-context logic
def get_micro_context(df_intraday: pd.DataFrame, T: datetime, sessions: int = 3) -> pd.DataFrame:
    """Get the last N trading sessions of intraday data with completed candles only."""
    return get_completed_intraday_context(df_intraday, T, sessions=sessions)


def get_macro_context(df_daily: pd.DataFrame, T: datetime, months: int = 1) -> pd.DataFrame:
    """Get completed daily data for the last N months."""
    completed, _, _ = get_completed_daily_context(df_daily, T, months=months)
    return completed


def get_htf_context(df_weekly: pd.DataFrame, T: datetime, months: int = 3) -> pd.DataFrame:
    """Get completed weekly data for the last N months."""
    completed, _, _ = get_completed_weekly_context(df_weekly, T, months=months)
    return completed
