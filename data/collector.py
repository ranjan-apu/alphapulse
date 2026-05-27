"""
Data collector: fetches OHLCV data from Yahoo Finance.
Provides 5-minute, daily, and weekly data for the configured instrument.
"""
import json
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import yfinance as yf

from config import config


def _standardize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize Yahoo Finance OHLCV columns and index."""
    if df.empty:
        return df

    df = df.rename(columns={
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Volume": "volume",
    })
    df = df[["open", "high", "low", "close", "volume"]]

    if df.index.tz is not None:
        df.index = df.index.tz_convert("Asia/Kolkata")
    else:
        df.index = df.index.tz_localize("Asia/Kolkata")

    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df.dropna()


def _cache_paths(symbol: str) -> dict:
    return {
        "5m": config.RAW_DATA_DIR / f"{symbol}_5m.csv",
        "daily": config.RAW_DATA_DIR / f"{symbol}_daily.csv",
        "weekly": config.RAW_DATA_DIR / f"{symbol}_weekly.csv",
        "metadata": config.RAW_DATA_DIR / f"{symbol}_metadata.json",
    }


def _merge_ohlcv(existing: pd.DataFrame, fresh: pd.DataFrame) -> pd.DataFrame:
    """Append new OHLCV rows to existing cache without rewriting history."""
    if existing is None or existing.empty:
        return fresh.sort_index().copy()
    if fresh is None or fresh.empty:
        return existing.sort_index().copy()

    merged = pd.concat([existing, fresh])
    merged = merged.sort_index()
    # Keep the newest fetched copy on overlapping timestamps while preserving
    # older non-overlapping cache history.
    merged = merged[~merged.index.duplicated(keep="last")]
    return merged.dropna()


def fetch_5m_data(symbol: str, days: int = None) -> pd.DataFrame:
    """
    Fetch 5-minute OHLCV data from Yahoo Finance.

    Yahoo generally exposes only recent 5m candles. The harness requests the
    configured window and records the actual returned range in cache metadata.
    """
    days = days or config.FETCH_5M_DAYS
    ticker = yf.Ticker(symbol)
    period = config.FETCH_5M_PERIOD
    print(f"  Fetching {period} of 5m data for {symbol}...")
    df = ticker.history(period=period, interval="5m")
    if df.empty and period != f"{days}d":
        fallback_period = f"{days}d"
        print(f"  No 5m data for {period}; falling back to {fallback_period}...")
        df = ticker.history(period=fallback_period, interval="5m")
    if df.empty:
        raise ValueError(f"No 5m data returned for {symbol}")
    return _standardize_ohlcv(df)


def fetch_daily_data(symbol: str, months: int = None) -> pd.DataFrame:
    """Fetch daily OHLCV data."""
    months = months or config.FETCH_DAILY_MONTHS
    ticker = yf.Ticker(symbol)
    period = f"{months}mo"
    print(f"  Fetching {period} of daily data for {symbol}...")

    df = ticker.history(period=period, interval="1d")
    if df.empty:
        raise ValueError(f"No daily data returned for {symbol}")
    return _standardize_ohlcv(df)


def fetch_weekly_data(symbol: str, months: int = None) -> pd.DataFrame:
    """Fetch weekly OHLCV data."""
    months = months or config.FETCH_WEEKLY_MONTHS
    ticker = yf.Ticker(symbol)
    period = f"{months}mo"
    print(f"  Fetching {period} of weekly data for {symbol}...")

    df = ticker.history(period=period, interval="1wk")
    if df.empty:
        raise ValueError(f"No weekly data returned for {symbol}")
    return _standardize_ohlcv(df)


def resample_to_timeframe(df_5m: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Resample 5-minute source data into a closed higher-timeframe series."""
    rule_map = {
        "5m": "5min", "5min": "5min", "5T": "5min",
        "15min": "15min", "15T": "15min",
        "30min": "30min", "30T": "30min",
        "1h": "1h", "60min": "1h", "1H": "1h",
        "4h": "4h",
    }
    rule = rule_map.get(timeframe, timeframe)

    if rule == "5min":
        return df_5m.sort_index().copy()

    resampled = df_5m.resample(
        rule,
        label="right",
        closed="left",
        origin="start_day",
    ).agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum",
    }).dropna()
    return resampled


def collect_all_data(symbol: str = None, output_dir: Path = None) -> dict:
    """Collect all required data: 5m, daily, weekly. Saves CSVs."""
    symbol = symbol or config.SYMBOL
    output_dir = output_dir or config.RAW_DATA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = _cache_paths(symbol)

    print(f"\n{'='*60}")
    print(f"Collecting data for {symbol} ({config.INSTRUMENT_NAME})")
    print(f"{'='*60}")

    fetched_at = datetime.now().astimezone().isoformat()
    existing = load_cached_data(symbol)
    fresh_5m = fetch_5m_data(symbol)
    fresh_daily = fetch_daily_data(symbol)
    fresh_weekly = fetch_weekly_data(symbol)

    df_5m = _merge_ohlcv(existing.get("5m"), fresh_5m)
    df_daily = _merge_ohlcv(existing.get("daily"), fresh_daily)
    df_weekly = _merge_ohlcv(existing.get("weekly"), fresh_weekly)

    df_5m.to_csv(paths["5m"])
    df_daily.to_csv(paths["daily"])
    df_weekly.to_csv(paths["weekly"])

    metadata = {
        "symbol": symbol,
        "instrument": config.INSTRUMENT_NAME,
        "fetched_at": fetched_at,
        "fetch_months": config.FETCH_MONTHS,
        "fetch_5m_days_requested": config.FETCH_5M_DAYS,
        "ranges": {
            "5m": {"start": str(df_5m.index[0]), "end": str(df_5m.index[-1]), "rows": len(df_5m)},
            "daily": {"start": str(df_daily.index[0]), "end": str(df_daily.index[-1]), "rows": len(df_daily)},
            "weekly": {"start": str(df_weekly.index[0]), "end": str(df_weekly.index[-1]), "rows": len(df_weekly)},
        },
        "last_fetch_ranges": {
            "5m": {"start": str(fresh_5m.index[0]), "end": str(fresh_5m.index[-1]), "rows": len(fresh_5m)},
            "daily": {"start": str(fresh_daily.index[0]), "end": str(fresh_daily.index[-1]), "rows": len(fresh_daily)},
            "weekly": {"start": str(fresh_weekly.index[0]), "end": str(fresh_weekly.index[-1]), "rows": len(fresh_weekly)},
        },
        "note": "Refreshes merge fresh rows into existing local cache and keep prior non-overlapping history. Yahoo Finance may not provide six full months of 5m candles in one fetch.",
    }
    paths["metadata"].write_text(json.dumps(metadata, indent=2, default=str))

    print(f"\n  Data collected:")
    print(f"    5m:    {len(df_5m)} candles from {df_5m.index[0]} to {df_5m.index[-1]}")
    print(f"    Daily: {len(df_daily)} candles from {df_daily.index[0]} to {df_daily.index[-1]}")
    print(f"    Weekly:{len(df_weekly)} candles from {df_weekly.index[0]} to {df_weekly.index[-1]}")
    print(f"  Saved to {output_dir}")

    return {"5m": df_5m, "daily": df_daily, "weekly": df_weekly}


def load_cached_data(symbol: str = None) -> dict:
    """Load previously cached data from CSV files."""
    symbol = symbol or config.SYMBOL
    data_dir = config.RAW_DATA_DIR
    result = {}

    paths = _cache_paths(symbol)
    for tf in ["5m", "daily", "weekly"]:
        path = paths[tf]
        if path.exists():
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            if df.index.tz is None:
                df.index = df.index.tz_localize("Asia/Kolkata")
            else:
                df.index = df.index.tz_convert("Asia/Kolkata")
            result[tf] = df
    if paths["metadata"].exists():
        result["metadata"] = json.loads(paths["metadata"].read_text())

    return result


def cache_is_fresh(cached: dict, max_age_hours: int = None) -> bool:
    """Return whether local cache metadata is recent enough for startup use."""
    max_age_hours = max_age_hours or config.CACHE_MAX_AGE_HOURS
    metadata = cached.get("metadata") or {}
    fetched_at = metadata.get("fetched_at")
    if not fetched_at:
        return False

    try:
        fetched_ts = pd.Timestamp(fetched_at)
        if fetched_ts.tzinfo is None:
            fetched_ts = fetched_ts.tz_localize("Asia/Kolkata")
        now = pd.Timestamp.now(tz=fetched_ts.tz)
        return now - fetched_ts <= pd.Timedelta(hours=max_age_hours)
    except Exception:
        return False


def get_historical_slice(
    df_full: pd.DataFrame,
    start_date: datetime,
    end_date: datetime,
) -> pd.DataFrame:
    """
    Extract a historical slice from a DataFrame.
    Used by the get_historical_data tool.
    Respects the time boundary (end_date acts as the decision time T).
    """
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)
    if start_date.tzinfo is None:
        start_date = start_date.tz_localize("Asia/Kolkata")
    if end_date.tzinfo is None:
        end_date = end_date.tz_localize("Asia/Kolkata")

    mask = (df_full.index >= start_date) & (df_full.index <= end_date)
    return df_full[mask].copy()


if __name__ == "__main__":
    config.ensure_dirs()
    data = collect_all_data()
