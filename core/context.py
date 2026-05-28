"""
Market state package builder. Creates the compact context the agent receives
at each decision point T. Strictly filters all data to <= T.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

from config import config
from core.summarizer import (
    compute_all_indicators,
    detect_swings,
    find_levels,
    detect_pattern,
    summarize_trend,
    price_location,
    estimate_risk,
)
from core.context_window import (
    ContextWindowPolicy,
    build_context_contract,
    get_completed_daily_context,
    get_completed_intraday_context,
    get_completed_weekly_context,
)


def filter_to_t(df: pd.DataFrame, T: datetime) -> pd.DataFrame:
    """Filter DataFrame to only include rows with timestamp <= T."""
    return df[df.index <= T].copy()


def get_micro_context(df_intraday: pd.DataFrame, T: datetime) -> pd.DataFrame:
    """
    Get the last configured trading sessions of completed decision-timeframe data.
    """
    return get_completed_intraday_context(
        df_intraday,
        T,
        sessions=config.MICRO_DAYS,
        timeframe=config.DECISION_INTERVAL,
    )


def has_full_micro_context(df_intraday: pd.DataFrame, T: datetime) -> bool:
    """Return whether T has the configured number of prior/current sessions."""
    available = df_intraday[df_intraday.index <= T]
    if available.empty:
        return False
    session_dates = pd.Index(available.index.date).unique()
    if len(session_dates) < config.MICRO_DAYS:
        return False
    return len(get_micro_context(df_intraday, T)) >= config.MIN_CANDLES_INTRADAY


def get_macro_context(df_daily: pd.DataFrame, T: datetime) -> pd.DataFrame:
    """
    Get completed daily candles only; exclude the current intraday day.
    """
    completed, _, _ = get_completed_daily_context(df_daily, T, months=config.MACRO_MONTHS)
    return completed


def get_htf_context(df_weekly: pd.DataFrame, T: datetime) -> pd.DataFrame:
    """
    Get completed weekly candles only; exclude the week containing T.
    """
    completed, _, _ = get_completed_weekly_context(df_weekly, T, months=config.HTF_MONTHS)
    return completed


def build_market_state_package(
    T: datetime,
    data_5m: pd.DataFrame,
    data_daily: pd.DataFrame,
    data_weekly: pd.DataFrame,
    chart_paths: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Build the complete MarketStatePackage for decision time T.

    Provides compact summaries plus enough raw data for the agent to reason.
    Leakage rule: ALL data slices are filtered to timestamp <= T.
    """
    # ---- Context slices (all filtered to completed candles available at T) ----
    micro = get_micro_context(data_5m, T)
    macro = get_macro_context(data_daily, T)
    htf = get_htf_context(data_weekly, T)
    context_contract = build_context_contract(
        data_weekly,
        data_daily,
        data_5m,
        T,
        ContextWindowPolicy(
            weekly_months=config.HTF_MONTHS,
            daily_months=config.MACRO_MONTHS,
            intraday_sessions=config.MICRO_DAYS,
            intraday_timeframe=config.DECISION_INTERVAL,
        ),
    )

    timeframe_label = config.INTRADAY_TIMEFRAME_LABEL

    # ---- Latest intraday candle ----
    latest_candle = data_5m.loc[T] if T in data_5m.index else None
    if latest_candle is not None:
        latest_summary = {
            "open": float(latest_candle["open"]),
            "high": float(latest_candle["high"]),
            "low": float(latest_candle["low"]),
            "close": float(latest_candle["close"]),
            "volume": int(latest_candle["volume"]),
        }
        current_price = float(latest_candle["close"])
    else:
        latest_summary = None
        current_price = float(micro["close"].iloc[-1]) if len(micro) > 0 else 0

    # ---- Indicators (on micro context only) ----
    indicators = compute_all_indicators(micro) if len(micro) >= 14 else {}

    # ---- Swings (on micro context) ----
    swings = detect_swings(micro) if len(micro) >= 10 else {"swing_highs": [], "swing_lows": []}

    # ---- Support/Resistance levels ----
    levels = find_levels(micro, current_price) if len(micro) >= 10 else {}

    # ---- Price-action pattern ----
    pattern = detect_pattern(micro) if len(micro) >= 10 else "insufficient_data"

    # ---- Trend summaries ----
    trend_intraday = summarize_trend(micro, timeframe_label) if len(micro) >= config.MIN_CANDLES_INTRADAY else "insufficient data"
    trend_daily = summarize_trend(macro, "daily") if len(macro) >= config.MIN_CANDLES_DAILY else "insufficient data"
    trend_weekly = summarize_trend(htf, "weekly") if len(htf) >= config.MIN_CANDLES_WEEKLY else "insufficient data"

    # ---- Price location ----
    location = price_location(micro) if len(micro) >= 10 else "no_clear_area"

    # ---- Recent 5m candles (raw, for trigger context) ----
    recent_5m = micro.iloc[-20:].copy()
    recent_candles = []
    for idx, row in recent_5m.iterrows():
        recent_candles.append({
            "time": str(idx),
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": int(row["volume"]),
        })

    # ---- Daily candle summaries ----
    daily_summaries = []
    for idx, row in macro.iloc[-10:].iterrows():
        daily_summaries.append({
            "date": str(idx.date()),
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": int(row["volume"]),
            "range": float(round(row["high"] - row["low"], 2)),
        })

    # ---- Weekly candle summaries ----
    weekly_summaries = []
    for idx, row in htf.iterrows():
        weekly_summaries.append({
            "week": str(idx.date()),
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": int(row["volume"]),
            "range": float(round(row["high"] - row["low"], 2)),
        })

    # ---- Compile the package ----
    package = {
        "instrument": config.INSTRUMENT_NAME,
        "symbol": config.SYMBOL,
        "decision_time": str(T),
        "intraday_timeframe": timeframe_label,
        "context_windows": {
            "weekly": f"last {config.HTF_MONTHS} months of completed weekly candles; current week excluded",
            "daily": f"last {config.MACRO_MONTHS} month of completed daily candles; current day excluded",
            "intraday": f"last {config.MICRO_DAYS} trading sessions of completed {timeframe_label} candles ending at decision time",
        },
        "context_contract": context_contract,
        "context_row_counts": {
            "intraday": len(micro),
            "daily": len(macro),
            "weekly": len(htf),
        },
        "current_price": float(round(current_price, 2)),
        "latest_candle": latest_summary,
        "indicators": indicators,
        "swings": swings,
        "levels": levels,
        "pattern": pattern,
        "price_location": location,
        "trend_5m": trend_intraday,
        "trend_intraday": trend_intraday,
        "trend_daily": trend_daily,
        "trend_weekly": trend_weekly,
        "recent_5m_candles": recent_candles,
        "recent_intraday_candles": recent_candles,
        "daily_summaries": daily_summaries,
        "weekly_summaries": weekly_summaries,
        "chart_paths": chart_paths or {},
        "session_start": None,  # Filled by caller
        "session_end": None,    # Filled by caller
    }

    return package


def format_market_state_for_prompt(package: Dict[str, Any]) -> str:
    """
    Format the MarketStatePackage as a compact text prompt for the LLM.
    """
    lines = []
    lines.append("=" * 60)
    lines.append(f"MARKET STATE at {package['decision_time']}")
    lines.append(f"Instrument: {package['instrument']} ({package['symbol']})")
    lines.append(f"Current Price: {package['current_price']}")
    windows = package.get("context_windows", {})
    counts = package.get("context_row_counts", {})
    if windows:
        lines.append("Context windows:")
        lines.append(f"  Weekly: {windows.get('weekly')} ({counts.get('weekly', 0)} candles)")
        lines.append(f"  Daily: {windows.get('daily')} ({counts.get('daily', 0)} candles)")
        lines.append(f"  Intraday: {windows.get('intraday')} ({counts.get('intraday', 0)} candles)")
    contract = package.get("context_contract")
    if contract:
        lines.append("Context contract:")
        lines.append(
            f"  Weekly complete_only={contract['weekly']['complete_only']} "
            f"rows={contract['weekly']['completed_rows']} partial_rows={contract['weekly']['partial_rows']}"
        )
        lines.append(
            f"  Daily complete_only={contract['daily']['complete_only']} "
            f"rows={contract['daily']['completed_rows']} partial_rows={contract['daily']['partial_rows']}"
        )
        lines.append(
            f"  Intraday complete_only={contract['intraday']['complete_only']} "
            f"sessions={contract['intraday']['sessions']} rows={contract['intraday']['completed_rows']}"
        )
    lines.append("=" * 60)

    # Latest candle
    if package["latest_candle"]:
        c = package["latest_candle"]
        timeframe_label = package.get("intraday_timeframe", config.INTRADAY_TIMEFRAME_LABEL)
        lines.append(f"\nLATEST {timeframe_label} CANDLE: O={c['open']:.2f} H={c['high']:.2f} L={c['low']:.2f} C={c['close']:.2f} V={c['volume']}")
        # Candle color
        color = "bullish" if c["close"] >= c["open"] else "bearish"
        body = abs(c["close"] - c["open"])
        range_5m = c["high"] - c["low"]
        lines.append(f"  Color: {color}, Body: {body:.2f}, Range: {range_5m:.2f}")

    # Trend summaries
    lines.append(f"\nTRENDS:")
    timeframe_label = package.get("intraday_timeframe", config.INTRADAY_TIMEFRAME_LABEL)
    lines.append(f"  {timeframe_label} (micro): {package['trend_5m']}")
    lines.append(f"  Daily (macro): {package['trend_daily']}")
    lines.append(f"  Weekly (HTF): {package['trend_weekly']}")

    # Pattern
    lines.append(f"\nPRICE-ACTION PATTERN: {package['pattern']}")
    lines.append(f"PRICE LOCATION: {package['price_location']}")

    # Indicators
    ind = package["indicators"]
    if ind:
        lines.append(f"\nINDICATORS:")
        if ind.get("rsi_14") is not None:
            rsi_status = "overbought (>70)" if ind["rsi_14"] > 70 else ("oversold (<30)" if ind["rsi_14"] < 30 else "neutral")
            lines.append(f"  RSI(14): {ind['rsi_14']} ({rsi_status})")
        if ind.get("atr_14") is not None:
            lines.append(f"  ATR(14): {ind['atr_14']}")
        if ind.get("ma_20_slope") is not None:
            slope_desc = "rising" if ind["ma_20_slope"] > 0 else "falling"
            lines.append(f"  MA20 slope: {ind['ma_20_slope']:.4f} ({slope_desc})")
        if ind.get("ma_50_slope") is not None:
            slope_desc = "rising" if ind["ma_50_slope"] > 0 else "falling"
            lines.append(f"  MA50 slope: {ind['ma_50_slope']:.4f} ({slope_desc})")
        if ind.get("momentum_10_pct") is not None:
            lines.append(f"  Momentum (10p): {ind['momentum_10_pct']:.2f}%")
        if ind.get("volume_ratio") is not None:
            vol_desc = "above avg" if ind["volume_ratio"] > 1.2 else ("below avg" if ind["volume_ratio"] < 0.8 else "average")
            lines.append(f"  Volume vs avg: {ind['volume_ratio']:.2f}x ({vol_desc})")

    # Levels
    lvls = package["levels"]
    if lvls:
        lines.append(f"\nSUPPORT/RESISTANCE LEVELS:")
        if lvls.get("nearest_support"):
            lines.append(f"  Nearest Support: {lvls['nearest_support']}")
        if lvls.get("nearest_resistance"):
            lines.append(f"  Nearest Resistance: {lvls['nearest_resistance']}")
        if lvls.get("supports"):
            lines.append(f"  Supports: {', '.join(str(s) for s in lvls['supports'][-3:])}")
        if lvls.get("resistances"):
            lines.append(f"  Resistances: {', '.join(str(r) for r in lvls['resistances'][:3])}")

    # Swings
    sw = package["swings"]
    if sw.get("swing_highs") or sw.get("swing_lows"):
        lines.append(f"\nRECENT SWINGS:")
        if sw.get("most_recent_high"):
            lines.append(f"  Most Recent Swing High: {sw['most_recent_high']}")
        if sw.get("most_recent_low"):
            lines.append(f"  Most Recent Swing Low: {sw['most_recent_low']}")

    # Recent intraday candles (show last 8 as raw data)
    recent = package.get("recent_intraday_candles") or package["recent_5m_candles"]
    if recent:
        lines.append(f"\nLAST 8 {timeframe_label} CANDLES (most recent first):")
        lines.append(f"  {'Time':<30s} {'Open':>8s} {'High':>8s} {'Low':>8s} {'Close':>8s} {'Volume':>10s}")
        for c in reversed(recent[-8:]):
            t_short = c["time"].split(" ")[-1] if " " in c["time"] else c["time"][-8:]
            lines.append(f"  {t_short:<30s} {c['open']:>8.2f} {c['high']:>8.2f} {c['low']:>8.2f} {c['close']:>8.2f} {c['volume']:>10d}")

    # Daily summaries (last 5)
    daily = package["daily_summaries"]
    if daily:
        lines.append(f"\nLAST 5 DAILY CANDLES:")
        lines.append(f"  {'Date':<12s} {'Open':>8s} {'High':>8s} {'Low':>8s} {'Close':>8s} {'Range':>8s}")
        for d in daily[-5:]:
            lines.append(f"  {d['date']:<12s} {d['open']:>8.2f} {d['high']:>8.2f} {d['low']:>8.2f} {d['close']:>8.2f} {d['range']:>8.2f}")

    # Weekly summaries (last 3)
    weekly = package["weekly_summaries"]
    if weekly:
        lines.append(f"\nLAST 3 WEEKLY CANDLES:")
        lines.append(f"  {'Week':<12s} {'Open':>8s} {'High':>8s} {'Low':>8s} {'Close':>8s} {'Range':>8s}")
        for w in weekly[-3:]:
            lines.append(f"  {w['week']:<12s} {w['open']:>8.2f} {w['high']:>8.2f} {w['low']:>8.2f} {w['close']:>8.2f} {w['range']:>8.2f}")

    # Visual context pack
    chart_paths = package.get("chart_paths") or {}
    if chart_paths:
        lines.append("\nVISUAL CONTEXT PACK:")
        lines.append("  These charts were generated only from data available at decision time T.")
        chart_descriptions = {
            "context_dashboard": "combined price-action, volume, indicator, and higher-timeframe view",
            "decision_zoom_chart": f"recent {timeframe_label} candles around T for trigger and immediate levels",
            "micro_5m_chart": f"last 3 trading sessions of {timeframe_label} candles for intraday structure",
            "volume_chart": "volume bars and relative participation context",
            "daily_context_chart": "daily candles for macro bias and nearby zones",
            "weekly_context_chart": "weekly candles for broad regime context",
            "indicator_panel": "compact RSI, moving-average, ATR, and momentum panel",
        }
        for key, path in chart_paths.items():
            description = chart_descriptions.get(key, "generated decision-time chart")
            lines.append(f"  - {key}: {description}")
            lines.append(f"    path: {path}")

    lines.append("\n" + "=" * 60)
    lines.append("DECISION REQUIRED: Based on DART framework (Direction, Area, Risk, Trigger).")
    lines.append("Output BUY, SELL, or HOLD with structured reasoning.")
    lines.append("=" * 60)

    return "\n".join(lines)
