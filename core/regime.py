"""
Market regime detection and session type classification.

Regime types:
- trend: directional movement with HH/HL or LH/LL
- range: sideways movement within defined boundaries
- volatile: high volatility, wide ranges
- compression: low volatility, narrowing ranges

Session types:
- trend_day: persistent directional move, VWAP sloping
- range_day: sideways within boundaries
- reversal_day: sharp turn from initial direction
- inside_day: entirely within prior day's range
- opening_drive: strong initial thrust in first hour
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class RegimeResult:
    """Market regime analysis."""
    regime: str          # 'trend', 'range', 'volatile', 'compression', 'unclear'
    session_type: str    # 'trend_day', 'range_day', 'reversal_day', 'inside_day', 'opening_drive', 'unclear'
    volatility_bucket: str  # 'high', 'medium', 'low'
    confidence: float = 0.0
    notes: str = ""


def detect_market_regime(
    df_intraday: pd.DataFrame,
    atr: Optional[float] = None,
) -> RegimeResult:
    """
    Detect the current market regime and session type.

    Uses:
    - ATR and range expansion/contraction for volatility
    - Trend strength measured by MA slope and ADX-like directional movement
    - Range detection using recent high-low boundaries
    """
    if df_intraday.empty or len(df_intraday) < 10:
        return RegimeResult(regime="unclear", session_type="unclear", volatility_bucket="medium")

    df = df_intraday.copy()
    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values

    # ---- Volatility analysis ----
    if atr is None:
        # Simple ATR
        tr = np.maximum(
            highs[1:] - lows[1:],
            np.maximum(
                abs(highs[1:] - closes[:-1]),
                abs(lows[1:] - closes[:-1])
            )
        )
        atr = float(np.mean(tr[-14:])) if len(tr) >= 14 else float(np.mean(tr))

    # Range analysis
    recent_range = float(np.max(highs[-10:]) - np.min(lows[-10:]))
    range_pct = (recent_range / closes[-1] * 100) if closes[-1] > 0 else 0

    # Volatility bucket
    if atr and closes[-1] > 0:
        atr_pct = (atr / closes[-1]) * 100
        if atr_pct > 2.0:
            volatility_bucket = "high"
        elif atr_pct < 0.5:
            volatility_bucket = "low"
        else:
            volatility_bucket = "medium"
    else:
        volatility_bucket = "medium"

    # ---- Compression detection ----
    # Check if ranges are narrowing
    if len(df) >= 15:
        ranges_1 = float(np.max(highs[-5:]) - np.min(lows[-5:]))
        ranges_2 = float(np.max(highs[-10:-5]) - np.min(lows[-10:-5]))
        ranges_3 = float(np.max(highs[-15:-10]) - np.min(lows[-15:-10]))
        compressing = ranges_1 < ranges_2 < ranges_3 if ranges_3 > 0 else False
    else:
        compressing = False

    # ---- Trend detection ----
    # Simple moving average slope
    ma_period = min(10, len(closes) // 2)
    if len(closes) >= ma_period:
        ma = np.convolve(closes, np.ones(ma_period) / ma_period, mode="valid")
        if len(ma) >= 2:
            slope = (ma[-1] - ma[0]) / ma[0] * 100 if ma[0] > 0 else 0
        else:
            slope = 0
    else:
        slope = 0

    # ---- High/low analysis ----
    first_price = closes[0] if len(closes) > 0 else closes[-1]
    last_price = closes[-1]
    total_change = abs(last_price - first_price) / first_price * 100 if first_price > 0 else 0

    # Maximum excursion vs net change
    max_up = float(np.max(highs))
    min_down = float(np.min(lows))
    up_move = (max_up - first_price) / first_price * 100 if first_price > 0 else 0
    down_move = (first_price - min_down) / first_price * 100 if first_price > 0 else 0

    # ---- Classify regime ----
    if volatility_bucket == "high" and range_pct > 3.0:
        regime = "volatile"
    elif compressing and atr_pct if atr else 0 < 1.0:
        regime = "compression"
    elif abs(slope) > 0.5 and total_change > 0.5:
        regime = "trend"
    elif range_pct < 2.0:
        regime = "range"
    else:
        regime = "unclear"

    # ---- Session type ----
    session_type = _classify_session_type(
        df, first_price, last_price, up_move, down_move, slope
    )

    confidence = min(abs(slope) / 2.0, 1.0) if abs(slope) > 0 else 0.3
    if regime == "range":
        confidence = min(range_pct / 3.0, 0.7)

    return RegimeResult(
        regime=regime,
        session_type=session_type,
        volatility_bucket=volatility_bucket,
        confidence=round(confidence, 2),
        notes=f"slope={slope:.2f}%, range={range_pct:.2f}%, atr_pct={atr_pct:.2f}%",
    )


def _classify_session_type(
    df: pd.DataFrame,
    first_price: float,
    last_price: float,
    up_move: float,
    down_move: float,
    slope: float,
) -> str:
    """Classify the session type based on price behavior."""
    if len(df) < 5:
        return "unclear"

    # Opening drive: strong first-hour directional move
    first_hour_candles = min(4, len(df))  # ~1 hour of 15m candles
    if first_hour_candles >= 2:
        first_hour_change = abs(
            df["close"].iloc[first_hour_candles - 1] - df["open"].iloc[0]
        ) / df["open"].iloc[0] * 100
        total_change = abs(last_price - first_price) / first_price * 100
        if first_hour_change > 0.5 and first_hour_change > total_change * 0.6:
            return "opening_drive"

    # Inside day: entire range within prior day (requires daily data, approximate here)
    if up_move < 0.5 and down_move < 0.5:
        return "inside_day"

    # Reversal day: large excursion both ways
    if up_move > 0.5 and down_move > 0.5 and abs(up_move - down_move) < 1.0:
        return "reversal_day"

    # Trend day: persistent directional move
    if abs(slope) > 0.3:
        return "trend_day"

    # Range day: sideways movement
    if up_move < 1.5 and down_move < 1.5:
        return "range_day"

    return "unclear"


def classify_time_bucket(decision_time) -> str:
    """Classify the time of day into a bucket."""
    if hasattr(decision_time, 'time'):
        t = decision_time.time()
    elif hasattr(decision_time, 'hour'):
        import datetime
        t = datetime.time(decision_time.hour, decision_time.minute)
    else:
        return "unknown"

    hour = t.hour
    if hour < 10:
        return "morning_open"
    elif hour < 12:
        return "morning"
    elif hour < 14:
        return "midday"
    elif hour < 15:
        return "afternoon"
    else:
        return "late_session"


def classify_volatility(indicators: dict) -> str:
    """Classify volatility bucket from indicators."""
    atr = indicators.get("atr_14")
    if atr is None:
        return "medium"

    # This is relative - would need price to compute properly
    return "medium"


def regime_result_to_dict(result: RegimeResult) -> dict:
    """Convert RegimeResult to dict."""
    return {
        "regime": result.regime,
        "session_type": result.session_type,
        "volatility_bucket": result.volatility_bucket,
        "confidence": result.confidence,
        "notes": result.notes,
    }
