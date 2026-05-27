"""
Price-action summarizer: computes indicators, detects patterns,
finds swing points, support/resistance levels, and summarizes trends.

All functions receive data <= decision_time T (no future leakage).
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from config import config


# ---------------------------------------------------------------------------
# Indicator computations
# ---------------------------------------------------------------------------

def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Compute RSI for a series of close prices."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    # Wilder smoothing after initial SMA
    for i in range(period, len(avg_gain)):
        avg_gain.iloc[i] = (avg_gain.iloc[i-1] * (period - 1) + gain.iloc[i]) / period
        avg_loss.iloc[i] = (avg_loss.iloc[i-1] * (period - 1) + loss.iloc[i]) / period

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi[avg_loss == 0] = 100.0
    return rsi


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute Average True Range."""
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=period, min_periods=period).mean()

    # Wilder smoothing
    for i in range(period, len(atr)):
        atr.iloc[i] = (atr.iloc[i-1] * (period - 1) + true_range.iloc[i]) / period

    return atr


def compute_ma_slope(close: pd.Series, period: int = 20) -> Optional[float]:
    """Compute slope of moving average over last few bars."""
    if len(close) < period + 3:
        return None
    ma = close.rolling(window=period).mean().dropna()
    if len(ma) < 3:
        return None
    # Slope over last 3 MA values
    y = ma.iloc[-3:].values
    x = np.array([0, 1, 2])
    slope = np.polyfit(x, y, 1)[0]
    return float(slope)


def compute_momentum(close: pd.Series, lookback: int = 10) -> Optional[float]:
    """Rate of change over lookback periods."""
    if len(close) < lookback:
        return None
    roc = (close.iloc[-1] - close.iloc[-(lookback)]) / close.iloc[-(lookback)] * 100
    return float(roc) if not pd.isna(roc) else None


def compute_volume_change(df: pd.DataFrame, lookback: int = 20) -> Optional[float]:
    """Current volume vs average volume ratio."""
    if len(df) < lookback:
        return None
    avg_vol = df["volume"].iloc[-(lookback+1):-1].mean()
    current_vol = df["volume"].iloc[-1]
    if avg_vol == 0:
        return None
    return float(current_vol / avg_vol)


def compute_all_indicators(df: pd.DataFrame) -> Dict:
    """Compute all basic indicators for a DataFrame."""
    close = df["close"]

    rsi = compute_rsi(close)
    atr = compute_atr(df)
    ma_slope_20 = compute_ma_slope(close, 20)
    ma_slope_50 = compute_ma_slope(close, 50)
    momentum_10 = compute_momentum(close, 10)
    vol_change = compute_volume_change(df)

    latest = {
        "rsi_14": float(round(rsi.iloc[-1], 1)) if not pd.isna(rsi.iloc[-1]) else None,
        "atr_14": float(round(atr.iloc[-1], 2)) if not pd.isna(atr.iloc[-1]) else None,
        "ma_20_slope": float(round(ma_slope_20, 4)) if ma_slope_20 is not None else None,
        "ma_50_slope": float(round(ma_slope_50, 4)) if ma_slope_50 is not None else None,
        "momentum_10_pct": float(round(momentum_10, 2)) if momentum_10 is not None else None,
        "volume_ratio": float(round(vol_change, 2)) if vol_change is not None else None,
        "current_close": float(close.iloc[-1]),
        "current_volume": int(df["volume"].iloc[-1]),
    }

    return latest


# ---------------------------------------------------------------------------
# Swing detection
# ---------------------------------------------------------------------------

def detect_swings(df: pd.DataFrame, lookback: int = 5) -> Dict:
    """
    Detect recent swing highs and lows using local extrema.
    A swing high is a candle whose high is higher than lookback bars on both sides.
    """
    highs = df["high"].values
    lows = df["low"].values
    index = df.index

    swing_highs = []
    swing_lows = []

    for i in range(lookback, len(df) - lookback):
        if highs[i] == max(highs[i-lookback:i+lookback+1]):
            swing_highs.append({
                "time": str(index[i]),
                "price": float(highs[i]),
            })
        if lows[i] == min(lows[i-lookback:i+lookback+1]):
            swing_lows.append({
                "time": str(index[i]),
                "price": float(lows[i]),
            })

    # Return only the most recent swings (last 10)
    return {
        "swing_highs": swing_highs[-10:],
        "swing_lows": swing_lows[-10:],
        "most_recent_high": swing_highs[-1]["price"] if swing_highs else None,
        "most_recent_low": swing_lows[-1]["price"] if swing_lows else None,
    }


# ---------------------------------------------------------------------------
# Support / Resistance detection
# ---------------------------------------------------------------------------

def find_levels(df: pd.DataFrame, current_price: float) -> Dict:
    """
    Find nearby support and resistance levels from swing points and
    prior highs/lows.
    """
    # Use swing high/low clusters as levels
    swings = detect_swings(df)
    all_levels = []

    for sh in swings["swing_highs"]:
        all_levels.append(sh["price"])
    for sl in swings["swing_lows"]:
        all_levels.append(sl["price"])

    # Add prior day high/low if available
    if len(df) >= 75:  # roughly a day's worth
        prior_day = df.iloc[:-75] if len(df) > 75 else df
        if len(prior_day) > 0:
            all_levels.append(float(prior_day["high"].max()))
            all_levels.append(float(prior_day["low"].min()))

    # Round and deduplicate (cluster nearby levels)
    rounded = {}
    for level in all_levels:
        key = round(level, 1)
        if key not in rounded:
            rounded[key] = []
        rounded[key].append(level)

    # Average clustered levels
    unique_levels = sorted([np.mean(v) for v in rounded.values()])

    # Classify as support (below price) or resistance (above price)
    supports = [l for l in unique_levels if l < current_price * 0.999]
    resistances = [l for l in unique_levels if l > current_price * 1.001]

    return {
        "supports": [float(round(s, 2)) for s in supports[-5:]],
        "resistances": [float(round(r, 2)) for r in resistances[:5]],
        "nearest_support": float(round(supports[-1], 2)) if supports else None,
        "nearest_resistance": float(round(resistances[0], 2)) if resistances else None,
        "all_levels": [float(round(l, 2)) for l in unique_levels],
    }


# ---------------------------------------------------------------------------
# Price-action pattern detection
# ---------------------------------------------------------------------------

def detect_pattern(df: pd.DataFrame) -> str:
    """
    Detect basic price-action patterns from recent candles.
    Returns a label string.
    """
    if len(df) < 10:
        return "insufficient_data"

    close = df["close"].values
    high = df["high"].values
    low = df["low"].values

    # Recent price range
    recent_high = float(np.max(high[-10:]))
    recent_low = float(np.min(low[-10:]))
    price_range = recent_high - recent_low
    current = float(close[-1])

    # Consolidation: tight range
    if price_range < np.mean(high[-20:] - low[-20:]) * 0.4 if len(df) >= 20 else price_range < recent_high * 0.005:
        return "consolidation"

    # Breakout: price near top of range and making new highs
    if current > recent_high * 0.995 and close[-1] >= np.max(close[-6:-1]):
        return "breakout"

    # Retest: price near a prior level
    # Check if current price is near any prior swing
    swings = detect_swings(df)
    for sh in swings["swing_highs"]:
        if abs(current - sh["price"]) / sh["price"] < 0.003:
            return "retest"

    # Rejection: long wick
    last_candle_range = high[-1] - low[-1]
    if last_candle_range > 0:
        upper_wick = high[-1] - max(close[-1], df["open"].values[-1])
        lower_wick = min(close[-1], df["open"].values[-1]) - low[-1]
        body = abs(close[-1] - df["open"].values[-1])

        if upper_wick > body * 2 and upper_wick > last_candle_range * 0.6:
            return "rejection (bearish wick)"
        if lower_wick > body * 2 and lower_wick > last_candle_range * 0.6:
            return "rejection (bullish wick)"

    # Pullback: price pulling back in a trend
    if len(close) >= 30:
        ma_20 = pd.Series(close).rolling(20).mean().values
        if not np.isnan(ma_20[-1]):
            # Price near MA in a trending environment
            if abs(current - ma_20[-1]) / ma_20[-1] < 0.01:
                slope = np.polyfit(range(min(10, len(ma_20))), ma_20[-10:], 1)[0]
                if abs(slope) > 0.01:
                    return "pullback"

    return "no_clear_pattern"


# ---------------------------------------------------------------------------
# Trend summary
# ---------------------------------------------------------------------------

def summarize_trend(df: pd.DataFrame, label: str = "") -> str:
    """
    Summarize trend direction for a given timeframe.
    Uses MA slope, recent price action, and structure.
    Relaxed minimums: 5 candles for basic direction, 20 for MA-based analysis.
    """
    if len(df) < config.MIN_CANDLES_5M:
        return "unclear (insufficient data)"

    close = df["close"]
    current = float(close.iloc[-1])
    ma_20 = close.rolling(20).mean()
    ma_50 = close.rolling(50).mean()

    # MA slopes (only if we have enough data)
    slope_20 = compute_ma_slope(close, 20) if len(close) >= 22 else None
    slope_50 = compute_ma_slope(close, 50) if len(close) >= 52 else None

    # Price vs MAs
    above_ma20 = current > ma_20.iloc[-1] if not pd.isna(ma_20.iloc[-1]) else None
    above_ma50 = current > ma_50.iloc[-1] if len(ma_50) > 0 and not pd.isna(ma_50.iloc[-1]) else None

    # Recent structure: higher highs / lower lows
    half = len(close) // 2
    first_half_high = float(close.iloc[:half].max())
    second_half_high = float(close.iloc[half:].max())
    first_half_low = float(close.iloc[:half].min())
    second_half_low = float(close.iloc[half:].min())

    # Compressing or expanding
    recent_range = float(close.iloc[-10:].max() - close.iloc[-10:].min())
    older_range = float(close.iloc[-20:-10].max() - close.iloc[-20:-10].min()) if len(close) >= 20 else recent_range
    range_change = recent_range / older_range if older_range > 0 else 1.0

    # Build summary
    parts = []

    # Direction
    if slope_20 is not None:
        if slope_20 > 0.0005:
            parts.append("bullish")
        elif slope_20 < -0.0005:
            parts.append("bearish")
        else:
            parts.append("ranging")
    else:
        parts.append("unclear trend")

    # Structure
    if second_half_high > first_half_high and second_half_low > first_half_low:
        parts.append("higher highs/higher lows")
    elif second_half_high < first_half_high and second_half_low < first_half_low:
        parts.append("lower highs/lower lows")
    else:
        parts.append("mixed structure")

    # Volatility regime
    if range_change > 1.3:
        parts.append("expanding")
    elif range_change < 0.7:
        parts.append("compressing")
    else:
        parts.append("normal volatility")

    # MA position
    ma_context = []
    if above_ma20 is True:
        ma_context.append("above 20MA")
    elif above_ma20 is False:
        ma_context.append("below 20MA")
    if above_ma50 is True:
        ma_context.append("above 50MA")
    elif above_ma50 is False:
        ma_context.append("below 50MA")
    if ma_context:
        parts.append(", ".join(ma_context))

    prefix = f"[{label}] " if label else ""
    return prefix + ", ".join(parts)


# ---------------------------------------------------------------------------
# Price location
# ---------------------------------------------------------------------------

def price_location(df: pd.DataFrame) -> str:
    """Determine where current price sits relative to structure."""
    if len(df) < 10:
        return "no_clear_area"

    current = float(df["close"].iloc[-1])
    levels = find_levels(df, current)
    recent_high = float(df["high"].iloc[-20:].max())
    recent_low = float(df["low"].iloc[-20:].min())

    near_support = levels["nearest_support"]
    near_resistance = levels["nearest_resistance"]

    # Check proximity to levels
    if near_resistance and abs(current - near_resistance) / near_resistance < 0.005:
        return "near_resistance"
    if near_support and abs(current - near_support) / near_support < 0.005:
        return "near_support"

    # Position in range
    range_size = recent_high - recent_low
    if range_size > 0:
        position = (current - recent_low) / range_size
        if position > 0.7:
            return "range_upper_area"
        elif position < 0.3:
            return "range_lower_area"
        else:
            return "range_middle"

    return "no_clear_area"


# ---------------------------------------------------------------------------
# Risk estimation
# ---------------------------------------------------------------------------

def estimate_risk(
    direction: str,
    entry_price: float,
    df: pd.DataFrame,
) -> Dict:
    """
    Estimate candidate invalidation (stop), target, and reward-to-risk.
    """
    atr_series = compute_atr(df)
    atr_value = float(atr_series.iloc[-1]) if not pd.isna(atr_series.iloc[-1]) else entry_price * 0.005
    current = entry_price

    levels = find_levels(df, current)

    if direction.upper() == "BUY":
        # Stop below nearest support or recent low
        stop_candidates = []
        if levels["nearest_support"]:
            stop_candidates.append(levels["nearest_support"] * 0.998)  # just below support
        stop_candidates.append(current - atr_value * 1.5)

        # Conservative: use the tighter stop (higher price = less risk)
        stop = max(stop_candidates)

        # Target at nearest resistance or 2*risk
        target_candidates = []
        if levels["nearest_resistance"]:
            target_candidates.append(levels["nearest_resistance"])
        target_candidates.append(current + (current - stop) * config.CANDIDATE_GROSS_REWARD_TO_RISK)

        target = min(target_candidates)  # conservative target
        if target <= current:
            target = current + (current - stop) * config.CANDIDATE_GROSS_REWARD_TO_RISK

    else:  # SELL
        # Stop above nearest resistance or recent high
        stop_candidates = []
        if levels["nearest_resistance"]:
            stop_candidates.append(levels["nearest_resistance"] * 1.002)
        stop_candidates.append(current + atr_value * 1.5)

        stop = min(stop_candidates)

        # Target at nearest support or 2*risk
        target_candidates = []
        if levels["nearest_support"]:
            target_candidates.append(levels["nearest_support"])
        target_candidates.append(current - (stop - current) * config.CANDIDATE_GROSS_REWARD_TO_RISK)

        target = max(target_candidates)
        if target >= current:
            target = current - (stop - current) * config.CANDIDATE_GROSS_REWARD_TO_RISK

    risk = abs(current - stop)
    reward = abs(target - current)
    rr = reward / risk if risk > 0 else 0

    return {
        "entry": float(round(current, 2)),
        "stop": float(round(stop, 2)),
        "target": float(round(target, 2)),
        "risk": float(round(risk, 2)),
        "reward": float(round(reward, 2)),
        "reward_to_risk": float(round(rr, 2)),
        "atr": float(round(atr_value, 2)),
    }
