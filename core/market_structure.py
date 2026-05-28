"""
Market structure detection: swings, BOS (Break of Structure), CHOCH (Change of Character),
HH/HL/LH/LL patterns, range state, and breakout quality.

Key concepts:
- HH: Higher High (price makes a new swing high above prior swing high)
- HL: Higher Low (pullback low is higher than prior swing low)
- LH: Lower High (rally high is lower than prior swing high)
- LL: Lower Low (price makes a new swing low below prior swing low)
- BOS: Break of Structure (continuation - breaks a swing in trend direction)
- CHOCH: Change of Character (reversal - breaks the last opposing swing)
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class SwingPoint:
    """A detected swing high or low."""
    price: float
    timestamp: str
    swing_type: str  # 'high' or 'low'
    index: int


@dataclass
class MarketStructure:
    """Market structure analysis."""
    state: str = "unclear"  # 'bullish_bos', 'bearish_bos', 'range_bound', 'choch'
    trend_structure: str = "unclear"  # 'HH_HL', 'LH_LL', 'mixed'
    swings: List[SwingPoint] = field(default_factory=list)
    recent_bos: Optional[Dict] = None   # Most recent break of structure
    recent_choch: Optional[Dict] = None # Most recent change of character
    range_high: Optional[float] = None
    range_low: Optional[float] = None
    range_state: str = "no_range"  # 'range_bound', 'range_breakout', 'no_range'
    breakout_quality: str = "none"  # 'strong', 'weak', 'fakeout_risk', 'none'
    liquidity_above: List[float] = field(default_factory=list)
    liquidity_below: List[float] = field(default_factory=list)


def detect_market_structure(
    df_intraday: pd.DataFrame,
    lookback: int = 20,
    swing_strength: int = 2,
) -> MarketStructure:
    """
    Detect market structure from intraday price data.

    Uses swing detection to find HH/HL/LH/LL patterns,
    then identifies BOS and CHOCH events.

    Args:
        df_intraday: OHLCV data
        lookback: Number of candles to analyze
        swing_strength: Minimum candles on each side for a swing point

    Returns:
        MarketStructure with full analysis
    """
    if df_intraday.empty or len(df_intraday) < swing_strength * 2 + 1:
        return MarketStructure()

    df = df_intraday.iloc[-lookback:].copy() if len(df_intraday) > lookback else df_intraday.copy()
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    timestamps = df.index

    # Detect swing highs and lows
    swings = _detect_swing_points(highs, lows, timestamps, swing_strength)

    # Classify structure
    structure = _classify_structure(swings, highs, lows)

    # Detect BOS and CHOCH
    structure.recent_bos = _detect_bos(swings, closes, structure.trend_structure)
    structure.recent_choch = _detect_choch(swings, closes)

    # Range analysis
    range_high, range_low, range_state = _analyze_range(highs, lows, lookback // 2)
    structure.range_high = range_high
    structure.range_low = range_low
    structure.range_state = range_state

    # Breakout quality
    structure.breakout_quality = _assess_breakout_quality(df, swings)

    # Liquidity zones
    structure.liquidity_above, structure.liquidity_below = _find_liquidity_zones(swings)

    return structure


def _detect_swing_points(
    highs: np.ndarray,
    lows: np.ndarray,
    timestamps,
    strength: int,
) -> List[SwingPoint]:
    """Detect swing highs and lows using local extrema."""
    swings = []
    n = len(highs)

    for i in range(strength, n - strength):
        # Check if swing high
        is_swing_high = True
        for j in range(1, strength + 1):
            if highs[i] <= highs[i - j] or highs[i] <= highs[i + j]:
                is_swing_high = False
                break
        if is_swing_high:
            swings.append(SwingPoint(
                price=float(highs[i]),
                timestamp=str(timestamps[i]),
                swing_type="high",
                index=i,
            ))

        # Check if swing low
        is_swing_low = True
        for j in range(1, strength + 1):
            if lows[i] >= lows[i - j] or lows[i] >= lows[i + j]:
                is_swing_low = False
                break
        if is_swing_low:
            swings.append(SwingPoint(
                price=float(lows[i]),
                timestamp=str(timestamps[i]),
                swing_type="low",
                index=i,
            ))

    # Sort by index
    swings.sort(key=lambda s: s.index)
    return swings


def _classify_structure(
    swings: List[SwingPoint],
    highs: np.ndarray,
    lows: np.ndarray,
) -> MarketStructure:
    """Classify market structure as HH/HL, LH/LL, or mixed."""
    structure = MarketStructure()
    structure.swings = swings

    if len(swings) < 4:
        structure.state = "unclear"
        structure.trend_structure = "unclear"
        return structure

    # Get last 4 swing points
    recent = swings[-4:]
    types = [s.swing_type for s in recent]

    # Need alternating high-low-high-low or low-high-low-high
    if types != ["high", "low", "high", "low"] and types != ["low", "high", "low", "high"]:
        structure.state = "unclear"
        structure.trend_structure = "mixed"
        return structure

    prices = [s.price for s in recent]

    if types == ["high", "low", "high", "low"]:
        h1, l1, h2, l2 = prices
        if h2 > h1 and l2 > l1:
            structure.trend_structure = "HH_HL"
            structure.state = "bullish_bos"
        elif h2 < h1 and l2 < l1:
            structure.trend_structure = "LH_LL"
            structure.state = "bearish_bos"
        elif h2 < h1 and l2 > l1:
            structure.state = "choch"
            structure.trend_structure = "LH_HL"
        else:
            structure.state = "range_bound"
            structure.trend_structure = "mixed"
    elif types == ["low", "high", "low", "high"]:
        l1, h1, l2, h2 = prices
        if h2 > h1 and l2 > l1:
            structure.trend_structure = "HH_HL"
            structure.state = "bullish_bos"
        elif h2 < h1 and l2 < l1:
            structure.trend_structure = "LH_LL"
            structure.state = "bearish_bos"
        elif h2 < h1 and l2 > l1:
            structure.state = "choch"
            structure.trend_structure = "LH_HL"
        else:
            structure.state = "range_bound"
            structure.trend_structure = "mixed"

    return structure


def _detect_bos(
    swings: List[SwingPoint],
    closes: np.ndarray,
    trend_structure: str = "unclear",
) -> Optional[Dict]:
    """
    Detect most recent Break of Structure.
    
    A BOS must be in the trend direction:
    - Bullish trend (HH_HL): BOS occurs when price breaks above last swing HIGH
    - Bearish trend (LH_LL): BOS occurs when price breaks below last swing LOW
    """
    if len(swings) < 3:
        return None

    # A BOS occurs when price breaks the last swing in trend direction
    recent_swings = swings[-3:]
    last_close = float(closes[-1]) if len(closes) > 0 else 0

    swing_highs = [s for s in recent_swings if s.swing_type == "high"]
    swing_lows = [s for s in recent_swings if s.swing_type == "low"]

    # Bullish BOS: price breaks above last swing high (in trend direction)
    if swing_highs and last_close > swing_highs[-1].price:
        # Only valid in bullish trend or unclear
        if trend_structure in ("HH_HL", "unclear", ""):
            return {
                "type": "bullish_bos",
                "broken_level": swing_highs[-1].price,
                "broken_at": swing_highs[-1].timestamp,
                "current_price": last_close,
            }

    # Bearish BOS: price breaks below last swing low (in trend direction)
    if swing_lows and last_close < swing_lows[-1].price:
        if trend_structure in ("LH_LL", "unclear", ""):
            return {
                "type": "bearish_bos",
                "broken_level": swing_lows[-1].price,
                "broken_at": swing_lows[-1].timestamp,
                "current_price": last_close,
            }

    return None


def _detect_choch(swings: List[SwingPoint], closes: np.ndarray) -> Optional[Dict]:
    """Detect most recent Change of Character (reversal)."""
    if len(swings) < 3:
        return None

    recent_swings = swings[-3:]
    last_close = float(closes[-1]) if len(closes) > 0 else 0

    # CHOCH: price breaks the last opposing swing
    # In an uptrend, breaking below last swing low = bearish CHOCH
    swing_types = [s.swing_type for s in recent_swings]

    if len(swing_types) >= 3 and swing_types[-1] == "high":
        # Last swing was a high, check if price broke below prior low
        swing_lows = [s for s in recent_swings if s.swing_type == "low"]
        if swing_lows and last_close < swing_lows[-1].price:
            return {
                "type": "bearish_choch",
                "broken_level": swing_lows[-1].price,
                "direction_change": "bullish_to_bearish",
            }
    elif len(swing_types) >= 3 and swing_types[-1] == "low":
        swing_highs = [s for s in recent_swings if s.swing_type == "high"]
        if swing_highs and last_close > swing_highs[-1].price:
            return {
                "type": "bullish_choch",
                "broken_level": swing_highs[-1].price,
                "direction_change": "bearish_to_bullish",
            }

    return None


def _analyze_range(
    highs: np.ndarray,
    lows: np.ndarray,
    period: int,
) -> Tuple[Optional[float], Optional[float], str]:
    """Analyze if price is range-bound."""
    if len(highs) < period:
        return None, None, "no_range"

    recent_highs = highs[-period:]
    recent_lows = lows[-period:]

    range_high = float(np.max(recent_highs))
    range_low = float(np.min(recent_lows))
    range_size = range_high - range_low

    last_close = float(highs[-1]) if len(highs) > 0 else 0  # approximate

    # Check if price near range boundaries
    near_high = abs(last_close - range_high) / (range_size or 1) < 0.1
    near_low = abs(last_close - range_low) / (range_size or 1) < 0.1

    if near_high:
        return range_high, range_low, "range_high_test"
    elif near_low:
        return range_high, range_low, "range_low_test"
    elif range_high - range_low < range_high * 0.02:  # <2% range
        return range_high, range_low, "range_bound"
    else:
        return range_high, range_low, "range_breakout"


def _assess_breakout_quality(
    df: pd.DataFrame,
    swings: List[SwingPoint],
) -> str:
    """Assess the quality of a breakout."""
    if len(df) < 5:
        return "none"

    # Check volume confirmation
    recent_vol = df["volume"].iloc[-3:].values
    avg_vol = df["volume"].iloc[-10:-3].mean() if len(df) > 10 else recent_vol.mean()
    vol_expansion = recent_vol[-1] > avg_vol * 1.5 if avg_vol > 0 else False

    # Check candle size
    recent_candles = df.iloc[-3:]
    avg_range = (df["high"] - df["low"]).iloc[-10:-3].mean() if len(df) > 10 else 0
    expansion_candle = any(
        (c["high"] - c["low"]) > avg_range * 1.5
        for _, c in recent_candles.iterrows()
    )

    # Check retest status
    if len(swings) >= 2:
        last_swing = swings[-1]
        if last_swing.swing_type == "high":
            # Check if price retested the broken level
            recent_lows = df["low"].iloc[-3:].values
            if any(abs(low - last_swing.price) / last_swing.price < 0.002 for low in recent_lows):
                return "retest_holding" if vol_expansion else "retest_in_progress"

    if vol_expansion and expansion_candle:
        return "strong"
    elif vol_expansion or expansion_candle:
        return "moderate"
    else:
        return "weak"


def _find_liquidity_zones(swings: List[SwingPoint]) -> Tuple[List[float], List[float]]:
    """Find liquidity zones above/below swing points."""
    liquidity_above = []
    liquidity_below = []

    for swing in swings:
        if swing.swing_type == "high":
            liquidity_above.append(swing.price)
        else:
            liquidity_below.append(swing.price)

    # Filter to most recent and significant
    liquidity_above = sorted(set(round(p, 2) for p in liquidity_above))[-3:]
    liquidity_below = sorted(set(round(p, 2) for p in liquidity_below))[:3]

    return liquidity_above, liquidity_below


def structure_to_dict(structure: MarketStructure) -> dict:
    """Convert MarketStructure to dict for MarketStatePackage."""
    return {
        "state": structure.state,
        "trend_structure": structure.trend_structure,
        "recent_bos": structure.recent_bos,
        "recent_choch": structure.recent_choch,
        "range_high": structure.range_high,
        "range_low": structure.range_low,
        "range_state": structure.range_state,
        "breakout_quality": structure.breakout_quality,
        "liquidity_above": structure.liquidity_above,
        "liquidity_below": structure.liquidity_below,
        "swing_count": len(structure.swings),
    }
