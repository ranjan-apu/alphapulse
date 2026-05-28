"""
Confluence scoring and trade location evaluation.

Scores the quality of a potential trade setup by combining:
- Higher timeframe alignment
- Market structure
- Volume/VWAP
- Risk quality
- Level quality (touch count, recency, volume at level)
"""
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class ConfluenceScore:
    """Aggregated confluence score across multiple factors."""
    htf_alignment: int = 0     # 0-5
    structure_quality: int = 0  # 0-5
    volume_quality: int = 0     # 0-5
    risk_quality: int = 0       # 0-5
    level_quality: int = 0      # 0-5
    total: int = 0              # sum / 25
    
    @property
    def normalized(self) -> float:
        return self.total / 25.0
    
    @property
    def is_tradable(self) -> bool:
        """Minimum thresholds for a tradable setup."""
        return (
            self.structure_quality >= 3
            and self.risk_quality >= 4
            and self.total >= 15
        )


@dataclass
class TradeLocation:
    """Analysis of current price location relative to key levels."""
    at_support: bool = False
    at_resistance: bool = False
    at_vwap: bool = False
    at_value_edge: bool = False  # VAH or VAL
    at_range_edge: bool = False
    at_range_middle: bool = False
    near_poc: bool = False
    at_prior_level: bool = False
    
    # Closest levels
    nearest_support: Optional[float] = None
    nearest_resistance: Optional[float] = None
    distance_to_support: Optional[float] = None
    distance_to_resistance: Optional[float] = None


def score_level_quality(
    level_price: float,
    touches: int,
    recency: int,  # candles since last touch (lower = more recent)
    volume_at_level: float = 0,
    htf_confluence: bool = False,
) -> int:
    """
    Score the quality of a price level (0-5).
    
    Factors:
    - Touch count: more touches = stronger level
    - Recency: recently tested levels are more relevant
    - Volume: high volume at level = institutional interest
    - HTF confluence: level aligns with higher timeframe
    """
    score = 0
    
    # Touch count
    if touches >= 4:
        score += 2
    elif touches >= 2:
        score += 1
    
    # Recency (fewer candles since last touch = better)
    if recency <= 3:
        score += 1
    
    # Volume
    if volume_at_level > 0:
        score += 1
    
    # HTF confluence
    if htf_confluence:
        score += 1
    
    return min(score, 5)


def score_trade_location(
    current_price: float,
    supports: List[float],
    resistances: List[float],
    vwap: Optional[float] = None,
    vah: Optional[float] = None,
    val: Optional[float] = None,
    poc: Optional[float] = None,
    range_high: Optional[float] = None,
    range_low: Optional[float] = None,
    atr: Optional[float] = None,
) -> tuple:
    """
    Score the trade location quality.
    
    Returns:
        (TradeLocation, quality_score: 0-5)
    """
    location = TradeLocation()
    
    # Find nearest support and resistance
    supports_above = [s for s in supports if s < current_price]
    resistances_below = [r for r in resistances if r > current_price]
    
    location.nearest_support = max(supports_above) if supports_above else None
    location.nearest_resistance = min(resistances_below) if resistances_below else None
    
    if location.nearest_support:
        location.distance_to_support = round(current_price - location.nearest_support, 2)
    if location.nearest_resistance:
        location.distance_to_resistance = round(location.nearest_resistance - current_price, 2)
    
    # At support (within 0.2 ATR)
    threshold = (atr or (current_price * 0.005)) * 0.2
    if location.nearest_support and location.distance_to_support <= threshold:
        location.at_support = True
    
    if location.nearest_resistance and location.distance_to_resistance <= threshold:
        location.at_resistance = True
    
    # At VWAP
    if vwap and abs(current_price - vwap) <= threshold:
        location.at_vwap = True
    
    # At value edge
    if vah and abs(current_price - vah) <= threshold:
        location.at_value_edge = True
    elif val and abs(current_price - val) <= threshold:
        location.at_value_edge = True
    
    # At range edge
    if range_high and abs(current_price - range_high) <= threshold:
        location.at_range_edge = True
    elif range_low and abs(current_price - range_low) <= threshold:
        location.at_range_edge = True
    
    # At range middle
    if range_high and range_low:
        mid = (range_high + range_low) / 2
        range_size = range_high - range_low
        if range_size > 0 and abs(current_price - mid) < range_size * 0.2:
            location.at_range_middle = True
    
    # Near POC
    if poc and abs(current_price - poc) <= threshold:
        location.near_poc = True
    
    # Quality score (0-5)
    quality = 0
    
    if location.at_support or location.at_resistance:
        quality += 2  # Clear level proximity
    elif location.at_value_edge:
        quality += 2  # Value area edge
    elif location.at_vwap:
        quality += 1  # VWAP is a reference
    elif location.at_range_edge:
        quality += 1  # Range edge
    elif location.near_poc:
        quality += 1  # POC magnet area
    
    # Penalize range middle
    if location.at_range_middle:
        quality = max(0, quality - 2)
    
    # If near both support AND resistance (< 2 ATR apart), it's congested
    if (
        location.nearest_support
        and location.nearest_resistance
        and (location.nearest_resistance - location.nearest_support) < (atr or 0) * 2
    ):
        quality = max(0, quality - 1)
    
    return location, min(quality, 5)


def score_confluence(
    htf_bias: str,           # 'bullish', 'bearish', 'neutral'
    structure_state: str,     # 'bullish_bos', 'bearish_bos', etc
    vwap_relation: str,       # 'above_vwap', 'below_vwap', 'at_vwap'
    location_quality: int,    # 0-5
    trigger_quality: int,     # 0-5
    volume_confirmation: int, # 0-5
    risk_quality: int,        # 0-5
) -> ConfluenceScore:
    """
    Compute the overall confluence score for a trade setup.
    
    Combines HTF alignment, structure, volume/VWAP, risk,
    location, and trigger quality into a single score.
    """
    score = ConfluenceScore()
    
    # HTF alignment (0-5)
    if htf_bias == "bullish" and structure_state in ("bullish_bos", "HH_HL"):
        score.htf_alignment = 5
    elif htf_bias == "bearish" and structure_state in ("bearish_bos", "LH_LL"):
        score.htf_alignment = 5
    elif htf_bias == "neutral":
        score.htf_alignment = 3
    elif (htf_bias == "bullish" and structure_state in ("bearish_bos", "LH_LL")) or \
         (htf_bias == "bearish" and structure_state in ("bullish_bos", "HH_HL")):
        score.htf_alignment = 1  # Counter-trend
    else:
        score.htf_alignment = 2
    
    # Structure quality (0-5)
    if "bos" in structure_state.lower():
        score.structure_quality = 4
    elif "choch" in structure_state.lower():
        score.structure_quality = 3
    elif structure_state == "range_bound":
        score.structure_quality = 2
    else:
        score.structure_quality = 1
    
    # Volume quality (0-5)
    score.volume_quality = volume_confirmation
    
    # Risk quality (0-5)
    score.risk_quality = risk_quality
    
    # Level quality (derived from location)
    score.level_quality = location_quality
    
    score.total = (
        score.htf_alignment
        + score.structure_quality
        + score.volume_quality
        + score.risk_quality
        + score.level_quality
    )
    
    return score


def location_to_dict(location: TradeLocation) -> dict:
    """Convert TradeLocation to dict."""
    return {
        "at_support": location.at_support,
        "at_resistance": location.at_resistance,
        "at_vwap": location.at_vwap,
        "at_value_edge": location.at_value_edge,
        "at_range_edge": location.at_range_edge,
        "at_range_middle": location.at_range_middle,
        "near_poc": location.near_poc,
        "nearest_support": location.nearest_support,
        "nearest_resistance": location.nearest_resistance,
        "distance_to_support": location.distance_to_support,
        "distance_to_resistance": location.distance_to_resistance,
    }
