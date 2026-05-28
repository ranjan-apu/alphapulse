"""
Volume Profile computation for intraday trading.

Key concepts (from TradingView volume profile docs, Section 2.7 of plan):
- POC (Point of Control): highest-volume price level
- VAH (Value Area High): upper boundary of ~70% of session volume
- VAL (Value Area Low): lower boundary of ~70% of session volume
- HVN (High Volume Nodes): accepted/fair-value areas
- LVN (Low Volume Nodes): rejection/fast-move areas
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class VolumeProfileResult:
    """Volume profile analysis for a session or range."""
    poc: float                  # Point of Control (highest volume price)
    vah: Optional[float]        # Value Area High
    val: Optional[float]        # Value Area Low
    value_area_pct: float = 0.70  # Percentage of volume in value area
    hvn_levels: List[float] = None   # High Volume Nodes
    lvn_levels: List[float] = None   # Low Volume Nodes
    price_location: str = "no_data"  # 'inside_value', 'above_vah', 'below_val'
    profile_type: str = "unclear"    # 'normal', 'double_distribution', 'trending'

    def __post_init__(self):
        if self.hvn_levels is None:
            self.hvn_levels = []
        if self.lvn_levels is None:
            self.lvn_levels = []


def compute_volume_profile(
    df_intraday: pd.DataFrame,
    current_price: float,
    num_bins: int = 50,
    value_area_pct: float = 0.70,
) -> VolumeProfileResult:
    """
    Compute volume profile from intraday data.

    Uses the close price and volume for each candle.
    Creates price bins and accumulates volume at each bin.

    Args:
        df_intraday: Intraday OHLCV data
        current_price: Current price for location classification
        num_bins: Number of price bins for the profile
        value_area_pct: Percentage for value area (default 70%)

    Returns:
        VolumeProfileResult with POC, VAH, VAL, HVN, LVN
    """
    if df_intraday.empty or len(df_intraday) < 5:
        return VolumeProfileResult(
            poc=current_price,
            vah=None,
            val=None,
            price_location="no_data",
        )

    close = df_intraday["close"].values
    volume = df_intraday["volume"].astype(float).values
    typical_price = (df_intraday["high"] + df_intraday["low"] + df_intraday["close"]).values / 3.0

    # Create price bins
    price_min = float(np.min(typical_price))
    price_max = float(np.max(typical_price))
    price_range = price_max - price_min

    if price_range <= 0:
        return VolumeProfileResult(
            poc=current_price,
            vah=None,
            val=None,
            price_location="no_data",
        )

    bin_size = price_range / num_bins
    bins = np.linspace(price_min, price_max, num_bins + 1)

    # Accumulate volume in each bin
    vol_profile = np.zeros(num_bins)
    for i in range(len(typical_price)):
        tp = typical_price[i]
        vol = volume[i]
        bin_idx = min(int((tp - price_min) / bin_size), num_bins - 1)
        vol_profile[bin_idx] += vol

    # POC: bin with highest volume
    poc_idx = np.argmax(vol_profile)
    poc = float(bins[poc_idx] + bin_size / 2)  # center of bin

    # Total volume
    total_vol = float(np.sum(vol_profile))
    if total_vol <= 0:
        return VolumeProfileResult(
            poc=poc,
            vah=None,
            val=None,
            price_location="no_data",
        )

    # Value area: find range containing value_area_pct of volume around POC
    target_vol = total_vol * value_area_pct
    accumulated = vol_profile[poc_idx]
    upper_idx = poc_idx
    lower_idx = poc_idx

    while accumulated < target_vol:
        expanded = False
        # Try expand up
        if upper_idx < num_bins - 1:
            upper_idx += 1
            accumulated += vol_profile[upper_idx]
            expanded = True
        # Try expand down
        if accumulated < target_vol and lower_idx > 0:
            lower_idx -= 1
            accumulated += vol_profile[lower_idx]
            expanded = True
        if not expanded:
            break

    vah = float(bins[upper_idx + 1]) if upper_idx < num_bins - 1 else float(bins[upper_idx] + bin_size)
    val = float(bins[lower_idx])

    # HVN: bins with volume > 2x average
    avg_vol = total_vol / num_bins if num_bins > 0 else 0
    hvn_levels = []
    lvn_levels = []
    for i in range(num_bins):
        if vol_profile[i] > 2 * avg_vol:
            hvn_levels.append(round(float(bins[i] + bin_size / 2), 2))
        elif vol_profile[i] < 0.3 * avg_vol and vol_profile[i] > 0:
            lvn_levels.append(round(float(bins[i] + bin_size / 2), 2))

    # Price location vs value area
    if vah and val:
        if current_price > vah:
            price_location = "above_vah"
        elif current_price < val:
            price_location = "below_val"
        else:
            price_location = "inside_value"
    else:
        price_location = "no_value_area"

    # Profile type detection
    profile_type = _classify_profile_type(vol_profile, num_bins)

    return VolumeProfileResult(
        poc=round(poc, 2),
        vah=round(vah, 2) if vah else None,
        val=round(val, 2) if val else None,
        value_area_pct=value_area_pct,
        hvn_levels=hvn_levels[:5] if hvn_levels else [],
        lvn_levels=lvn_levels[:5] if lvn_levels else [],
        price_location=price_location,
        profile_type=profile_type,
    )


def _classify_profile_type(vol_profile: np.ndarray, num_bins: int) -> str:
    """Classify the shape of the volume profile."""
    total = float(np.sum(vol_profile))
    if total <= 0:
        return "unclear"

    # Find peaks (local maxima)
    peaks = []
    for i in range(1, num_bins - 1):
        if vol_profile[i] > vol_profile[i - 1] and vol_profile[i] > vol_profile[i + 1]:
            peaks.append((i, vol_profile[i]))

    # Sort peaks by volume
    peaks.sort(key=lambda x: x[1], reverse=True)

    if len(peaks) == 0:
        return "trending"
    elif len(peaks) == 1:
        return "normal"
    else:
        # Check if second peak has significant volume
        if len(peaks) >= 2 and peaks[1][1] > peaks[0][1] * 0.5:
            return "double_distribution"
        return "normal"


def volume_profile_to_dict(result: VolumeProfileResult) -> dict:
    """Convert VolumeProfileResult to dict for MarketStatePackage."""
    return {
        "poc": result.poc,
        "vah": result.vah,
        "val": result.val,
        "value_area_pct": result.value_area_pct,
        "hvn_levels": result.hvn_levels,
        "lvn_levels": result.lvn_levels,
        "price_location": result.price_location,
        "profile_type": result.profile_type,
    }


def compare_prior_value_area(
    current_price: float,
    prior_vah: Optional[float],
    prior_val: Optional[float],
    prior_poc: Optional[float],
) -> dict:
    """
    Compare current price to previous session's value area.

    Returns dict with relationship analysis.
    """
    if prior_vah is None or prior_val is None:
        return {"available": False, "message": "No prior value area data"}

    above_vah = current_price > prior_vah
    below_val = current_price < prior_val
    inside = not above_vah and not below_val

    distance_to_vah = current_price - prior_vah
    distance_to_val = current_price - prior_val
    distance_to_poc = current_price - prior_poc if prior_poc else None

    return {
        "available": True,
        "above_vah": above_vah,
        "below_val": below_val,
        "inside_value": inside,
        "distance_to_vah": round(distance_to_vah, 2),
        "distance_to_val": round(distance_to_val, 2),
        "distance_to_poc": round(distance_to_poc, 2) if distance_to_poc else None,
        "prior_vah": prior_vah,
        "prior_val": prior_val,
        "prior_poc": prior_poc,
    }
