"""
Gap open handling for intraday trading.
Detects and classifies gap opens, providing gap context
for the agent's session analysis.

Fallback classification (when volume profile not available):
- gap_up_above_value   -> use prior day high instead of VAH
- gap_down_below_value -> use prior day low instead of VAL
- gap_up_inside_value  -> open between prior low and prior high
- gap_down_inside_value -> open between prior low and prior high
"""
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class GapContext:
    """Gap open analysis for a trading session."""
    prior_close: float
    today_open: float
    gap_points: float
    gap_pct: float
    gap_atr_multiple: Optional[float]
    gap_direction: str          # 'gap_up', 'gap_down', 'flat_open'
    open_location_vs_prior_value: str  # 'above_vah', 'below_val', 'inside_value', etc.
    gap_status: str             # 'unfilled', 'filling', 'filled', 'no_gap'
    gap_fill_level: float       # prior close level
    gap_type: str               # classified gap type for agent reasoning
    prior_high: Optional[float] = None
    prior_low: Optional[float] = None
    prior_vah: Optional[float] = None
    prior_val: Optional[float] = None
    prior_poc: Optional[float] = None


def classify_gap(
    prior_close: float,
    today_open: float,
    prior_high: Optional[float] = None,
    prior_low: Optional[float] = None,
    prior_vah: Optional[float] = None,
    prior_val: Optional[float] = None,
    prior_poc: Optional[float] = None,
    atr: Optional[float] = None,
) -> GapContext:
    """
    Classify a gap open for the current session.

    Args:
        prior_close: Previous session's closing price
        today_open: Today's opening price
        prior_high: Prior day's high
        prior_low: Prior day's low
        prior_vah: Prior day's Value Area High (from volume profile)
        prior_val: Prior day's Value Area Low (from volume profile)
        prior_poc: Prior day's Point of Control (from volume profile)
        atr: Average True Range for volatility context

    Returns:
        GapContext with full classification
    """
    gap_points = today_open - prior_close
    gap_pct = (gap_points / prior_close) * 100 if prior_close > 0 else 0

    # Direction
    if abs(gap_pct) < 0.1:
        gap_direction = "flat_open"
    elif gap_points > 0:
        gap_direction = "gap_up"
    else:
        gap_direction = "gap_down"

    gap_fill_level = prior_close

    # ATR multiple
    gap_atr_multiple = abs(gap_points) / atr if atr and atr > 0 else None

    # Determine location vs prior value area
    # Fallback: use prior day high/low if volume profile not available
    upper_boundary = prior_vah if prior_vah is not None else (prior_high or prior_close)
    lower_boundary = prior_val if prior_val is not None else (prior_low or prior_close)

    if gap_direction == "gap_up":
        if today_open > upper_boundary:
            open_location = "above_value"
        else:
            open_location = "inside_value"
    elif gap_direction == "gap_down":
        if today_open < lower_boundary:
            open_location = "below_value"
        else:
            open_location = "inside_value"
    else:
        open_location = "at_open"

    # Classify gap type
    gap_type = _classify_gap_type(
        gap_direction, open_location, gap_atr_multiple, gap_pct
    )

    # Gap status (initially unfilled if there's a gap, no_gap if flat)
    if gap_direction == "flat_open":
        gap_status = "no_gap"
    else:
        gap_status = "unfilled"

    return GapContext(
        prior_close=round(prior_close, 2),
        today_open=round(today_open, 2),
        gap_points=round(gap_points, 2),
        gap_pct=round(gap_pct, 2),
        gap_atr_multiple=round(gap_atr_multiple, 2) if gap_atr_multiple else None,
        gap_direction=gap_direction,
        open_location_vs_prior_value=open_location,
        gap_status=gap_status,
        gap_fill_level=round(gap_fill_level, 2),
        gap_type=gap_type,
        prior_high=round(prior_high, 2) if prior_high else None,
        prior_low=round(prior_low, 2) if prior_low else None,
        prior_vah=round(prior_vah, 2) if prior_vah else None,
        prior_val=round(prior_val, 2) if prior_val else None,
        prior_poc=round(prior_poc, 2) if prior_poc else None,
    )


def _classify_gap_type(
    direction: str,
    location: str,
    atr_multiple: Optional[float],
    gap_pct: float,
) -> str:
    """Classify the gap type for the agent's reasoning."""
    if direction == "flat_open":
        return "no_gap"

    # Small gap (< 0.3%)
    if abs(gap_pct) < 0.3:
        return f"small_gap_{direction}"

    # Gap inside prior value area
    if location == "inside_value":
        return f"gap_{direction}_inside_value"

    # Gap outside prior value area
    if atr_multiple is None:
        if direction == "gap_up" and location == "above_value":
            return "gap_and_go_candidate"
        elif direction == "gap_down" and location == "below_value":
            return "gap_and_go_candidate"
        else:
            return "gap_fade_candidate"

    # Large gap (more than 1.5 ATR) = gap and go candidate
    if atr_multiple > 1.5:
        return "gap_and_go_candidate"

    # Moderate gap (0.5 - 1.5 ATR)
    if atr_multiple > 0.5:
        return "gap_fade_candidate"

    return "small_gap"


def get_gap_context_dict(gap: GapContext) -> dict:
    """Convert GapContext to a dict for MarketStatePackage."""
    return {
        "prior_close": gap.prior_close,
        "today_open": gap.today_open,
        "gap_points": gap.gap_points,
        "gap_pct": gap.gap_pct,
        "gap_atr_multiple": gap.gap_atr_multiple,
        "gap_direction": gap.gap_direction,
        "open_location_vs_prior_value": gap.open_location_vs_prior_value,
        "gap_status": gap.gap_status,
        "gap_fill_level": gap.gap_fill_level,
        "gap_type": gap.gap_type,
        "prior_high": gap.prior_high,
        "prior_low": gap.prior_low,
        "prior_vah": gap.prior_vah,
        "prior_val": gap.prior_val,
        "prior_poc": gap.prior_poc,
    }
