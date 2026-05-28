"""
Session VWAP (Volume-Weighted Average Price) computation.
Resets at each trading session. Provides VWAP, bands, slope,
and VWAP relation analysis.

Based on Investopedia VWAP reference (Section 2.6 of plan).
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict


@dataclass
class VWAPResult:
    """Computed VWAP for a session with analytics."""
    current_vwap: float
    vwap_slope: float          # Positive = rising, negative = falling
    distance_from_vwap: float   # In points
    distance_pct: float         # As percentage
    distance_atr: Optional[float]  # In ATR multiples
    relation: str               # 'above_vwap', 'below_vwap', 'at_vwap'
    band_upper_1: Optional[float]  # +1 standard deviation
    band_upper_2: Optional[float]  # +2 standard deviation
    band_lower_1: Optional[float]  # -1 standard deviation
    band_lower_2: Optional[float]  # -2 standard deviation
    vwap_reclaim: bool           # Price recently reclaimed VWAP
    vwap_rejection: bool         # Price recently rejected VWAP
    trend_interpretation: str    # 'trend_day', 'mean_reversion', 'unclear'


def compute_session_vwap(
    df_intraday: pd.DataFrame,
    current_price: float,
    session_start_col: str = "timestamp",
    atr: Optional[float] = None,
) -> VWAPResult:
    """
    Compute session VWAP from intraday data.

    VWAP = sum(price * volume) / sum(volume) for the session.
    Uses typical price: (high + low + close) / 3

    Args:
        df_intraday: Intraday OHLCV data for current session
        current_price: Current/last price
        session_start_col: Timestamp column (default index)
        atr: ATR for distance measurement

    Returns:
        VWAPResult with analytics
    """
    if df_intraday.empty:
        return VWAPResult(
            current_vwap=current_price,
            vwap_slope=0.0,
            distance_from_vwap=0.0,
            distance_pct=0.0,
            distance_atr=None,
            relation="at_vwap",
            band_upper_1=None,
            band_upper_2=None,
            band_lower_1=None,
            band_lower_2=None,
            vwap_reclaim=False,
            vwap_rejection=False,
            trend_interpretation="unclear",
        )

    # Typical price = (H + L + C) / 3
    typical_price = (df_intraday["high"] + df_intraday["low"] + df_intraday["close"]) / 3.0
    volume = df_intraday["volume"].astype(float)

    # Cumulative VWAP
    cum_price_vol = (typical_price * volume).cumsum()
    cum_vol = volume.cumsum()
    vwap_series = cum_price_vol / cum_vol.replace(0, np.nan)

    current_vwap = float(vwap_series.iloc[-1]) if not vwap_series.empty else current_price

    # Standard deviation bands
    # Std of typical price deviations from VWAP
    if len(vwap_series) >= 5 and volume.sum() > 0:
        deviations = typical_price - vwap_series
        std_dev = float(np.sqrt((deviations ** 2 * volume).sum() / volume.sum()))
    else:
        std_dev = None

    band_upper_1 = round(current_vwap + std_dev, 2) if std_dev else None
    band_upper_2 = round(current_vwap + 2 * std_dev, 2) if std_dev else None
    band_lower_1 = round(current_vwap - std_dev, 2) if std_dev else None
    band_lower_2 = round(current_vwap - 2 * std_dev, 2) if std_dev else None

    # VWAP slope (linear regression on last N points)
    if len(vwap_series) >= 5:
        y = vwap_series.iloc[-10:].values
        x = np.arange(len(y))
        if len(y) >= 2 and np.std(y) > 0:
            slope, _ = np.polyfit(x, y, 1)
            vwap_slope = float(slope)
        else:
            vwap_slope = 0.0
    else:
        vwap_slope = 0.0

    # Distance from VWAP
    distance = current_price - current_vwap
    distance_pct = (distance / current_vwap * 100) if current_vwap > 0 else 0
    distance_atr = distance / atr if atr and atr > 0 else None

    # Relation
    if abs(distance_pct) < 0.1:
        relation = "at_vwap"
    elif distance > 0:
        relation = "above_vwap"
    else:
        relation = "below_vwap"

    # VWAP reclaim/rejection detection (last 3 candles)
    vwap_reclaim = False
    vwap_rejection = False
    if len(df_intraday) >= 5:
        recent = df_intraday.iloc[-5:]
        vwap_recent = vwap_series.iloc[-5:]

        for i in range(len(recent) - 1):
            prev_close = recent["close"].iloc[i]
            curr_close = recent["close"].iloc[i + 1]
            prev_vwap = vwap_recent.iloc[i]
            curr_vwap = vwap_recent.iloc[i + 1]

            # Reclaim: crossed above VWAP
            if prev_close <= prev_vwap and curr_close > curr_vwap:
                vwap_reclaim = True
            # Rejection: touched VWAP but closed away
            if (
                recent["low"].iloc[i + 1] <= curr_vwap <= recent["high"].iloc[i + 1]
                and curr_close < curr_vwap
                and prev_close > prev_vwap
            ):
                vwap_rejection = True

    # Trend interpretation
    if vwap_slope > 0.001 and relation == "above_vwap":
        trend_interp = "trend_day"
    elif vwap_slope < -0.001 and relation == "below_vwap":
        trend_interp = "trend_day"
    elif abs(vwap_slope) < 0.0005:
        trend_interp = "mean_reversion"
    else:
        trend_interp = "unclear"

    return VWAPResult(
        current_vwap=round(current_vwap, 2),
        vwap_slope=round(vwap_slope, 6),
        distance_from_vwap=round(distance, 2),
        distance_pct=round(distance_pct, 2),
        distance_atr=round(distance_atr, 2) if distance_atr else None,
        relation=relation,
        band_upper_1=band_upper_1,
        band_upper_2=band_upper_2,
        band_lower_1=band_lower_1,
        band_lower_2=band_lower_2,
        vwap_reclaim=vwap_reclaim,
        vwap_rejection=vwap_rejection,
        trend_interpretation=trend_interp,
    )


def vwap_result_to_dict(result: VWAPResult) -> dict:
    """Convert VWAPResult to dict for MarketStatePackage."""
    return {
        "current_vwap": result.current_vwap,
        "slope": result.vwap_slope,
        "distance_from_vwap": result.distance_from_vwap,
        "distance_pct": result.distance_pct,
        "distance_atr": result.distance_atr,
        "relation": result.relation,
        "bands": {
            "upper_1": result.band_upper_1,
            "upper_2": result.band_upper_2,
            "lower_1": result.band_lower_1,
            "lower_2": result.band_lower_2,
        },
        "vwap_reclaim": result.vwap_reclaim,
        "vwap_rejection": result.vwap_rejection,
        "trend_interpretation": result.trend_interpretation,
    }
