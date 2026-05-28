"""
Configurable slippage model for trade execution simulation.
Applies adverse slippage to entry, exit, stop, and target prices.
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SlippageConfig:
    """Configuration for slippage applied at order execution."""
    mode: str = "fixed_paise_per_share"  # or "percentage" or "atr_based"
    
    # Fixed mode: paise (Rupee cents) per share adverse
    entry_slippage: float = 0.50          # ₹0.50 per share adverse (entry)
    exit_slippage: float = 0.50           # ₹0.50 per share adverse (exit)
    stop_slippage: float = 1.00           # ₹1.00 per share adverse (stop loss - wider)
    target_slippage: float = 0.50         # ₹0.50 per share adverse (target - smaller)
    force_squareoff_slippage: float = 0.75

    # Percentage mode
    entry_slippage_pct: float = 0.0002    # 0.02% per share
    exit_slippage_pct: float = 0.0002
    stop_slippage_pct: float = 0.0004
    target_slippage_pct: float = 0.0002
    force_squareoff_slippage_pct: float = 0.0003

    # ATR-based mode
    atr_multiplier_entry: float = 0.05
    atr_multiplier_exit: float = 0.05
    atr_multiplier_stop: float = 0.1
    atr_multiplier_target: float = 0.05
    atr_multiplier_force_squareoff: float = 0.05  # Previously hardcoded, now configurable


def apply_entry_slippage(
    price: float,
    direction: str,
    config: SlippageConfig,
    atr: Optional[float] = None,
) -> float:
    """
    Apply adverse slippage to an entry price.
    
    For BUY: entry becomes higher (worse)
    For SELL: entry becomes lower (worse)
    """
    slippage_amount = _get_slippage_amount(
        config.mode, config.entry_slippage, config.entry_slippage_pct,
        config.atr_multiplier_entry, price, atr
    )
    
    if direction.upper() == "BUY":
        return round(price + slippage_amount, 2)
    else:
        return round(price - slippage_amount, 2)


def apply_exit_slippage(
    price: float,
    direction: str,
    config: SlippageConfig,
    atr: Optional[float] = None,
) -> float:
    """
    Apply adverse slippage to an exit/profit-taking price.
    
    For BUY exit (sell to close): exit becomes lower (worse)
    For SELL exit (buy to close): exit becomes higher (worse)
    """
    slippage_amount = _get_slippage_amount(
        config.mode, config.exit_slippage, config.exit_slippage_pct,
        config.atr_multiplier_exit, price, atr
    )
    
    if direction.upper() == "BUY":
        return round(price - slippage_amount, 2)
    else:
        return round(price + slippage_amount, 2)


def apply_stop_slippage(
    stop_price: float,
    direction: str,
    config: SlippageConfig,
    atr: Optional[float] = None,
) -> float:
    """
    Apply adverse slippage to a stop-loss price.
    
    For BUY stop (sell to close): stop triggers lower (worse - more loss)
    For SELL stop (buy to close): stop triggers higher (worse - more loss)
    """
    slippage_amount = _get_slippage_amount(
        config.mode, config.stop_slippage, config.stop_slippage_pct,
        config.atr_multiplier_stop, stop_price, atr
    )
    
    if direction.upper() == "BUY":
        return round(stop_price - slippage_amount, 2)
    else:
        return round(stop_price + slippage_amount, 2)


def apply_target_slippage(
    target_price: float,
    direction: str,
    config: SlippageConfig,
    atr: Optional[float] = None,
) -> float:
    """
    Apply adverse slippage to a target price.
    
    For BUY target (sell to close): target price lower (worse)
    For SELL target (buy to close): target price higher (worse)
    """
    slippage_amount = _get_slippage_amount(
        config.mode, config.target_slippage, config.target_slippage_pct,
        config.atr_multiplier_target, target_price, atr
    )
    
    if direction.upper() == "BUY":
        return round(target_price - slippage_amount, 2)
    else:
        return round(target_price + slippage_amount, 2)


def apply_force_squareoff_slippage(
    price: float,
    direction: str,
    config: SlippageConfig,
    atr: Optional[float] = None,
) -> float:
    """Apply slippage for forced square-off at session end."""
    slippage_amount = _get_slippage_amount(
        config.mode, config.force_squareoff_slippage,
        config.force_squareoff_slippage_pct,
        config.atr_multiplier_force_squareoff, price, atr
    )
    
    if direction.upper() == "BUY":
        return round(price - slippage_amount, 2)
    else:
        return round(price + slippage_amount, 2)


def _get_slippage_amount(
    mode: str,
    fixed_amount: float,
    pct_amount: float,
    atr_mult: float,
    price: float,
    atr: Optional[float] = None,
) -> float:
    """Get slippage amount based on mode."""
    if mode == "percentage":
        return price * pct_amount
    elif mode == "atr_based" and atr is not None:
        return atr * atr_mult
    else:
        return fixed_amount


def compute_executed_prices(
    entry_requested: float,
    stop_requested: float,
    target_requested: float,
    direction: str,
    config: Optional[SlippageConfig] = None,
    atr: Optional[float] = None,
) -> dict:
    """
    Compute all executed prices after slippage.
    """
    if config is None:
        config = SlippageConfig()

    entry_exec = apply_entry_slippage(entry_requested, direction, config, atr)
    stop_exec = apply_stop_slippage(stop_requested, direction, config, atr)
    target_exec = apply_target_slippage(target_requested, direction, config, atr)

    return {
        "entry_executed": entry_exec,
        "stop_executed": stop_exec,
        "target_executed": target_exec,
        "slippage_entry_points": round(abs(entry_exec - entry_requested), 2),
        "slippage_stop_points": round(abs(stop_exec - stop_requested), 2),
        "slippage_target_points": round(abs(target_exec - target_requested), 2),
    }
