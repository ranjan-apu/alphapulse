"""
Risk-based position sizing capped by capital ceiling.

Resolves the conflict between:
- Formula 1: Capital ceiling (qty = floor(max_capital / entry_price))
- Formula 2: Risk budget (qty = floor(risk_budget / risk_per_share))

Resolution: risk-based sizing capped by capital ceiling.
1. Compute risk_per_share = abs(entry_price - stop_price)
2. Compute qty_risk = floor(risk_budget / risk_per_share)
3. Compute qty_capital = floor(max_capital_per_trade / entry_price)
4. quantity = min(qty_risk, qty_capital)
"""
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class PositionSizingConfig:
    """Configuration for position sizing."""
    starting_capital: float = 100000.0      # ₹1,00,000
    risk_budget_pct: float = 0.01           # 1% of starting capital per trade
    max_capital_per_trade: float = 30000.0  # ₹30,000 capital ceiling
    min_net_reward_risk: float = 2.0        # Minimum net R:R to accept

    @property
    def risk_budget(self) -> float:
        """Risk budget in Rupees."""
        return self.starting_capital * self.risk_budget_pct


@dataclass
class SizingResult:
    """Result of position sizing calculation."""
    quantity: int
    deployed_capital: float
    gross_risk: float
    gross_reward: float
    net_reward: float
    net_reward_risk: float
    gross_reward_risk: float
    risk_per_share: float
    reward_per_share: float
    risk_budget_used: float
    capital_ceiling_hit: bool
    risk_budget_hit: bool
    actionable: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def compute_position_size(
    entry_price: float,
    stop_price: float,
    target_price: float,
    direction: str,
    config: Optional[PositionSizingConfig] = None,
    total_charges: Optional[float] = None,
) -> SizingResult:
    """
    Compute position size using risk-based sizing capped by capital ceiling.

    Args:
        entry_price: Entry price per share
        stop_price: Stop-loss price per share
        target_price: Target price per share
        direction: 'BUY' or 'SELL'
        config: Sizing configuration
        total_charges: Estimated round-trip charges (if None, use 0)

    Returns:
        SizingResult with quantity, risk, and validation status.
    """
    if config is None:
        config = PositionSizingConfig()

    errors = []
    warnings = []
    capital_ceiling_hit = False
    risk_budget_hit = False

    # Validate prices
    if entry_price <= 0:
        return SizingResult(
            quantity=0, deployed_capital=0, gross_risk=0, gross_reward=0,
            net_reward=0, net_reward_risk=0, gross_reward_risk=0,
            risk_per_share=0, reward_per_share=0,
            risk_budget_used=0, capital_ceiling_hit=False, risk_budget_hit=False,
            actionable=False, errors=["Entry price must be positive"], warnings=[]
        )

    if stop_price <= 0 or target_price <= 0:
        return SizingResult(
            quantity=0, deployed_capital=0, gross_risk=0, gross_reward=0,
            net_reward=0, net_reward_risk=0, gross_reward_risk=0,
            risk_per_share=0, reward_per_share=0,
            risk_budget_used=0, capital_ceiling_hit=False, risk_budget_hit=False,
            actionable=False, errors=["Stop and target must be positive"], warnings=[]
        )

    # Compute risk and reward per share
    if direction.upper() == "BUY":
        risk_per_share = entry_price - stop_price
        reward_per_share = target_price - entry_price
    else:  # SELL
        risk_per_share = stop_price - entry_price
        reward_per_share = entry_price - target_price

    if risk_per_share <= 0:
        errors.append(f"Stop is on wrong side of entry (risk_per_share={risk_per_share})")
        return SizingResult(
            quantity=0, deployed_capital=0, gross_risk=0, gross_reward=0,
            net_reward=0, net_reward_risk=0, gross_reward_risk=0,
            risk_per_share=risk_per_share, reward_per_share=reward_per_share,
            risk_budget_used=0, capital_ceiling_hit=False, risk_budget_hit=False,
            actionable=False, errors=errors, warnings=warnings
        )

    if reward_per_share <= 0:
        errors.append(f"Target is on wrong side of entry (reward_per_share={reward_per_share})")
        return SizingResult(
            quantity=0, deployed_capital=0, gross_risk=0, gross_reward=0,
            net_reward=0, net_reward_risk=0, gross_reward_risk=0,
            risk_per_share=risk_per_share, reward_per_share=reward_per_share,
            risk_budget_used=0, capital_ceiling_hit=False, risk_budget_hit=False,
            actionable=False, errors=errors, warnings=warnings
        )

    # Step 1: Risk-based quantity
    qty_risk = math.floor(config.risk_budget / risk_per_share) if risk_per_share > 0 else 0

    # Step 2: Capital-ceiling quantity
    qty_capital = math.floor(config.max_capital_per_trade / entry_price) if entry_price > 0 else 0

    # Step 3: Take the minimum
    quantity = min(qty_risk, qty_capital)

    if quantity <= 0:
        errors.append(f"Cannot size position: qty_risk={qty_risk}, qty_capital={qty_capital}")
        return SizingResult(
            quantity=0, deployed_capital=0, gross_risk=0, gross_reward=0,
            net_reward=0, net_reward_risk=0, gross_reward_risk=0,
            risk_per_share=risk_per_share, reward_per_share=reward_per_share,
            risk_budget_used=0, capital_ceiling_hit=False, risk_budget_hit=False,
            actionable=False, errors=errors, warnings=warnings
        )

    # Determine which constraint is binding
    deployed_capital = quantity * entry_price
    gross_risk = quantity * risk_per_share
    gross_reward = quantity * reward_per_share

    if quantity == qty_capital and qty_capital < qty_risk:
        capital_ceiling_hit = True
        warnings.append(f"Capital ceiling ({config.max_capital_per_trade}) limits quantity to {quantity}")
    elif quantity == qty_risk and qty_risk < qty_capital:
        risk_budget_hit = True
        warnings.append(f"Risk budget ({config.risk_budget}) limits quantity to {quantity}")

    # Net reward after charges
    charge_amount = total_charges if total_charges is not None else 0
    net_reward = gross_reward - charge_amount
    net_reward_risk = net_reward / gross_risk if gross_risk > 0 else 0

    # Gross R:R
    gross_reward_risk = reward_per_share / risk_per_share if risk_per_share > 0 else 0

    # Validate net R:R
    meets_rr = net_reward_risk >= config.min_net_reward_risk
    if not meets_rr:
        errors.append(
            f"Net R:R {net_reward_risk:.2f} below minimum {config.min_net_reward_risk}. "
            f"Gross R:R={gross_reward_risk:.2f}, charges={charge_amount}"
        )

    # Check deployed capital
    if deployed_capital > config.max_capital_per_trade:
        errors.append(f"Deployed capital {deployed_capital} exceeds ceiling {config.max_capital_per_trade}")

    # Check risk budget
    if gross_risk > config.risk_budget:
        errors.append(f"Gross risk {gross_risk} exceeds risk budget {config.risk_budget}")

    actionable = len(errors) == 0 and quantity > 0

    return SizingResult(
        quantity=quantity,
        deployed_capital=round(deployed_capital, 2),
        gross_risk=round(gross_risk, 2),
        gross_reward=round(gross_reward, 2),
        net_reward=round(net_reward, 2),
        net_reward_risk=round(net_reward_risk, 4),
        gross_reward_risk=round(gross_reward_risk, 4),
        risk_per_share=round(risk_per_share, 2),
        reward_per_share=round(reward_per_share, 2),
        risk_budget_used=round(gross_risk / config.risk_budget * 100, 1) if config.risk_budget > 0 else 0,
        capital_ceiling_hit=capital_ceiling_hit,
        risk_budget_hit=risk_budget_hit,
        actionable=actionable,
        errors=errors,
        warnings=warnings,
    )
