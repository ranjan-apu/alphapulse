"""
Indian equity cash market charges model (CNC delivery and MIS intraday).
Deterministic charge calculation - NOT embedded in LLM prompts.

Includes:
- EquityCashCharges (CNC delivery)
- EquityCashMISCharges (MIS intraday)
- compute_charges function
- ChargeResult data class
"""
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class EquityCashCharges:
    """Indian equity cash market charges. Delivery (CNC) rates."""
    # Flat per-order charges
    brokerage_per_order: float = 20.0       # ₹20 per executed order
    # Ad-valorem charges (as fractions, e.g. 0.001 = 0.1%)
    stt_buy: float = 0.001                  # 0.1% on buy side (delivery)
    stt_sell: float = 0.001                 # 0.1% on sell side (delivery)
    exchange_txn_charge: float = 0.0000345  # 0.00345% NSE
    sebi_fee_per_crore: float = 10.0        # ₹10 per crore turnover
    stamp_duty_buy: float = 0.00015         # 0.015% on buy side (delivery)
    stamp_duty_sell: float = 0.0            # 0% on sell side (delivery)
    gst_rate: float = 0.18                  # 18% on (brokerage + exchange charges)


@dataclass
class EquityCashMISCharges:
    """Intraday (MIS) charges - lower STT and stamp duty."""
    brokerage_per_order: float = 20.0
    stt_buy: float = 0.0                    # 0% on buy side (MIS)
    stt_sell: float = 0.00025               # 0.025% on sell side (MIS)
    exchange_txn_charge: float = 0.0000345
    sebi_fee_per_crore: float = 10.0
    stamp_duty_buy: float = 0.00003         # 0.003% on buy side (MIS)
    stamp_duty_sell: float = 0.0
    gst_rate: float = 0.18


@dataclass
class ChargeResult:
    """Result of a charge calculation."""
    total_charges: float
    breakeven_points: float  # how many points price must move to cover charges
    net_r_adjustment: float  # subtract this from gross reward when computing net R
    breakdown: Dict[str, float] = field(default_factory=dict)


def compute_charges(
    charges: EquityCashCharges,
    direction: str,          # 'BUY' or 'SELL'
    quantity: int,
    entry_price: float,
    exit_price: float,
) -> ChargeResult:
    """
    Compute all charges for a round-trip trade.

    Args:
        charges: The charges model (CNC or MIS)
        direction: 'BUY' or 'SELL'
        quantity: Number of shares
        entry_price: Entry execution price
        exit_price: Exit execution price

    Returns:
        ChargeResult with total charges, breakeven, and breakdown.
    """
    turnover = quantity * (entry_price + exit_price)

    # Brokerage: flat per order, both entry and exit
    brokerage = charges.brokerage_per_order * 2  # entry + exit

    # STT: buy side + sell side
    stt = (
        quantity * entry_price * charges.stt_buy
        + quantity * exit_price * charges.stt_sell
    )

    # Exchange transaction charge
    exchange = turnover * charges.exchange_txn_charge

    # SEBI fee (₹10 per crore of turnover)
    turnover_crores = turnover / 10_000_000
    sebi = turnover_crores * charges.sebi_fee_per_crore

    # Stamp duty: only on buy side for delivery, both sides factored
    stamp = (
        quantity * entry_price * charges.stamp_duty_buy
        + quantity * exit_price * charges.stamp_duty_sell
    )

    # GST: 18% on (brokerage + exchange charges)
    gst = charges.gst_rate * (brokerage + exchange)

    total = brokerage + stt + exchange + sebi + stamp + gst

    # Breakeven: how many points per share to cover charges
    breakeven = total / quantity if quantity > 0 else 0

    breakdown = {
        "brokerage": round(brokerage, 2),
        "stt": round(stt, 2),
        "exchange_txn": round(exchange, 2),
        "sebi": round(sebi, 2),
        "stamp_duty": round(stamp, 2),
        "gst": round(gst, 2),
    }

    return ChargeResult(
        total_charges=round(total, 2),
        breakeven_points=round(breakeven, 2),
        net_r_adjustment=round(breakeven, 2),
        breakdown=breakdown,
    )


def compute_entry_charges(
    charges: EquityCashCharges,
    direction: str,
    quantity: int,
    entry_price: float,
) -> Dict[str, float]:
    """Compute charges for just the entry leg."""
    entry_turnover = quantity * entry_price

    brokerage = charges.brokerage_per_order
    stt = quantity * entry_price * charges.stt_buy
    exchange = entry_turnover * charges.exchange_txn_charge
    sebi = (entry_turnover / 10_000_000) * charges.sebi_fee_per_crore
    stamp = quantity * entry_price * charges.stamp_duty_buy
    gst = charges.gst_rate * (brokerage + exchange)

    total = brokerage + stt + exchange + sebi + stamp + gst

    return {
        "brokerage": round(brokerage, 2),
        "stt": round(stt, 2),
        "exchange_txn": round(exchange, 2),
        "sebi": round(sebi, 2),
        "stamp_duty": round(stamp, 2),
        "gst": round(gst, 2),
        "total": round(total, 2),
    }


def compute_exit_charges(
    charges: EquityCashCharges,
    direction: str,
    quantity: int,
    exit_price: float,
) -> Dict[str, float]:
    """Compute charges for just the exit leg."""
    exit_turnover = quantity * exit_price

    brokerage = charges.brokerage_per_order
    stt = quantity * exit_price * charges.stt_sell
    exchange = exit_turnover * charges.exchange_txn_charge
    sebi = (exit_turnover / 10_000_000) * charges.sebi_fee_per_crore
    stamp = quantity * exit_price * charges.stamp_duty_sell
    gst = charges.gst_rate * (brokerage + exchange)

    total = brokerage + stt + exchange + sebi + stamp + gst

    return {
        "brokerage": round(brokerage, 2),
        "stt": round(stt, 2),
        "exchange_txn": round(exchange, 2),
        "sebi": round(sebi, 2),
        "stamp_duty": round(stamp, 2),
        "gst": round(gst, 2),
        "total": round(total, 2),
    }
