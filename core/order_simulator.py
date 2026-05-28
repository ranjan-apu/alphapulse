"""
Order simulation: bridges intent (decisions) to execution (positions).
Applies slippage, computes charges, and manages order lifecycle.

Every trade intent becomes at least one order record.
Positions have associated entry (and optionally exit) orders.
"""
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional

from core.charges import (
    EquityCashCharges,
    EquityCashMISCharges,
    compute_charges,
    compute_entry_charges,
    compute_exit_charges,
    ChargeResult,
)
from core.slippage import SlippageConfig, compute_executed_prices


@dataclass
class SimulatedOrder:
    """A simulated order bridging decision to execution."""
    order_id: str
    run_id: str
    decision_id: Optional[str]
    position_id: Optional[str]
    symbol: str
    instrument_type: str = "equity_cash"
    product_type: str = "CNC"
    order_side: str = "BUY"
    order_type: str = "ENTRY"       # ENTRY, STOP_LOSS, TARGET, EXIT, FORCED_SQUAREOFF
    requested_price: float = 0.0
    requested_quantity: int = 0
    executed_price: Optional[float] = None
    executed_quantity: Optional[int] = None
    slippage_points: float = 0.0
    slippage_pct: float = 0.0
    charges_brokerage: float = 0.0
    charges_stt: float = 0.0
    charges_exchange: float = 0.0
    charges_sebi: float = 0.0
    charges_stamp: float = 0.0
    charges_gst: float = 0.0
    charges_total: float = 0.0
    breakeven_adjustment: float = 0.0
    order_status: str = "PENDING"
    filled_at: Optional[datetime] = None
    created_at: datetime = field(default_factory=lambda: datetime.now().astimezone())


class OrderSimulator:
    """
    Simulates order execution with slippage and charge calculation.
    
    The simulator:
    1. Applies slippage to get executed price
    2. Computes charges for the order leg
    3. Creates order records
    4. Links orders to positions via position_id
    """

    def __init__(
        self,
        charges_model: Optional[EquityCashCharges] = None,
        slippage_config: Optional[SlippageConfig] = None,
        atr: Optional[float] = None,
    ):
        self.charges = charges_model or EquityCashCharges()
        self.slippage = slippage_config or SlippageConfig()
        self.atr = atr

    def simulate_entry_order(
        self,
        run_id: str,
        decision_id: str,
        symbol: str,
        direction: str,
        entry_price: float,
        stop_price: float,
        target_price: float,
        quantity: int,
        product_type: str = "CNC",
    ) -> Dict:
        """
        Simulate an entry order with slippage.
        
        Returns dict with:
        - entry_order: SimulatedOrder for entry
        - executed_entry: float (slippage-adjusted)
        - executed_stop: float (slippage-adjusted)
        - executed_target: float (slippage-adjusted)
        - charges_detail: dict with entry charges breakdown
        - position_id: str
        """
        position_id = f"pos_{uuid.uuid4().hex[:12]}"
        entry_order_id = f"ord_entry_{uuid.uuid4().hex[:12]}"

        # Compute executed prices with slippage
        executed = compute_executed_prices(
            entry_price, stop_price, target_price,
            direction, self.slippage, self.atr
        )

        # Compute entry charges
        entry_charges = compute_entry_charges(
            self.charges, direction, quantity, executed["entry_executed"]
        )

        # Create entry order
        entry_order = SimulatedOrder(
            order_id=entry_order_id,
            run_id=run_id,
            decision_id=decision_id,
            position_id=position_id,
            symbol=symbol,
            instrument_type="equity_cash",
            product_type=product_type,
            order_side=direction,
            order_type="ENTRY",
            requested_price=entry_price,
            requested_quantity=quantity,
            executed_price=executed["entry_executed"],
            executed_quantity=quantity,
            slippage_points=executed["slippage_entry_points"],
            slippage_pct=round(executed["slippage_entry_points"] / entry_price * 100, 4),
            charges_brokerage=entry_charges.get("brokerage", 0),
            charges_stt=entry_charges.get("stt", 0),
            charges_exchange=entry_charges.get("exchange_txn", 0),
            charges_sebi=entry_charges.get("sebi", 0),
            charges_stamp=entry_charges.get("stamp_duty", 0),
            charges_gst=entry_charges.get("gst", 0),
            charges_total=entry_charges.get("total", 0),
            order_status="FILLED",
            filled_at=datetime.now().astimezone(),
        )

        return {
            "entry_order": entry_order,
            "position_id": position_id,
            "executed_entry": executed["entry_executed"],
            "executed_stop": executed["stop_executed"],
            "executed_target": executed["target_executed"],
            "slippage_entry_points": executed["slippage_entry_points"],
            "slippage_stop_points": executed["slippage_stop_points"],
            "slippage_target_points": executed["slippage_target_points"],
            "charges_entry": entry_charges,
        }

    def simulate_exit_order(
        self,
        run_id: str,
        position_id: str,
        symbol: str,
        direction: str,
        exit_price: float,
        quantity: int,
        order_type: str = "EXIT",  # EXIT, STOP_LOSS, TARGET, FORCED_SQUAREOFF
        product_type: str = "CNC",
    ) -> Dict:
        """
        Simulate an exit order with slippage.
        
        Returns dict with:
        - exit_order: SimulatedOrder
        - executed_exit: float (slippage-adjusted)
        - charges_detail: dict with exit charges breakdown
        """
        from core.slippage import (
            apply_exit_slippage,
            apply_stop_slippage,
            apply_target_slippage,
            apply_force_squareoff_slippage,
        )

        exit_order_id = f"ord_exit_{uuid.uuid4().hex[:12]}"

        # Apply appropriate slippage based on order type
        if order_type == "STOP_LOSS":
            executed = apply_stop_slippage(exit_price, direction, self.slippage, self.atr)
            slippage = abs(executed - exit_price)
        elif order_type == "TARGET":
            executed = apply_target_slippage(exit_price, direction, self.slippage, self.atr)
            slippage = abs(executed - exit_price)
        elif order_type == "FORCED_SQUAREOFF":
            executed = apply_force_squareoff_slippage(exit_price, direction, self.slippage, self.atr)
            slippage = abs(executed - exit_price)
        else:  # EXIT
            executed = apply_exit_slippage(exit_price, direction, self.slippage, self.atr)
            slippage = abs(executed - exit_price)

        # Determine the SELL side for charge computation
        exit_side = "SELL" if direction == "BUY" else "BUY"
        exit_charges = compute_exit_charges(
            self.charges, exit_side, quantity, executed
        )

        exit_order = SimulatedOrder(
            order_id=exit_order_id,
            run_id=run_id,
            decision_id=None,
            position_id=position_id,
            symbol=symbol,
            instrument_type="equity_cash",
            product_type=product_type,
            order_side=exit_side,
            order_type=order_type,
            requested_price=exit_price,
            requested_quantity=quantity,
            executed_price=executed,
            executed_quantity=quantity,
            slippage_points=round(slippage, 2),
            slippage_pct=round(slippage / exit_price * 100, 4) if exit_price > 0 else 0,
            charges_brokerage=exit_charges.get("brokerage", 0),
            charges_stt=exit_charges.get("stt", 0),
            charges_exchange=exit_charges.get("exchange_txn", 0),
            charges_sebi=exit_charges.get("sebi", 0),
            charges_stamp=exit_charges.get("stamp_duty", 0),
            charges_gst=exit_charges.get("gst", 0),
            charges_total=exit_charges.get("total", 0),
            order_status="FILLED",
            filled_at=datetime.now().astimezone(),
        )

        return {
            "exit_order": exit_order,
            "executed_exit": executed,
            "slippage_exit_points": round(slippage, 2),
            "charges_exit": exit_charges,
        }

    def compute_round_trip_charges(
        self,
        direction: str,
        quantity: int,
        entry_price: float,
        exit_price: float,
    ) -> ChargeResult:
        """Compute total round-trip charges."""
        return compute_charges(self.charges, direction, quantity, entry_price, exit_price)
