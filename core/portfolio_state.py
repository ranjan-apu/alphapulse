"""
Portfolio and Position state readers for the agent.
Provides deterministic Postgres-backed portfolio state.
The LLM can inspect state but CANNOT mutate it.

Produces compact PortfolioStatePackage for the prompt and exposes
deterministic tools: get_portfolio_state, get_open_position, etc.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any

from config import config


@dataclass
class PortfolioState:
    """Snapshot of portfolio state at a point in time."""
    run_id: str = ""
    starting_capital: float = 100000.0
    cash_available: float = 100000.0
    capital_deployed: float = 0.0
    capital_reserved: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    charges_paid: float = 0.0
    max_capital_per_trade: float = 30000.0
    max_daily_loss: float = 3000.0
    daily_loss_used: float = 0.0
    trades_taken_today: int = 0
    max_trades_per_day: int = 5
    timestamp: Optional[datetime] = None

    def can_trade(self) -> tuple:
        """Check if trading is allowed under current risk limits."""
        errors = []
        if self.trades_taken_today >= self.max_trades_per_day:
            errors.append("Max trades per day reached")
        if abs(self.daily_loss_used) >= self.max_daily_loss:
            errors.append("Max daily loss reached")
        if self.cash_available < self.max_capital_per_trade * 0.5:
            errors.append("Insufficient cash available")
        return len(errors) == 0, errors

    def summary_text(self) -> str:
        """Compact text summary for the agent prompt."""
        return (
            f"Portfolio State:\n"
            f"- Cash available: ₹{self.cash_available:,.0f}\n"
            f"- Capital deployed: ₹{self.capital_deployed:,.0f}\n"
            f"- Realized P&L today: ₹{self.realized_pnl:,.2f}\n"
            f"- Unrealized P&L: ₹{self.unrealized_pnl:,.2f}\n"
            f"- Charges paid today: ₹{self.charges_paid:,.2f}\n"
            f"- Trades today: {self.trades_taken_today} / {self.max_trades_per_day}\n"
            f"- Daily loss used: ₹{abs(self.daily_loss_used):,.0f} / ₹{self.max_daily_loss:,.0f}"
        )


@dataclass
class OpenPosition:
    """Details of an open position."""
    position_id: str = ""
    symbol: str = ""
    direction: str = "BUY"
    entry: float = 0.0
    executed_entry: float = 0.0
    stop: float = 0.0
    target: float = 0.0
    quantity: int = 0
    entry_time: Optional[datetime] = None
    last_price: Optional[float] = None
    unrealized_pnl: float = 0.0
    r_multiple_live: Optional[float] = None
    charges_entry: float = 0.0
    slippage_entry: float = 0.0

    def summary_text(self) -> str:
        """Compact text summary for the agent prompt."""
        return (
            f"Open position: {self.direction} {self.symbol}, "
            f"qty={self.quantity}, entry=₹{self.executed_entry:,.2f}, "
            f"stop=₹{self.stop:,.2f}, target=₹{self.target:,.2f}\n"
            f"Unrealized P&L: ₹{self.unrealized_pnl:,.2f}"
            f"{', Live R: ' + str(self.r_multiple_live) if self.r_multiple_live else ''}"
        )


class PortfolioStateManager:
    """
    Manages portfolio and position state.
    In the harness (no live broker), this is simulated.

    The LLM reads from this but writes go through deterministic code.
    """

    def __init__(
        self,
        run_id: str = "",
        starting_capital: float = 100000.0,
        max_capital_per_trade: float = 30000.0,
        max_daily_loss: float = 3000.0,
        max_trades_per_day: int = 5,
        risk_budget_pct: float = 0.01,
    ):
        self._portfolio = PortfolioState(
            run_id=run_id,
            starting_capital=starting_capital,
            cash_available=starting_capital,
            max_capital_per_trade=max_capital_per_trade,
            max_daily_loss=max_daily_loss,
            max_trades_per_day=max_trades_per_day,
        )
        self._position: Optional[OpenPosition] = None
        self._risk_budget = starting_capital * risk_budget_pct

    # Read operations (safe for LLM inspection)

    def get_portfolio_state(self) -> PortfolioState:
        """Get current portfolio snapshot."""
        return self._portfolio

    def get_open_position(self) -> Optional[OpenPosition]:
        """Get open position details, or None if flat."""
        return self._position

    def has_position(self) -> bool:
        """Check if there's an open position."""
        return self._position is not None

    def get_portfolio_summary_text(self) -> str:
        """Get combined portfolio + position summary for prompt."""
        text = self._portfolio.summary_text()
        if self._position:
            text += "\n\n" + self._position.summary_text()
        else:
            text += "\n\nNo open position."
        return text

    def get_capital_constraints(self) -> dict:
        """Get capital constraints dict for tools."""
        return {
            "cash_available": self._portfolio.cash_available,
            "capital_deployed": self._portfolio.capital_deployed,
            "max_capital_per_trade": self._portfolio.max_capital_per_trade,
            "risk_budget_per_trade": self._risk_budget,
            "max_daily_loss": self._portfolio.max_daily_loss,
            "daily_loss_used": self._portfolio.daily_loss_used,
            "trades_taken_today": self._portfolio.trades_taken_today,
            "max_trades_per_day": self._portfolio.max_trades_per_day,
            "can_trade": self._portfolio.can_trade()[0],
        }

    # Write operations (deterministic code only, not LLM)

    def open_position(
        self,
        position_id: str,
        symbol: str,
        direction: str,
        entry_price: float,
        executed_entry: float,
        stop_price: float,
        target_price: float,
        quantity: int,
        entry_time: datetime,
        charges_entry: float = 0.0,
        slippage_entry: float = 0.0,
    ) -> OpenPosition:
        """Open a new position (deterministic code path only)."""
        deployed = quantity * executed_entry

        self._position = OpenPosition(
            position_id=position_id,
            symbol=symbol,
            direction=direction,
            entry=entry_price,
            executed_entry=executed_entry,
            stop=stop_price,
            target=target_price,
            quantity=quantity,
            entry_time=entry_time,
            charges_entry=charges_entry,
            slippage_entry=slippage_entry,
        )

        self._portfolio.capital_deployed += deployed
        self._portfolio.cash_available -= deployed
        self._portfolio.charges_paid += charges_entry
        self._portfolio.trades_taken_today += 1

        return self._position

    def close_position(
        self,
        exit_price: float,
        charges_exit: float = 0.0,
        slippage_exit: float = 0.0,
        reason: str = "EXIT",
    ) -> Optional[Dict]:
        """Close the open position and update P&L."""
        if not self._position:
            return None

        pos = self._position

        # Calculate realized P&L
        if pos.direction == "BUY":
            gross_pnl = pos.quantity * (exit_price - pos.executed_entry)
        else:
            gross_pnl = pos.quantity * (pos.executed_entry - exit_price)

        total_charges = pos.charges_entry + charges_exit
        net_pnl = gross_pnl - total_charges

        result = {
            "position_id": pos.position_id,
            "exit_price": exit_price,
            "exit_reason": reason,
            "gross_pnl": round(gross_pnl, 2),
            "net_pnl": round(net_pnl, 2),
            "charges_total": round(total_charges, 2),
        }

        # Update portfolio
        self._portfolio.capital_deployed -= pos.quantity * pos.executed_entry
        self._portfolio.cash_available += pos.quantity * exit_price - charges_exit
        self._portfolio.realized_pnl += net_pnl
        self._portfolio.charges_paid += charges_exit
        self._portfolio.daily_loss_used += abs(min(net_pnl, 0))
        self._portfolio.unrealized_pnl = 0.0

        self._position = None
        return result

    def update_market_price(self, last_price: float):
        """Update unrealized P&L with current market price."""
        if not self._position:
            return

        pos = self._position
        pos.last_price = last_price

        if pos.direction == "BUY":
            unrealized = pos.quantity * (last_price - pos.executed_entry)
        else:
            unrealized = pos.quantity * (pos.executed_entry - last_price)

        pos.unrealized_pnl = round(unrealized, 2)
        self._portfolio.unrealized_pnl = round(unrealized, 2)

        # Live R
        risk_per_share = abs(pos.executed_entry - pos.stop)
        if risk_per_share > 0:
            pos.r_multiple_live = round(unrealized / (pos.quantity * risk_per_share), 2)

    def reset_daily_state(self):
        """Reset daily counters for a new session."""
        self._portfolio.trades_taken_today = 0
        self._portfolio.daily_loss_used = 0.0
        self._portfolio.realized_pnl = 0.0
        self._portfolio.charges_paid = 0.0
        self._portfolio.unrealized_pnl = 0.0


# ---- Deterministic portfolio/position tools for the agent ----


def get_portfolio_state_tool(portfolio_mgr: PortfolioStateManager) -> dict:
    """Tool: Return current portfolio state."""
    state = portfolio_mgr.get_portfolio_state()
    return {
        "tool": "get_portfolio_state",
        "result": {
            "cash_available": round(state.cash_available, 2),
            "capital_deployed": round(state.capital_deployed, 2),
            "realized_pnl": round(state.realized_pnl, 2),
            "unrealized_pnl": round(state.unrealized_pnl, 2),
            "charges_paid": round(state.charges_paid, 2),
            "trades_taken_today": state.trades_taken_today,
            "max_trades_per_day": state.max_trades_per_day,
            "daily_loss_used": round(state.daily_loss_used, 2),
            "max_daily_loss": round(state.max_daily_loss, 2),
            "can_trade": state.can_trade()[0],
        },
    }


def get_open_position_tool(portfolio_mgr: PortfolioStateManager) -> dict:
    """Tool: Return current open position details."""
    pos = portfolio_mgr.get_open_position()
    if pos is None:
        return {
            "tool": "get_open_position",
            "result": {
                "has_position": False,
                "message": "No open position.",
            },
        }

    return {
        "tool": "get_open_position",
        "result": {
            "has_position": True,
            "position_id": pos.position_id,
            "symbol": pos.symbol,
            "direction": pos.direction,
            "entry": round(pos.entry, 2),
            "executed_entry": round(pos.executed_entry, 2),
            "stop": round(pos.stop, 2),
            "target": round(pos.target, 2),
            "quantity": pos.quantity,
            "unrealized_pnl": round(pos.unrealized_pnl, 2),
            "r_multiple_live": pos.r_multiple_live,
            "charges_entry": round(pos.charges_entry, 2),
        },
    }


def get_capital_constraints_tool(portfolio_mgr: PortfolioStateManager) -> dict:
    """Tool: Return capital and risk constraints."""
    return {
        "tool": "get_capital_constraints",
        "result": portfolio_mgr.get_capital_constraints(),
    }
