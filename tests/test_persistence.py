"""
Tests for repository CRUD operations and transactions (UnitOfWork).
"""
import pytest
from datetime import datetime, date, timezone
from db.unit_of_work import UnitOfWork

def test_repository_roundtrip_and_rollback():
    # 1. Roundtrip test for run, experiment_run, portfolio_snapshot, position
    run_id = f"test_run_{int(datetime.now().timestamp())}"
    
    with UnitOfWork() as uow:
        # Create experiment run
        exp_data = {
            "run_id": run_id,
            "symbol": "RELIANCE",
            "start_date": date(2026, 5, 1),
            "end_date": date(2026, 5, 28),
            "starting_capital": 100000.0,
            "max_capital_per_trade": 30000.0,
            "risk_budget_per_trade": 1000.0,
            "max_daily_loss": 3000.0,
            "max_trades_per_day": 5,
            "status": "running"
        }
        uow.runs.save_experiment_run(exp_data)
        
        # Verify get_experiment_run
        fetched = uow.runs.get_experiment_run(run_id)
        assert fetched is not None
        assert fetched["symbol"] == "RELIANCE"
        assert fetched["starting_capital"] == 100000.0
        
        # Save portfolio snapshot
        snapshot_id = f"snap_{run_id}"
        portfolio_data = {
            "snapshot_id": snapshot_id,
            "run_id": run_id,
            "timestamp": datetime.now(timezone.utc),
            "starting_capital": 100000.0,
            "cash_available": 100000.0,
            "capital_deployed": 0.0,
            "realized_pnl": 0.0,
            "max_capital_per_trade": 30000.0,
            "max_daily_loss": 3000.0,
            "trades_taken_today": 0,
        }
        uow.portfolio.save_portfolio_snapshot(portfolio_data)
        
        # Verify latest portfolio snapshot
        latest_portfolio = uow.portfolio.get_latest_snapshot(run_id)
        assert latest_portfolio is not None
        assert latest_portfolio["snapshot_id"] == snapshot_id
        
        # Save position
        position_id = f"pos_{run_id}"
        position_data = {
            "position_id": position_id,
            "run_id": run_id,
            "symbol": "RELIANCE",
            "active": True,
            "direction": "BUY",
            "entry": 2400.0,
            "executed_entry": 2400.5,
            "stop": 2370.0,
            "target": 2460.0,
            "quantity": 12,
            "entry_time": datetime.now(timezone.utc),
        }
        uow.positions.save_position(position_data)
        
        # Verify active position
        active_pos = uow.positions.get_active_position(run_id, "RELIANCE")
        assert active_pos is not None
        assert active_pos["position_id"] == position_id
        assert active_pos["active"] is True
        
        # Update position
        uow.positions.update_position(position_id, {"active": False, "exit_price": 2450.0})
        updated_pos = uow.positions.get_position(position_id)
        assert updated_pos["active"] is False
        assert updated_pos["exit_price"] == 2450.0

    # 2. Transaction rollback test
    failed_run_id = f"test_run_fail_{int(datetime.now().timestamp())}"
    try:
        with UnitOfWork() as uow:
            # Insert a valid experiment run
            exp_data = {
                "run_id": failed_run_id,
                "symbol": "RELIANCE",
                "start_date": date(2026, 5, 1),
                "end_date": date(2026, 5, 28),
                "starting_capital": 100000.0,
                "max_capital_per_trade": 30000.0,
                "risk_budget_per_trade": 1000.0,
                "max_daily_loss": 3000.0,
                "max_trades_per_day": 5,
                "status": "running"
            }
            uow.runs.save_experiment_run(exp_data)
            
            # Force an exception (e.g. key constraint error by inserting duplicate key or raising manual error)
            raise ValueError("Forced transaction rollback")
    except ValueError:
        pass
        
    # Verify the failed run was NOT committed
    with UnitOfWork() as uow:
        fetched = uow.runs.get_experiment_run(failed_run_id)
        assert fetched is None
