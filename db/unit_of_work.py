"""
Unit of Work pattern coordinating database transactions and repository instances.
"""
from typing import Optional
from db.connection import get_connection
from db.repository import (
    RunRepository,
    SnapshotRepository,
    PortfolioRepository,
    PositionRepository,
    OrderRepository,
    DecisionRepository,
    SessionRepository,
    MemoryRepository,
    CalibrationRepository,
    TradeLockRepository
)

_schema_checked = False


def _ensure_runtime_schema(conn) -> None:
    """Apply additive schema fixes needed by older local databases."""
    global _schema_checked
    if _schema_checked:
        return
    with conn.cursor() as cur:
        cur.execute("""
            ALTER TABLE decisions
            ADD COLUMN IF NOT EXISTS outcome_json JSONB DEFAULT '{}'::jsonb,
            ADD COLUMN IF NOT EXISTS outcome_label TEXT,
            ADD COLUMN IF NOT EXISTS outcome_net_r DOUBLE PRECISION,
            ADD COLUMN IF NOT EXISTS mfe_pct DOUBLE PRECISION,
            ADD COLUMN IF NOT EXISTS mae_pct DOUBLE PRECISION
        """)
    conn.commit()
    _schema_checked = True


class UnitOfWork:
    """
    Coordinates transactions and repositories. Use as a context manager.
    Example:
        with UnitOfWork() as uow:
            uow.positions.save_position(...)
        # committed automatically if no exception was raised
    """
    def __init__(self):
        self._conn_ctx = get_connection()
        self.conn = None
        self.cursor = None
        
        # Repositories
        self.runs: Optional[RunRepository] = None
        self.snapshots: Optional[SnapshotRepository] = None
        self.portfolio: Optional[PortfolioRepository] = None
        self.positions: Optional[PositionRepository] = None
        self.orders: Optional[OrderRepository] = None
        self.decisions: Optional[DecisionRepository] = None
        self.sessions: Optional[SessionRepository] = None
        self.memory: Optional[MemoryRepository] = None
        self.calibration: Optional[CalibrationRepository] = None
        self.locks: Optional[TradeLockRepository] = None

    def __enter__(self):
        self.conn = self._conn_ctx.__enter__()
        _ensure_runtime_schema(self.conn)
        self.cursor = self.conn.cursor()
        
        # Initialize repositories with the transaction cursor
        self.runs = RunRepository(self.cursor)
        self.snapshots = SnapshotRepository(self.cursor)
        self.portfolio = PortfolioRepository(self.cursor)
        self.positions = PositionRepository(self.cursor)
        self.orders = OrderRepository(self.cursor)
        self.decisions = DecisionRepository(self.cursor)
        self.sessions = SessionRepository(self.cursor)
        self.memory = MemoryRepository(self.cursor)
        self.calibration = CalibrationRepository(self.cursor)
        self.locks = TradeLockRepository(self.cursor)
        
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if exc_type is not None:
                self.rollback()
            else:
                self.commit()
        finally:
            if self.cursor:
                self.cursor.close()
            self._conn_ctx.__exit__(exc_type, exc_val, exc_tb)

    def commit(self):
        if self.conn:
            self.conn.commit()

    def rollback(self):
        if self.conn:
            self.conn.rollback()
