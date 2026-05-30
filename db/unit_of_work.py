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
    TradeLockRepository,
    AgentTurnRepository,
    ToolTraceRepository,
    AuditEventRepository,
    TradeEventRepository,
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
        cur.execute("""
            ALTER TABLE decisions
            ADD COLUMN IF NOT EXISTS context_data_hash TEXT,
            ADD COLUMN IF NOT EXISTS prompt_version TEXT,
            ADD COLUMN IF NOT EXISTS model_name TEXT,
            ADD COLUMN IF NOT EXISTS toolset_version TEXT,
            ADD COLUMN IF NOT EXISTS snapshot_set_id TEXT,
            ADD COLUMN IF NOT EXISTS validation_outcome TEXT,
            ADD COLUMN IF NOT EXISTS evaluation_labels JSONB DEFAULT '[]'::jsonb
        """)
        cur.execute("""
            ALTER TABLE experiment_runs
            ADD COLUMN IF NOT EXISTS model_name TEXT,
            ADD COLUMN IF NOT EXISTS prompt_version TEXT,
            ADD COLUMN IF NOT EXISTS toolset_version TEXT
        """)
        # Create 002 audit tables
        _apply_audit_migration(cur)
    conn.commit()
    _schema_checked = True


def _apply_audit_migration(cur) -> None:
    """Apply the 002_audit_schema migration if tables don't exist."""
    cur.execute("""
        CREATE TABLE IF NOT EXISTS agent_turn_records (
            turn_id             TEXT PRIMARY KEY,
            run_id              TEXT NOT NULL REFERENCES experiment_runs(run_id),
            decision_id         TEXT REFERENCES decisions(decision_id),
            turn_number         INTEGER NOT NULL,
            role                TEXT NOT NULL,
            raw_output          TEXT,
            parsed_type         TEXT,
            schema_valid        BOOLEAN DEFAULT TRUE,
            schema_errors       JSONB DEFAULT '[]'::jsonb,
            created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS tool_call_traces (
            trace_id            TEXT PRIMARY KEY,
            run_id              TEXT NOT NULL REFERENCES experiment_runs(run_id),
            decision_id         TEXT REFERENCES decisions(decision_id),
            turn_id             TEXT REFERENCES agent_turn_records(turn_id),
            round_num           INTEGER NOT NULL,
            tool_name           TEXT NOT NULL,
            arguments           JSONB DEFAULT '{}'::jsonb,
            result              JSONB DEFAULT '{}'::jsonb,
            status              TEXT NOT NULL DEFAULT 'success',
            error_message       TEXT,
            latency_ms          DOUBLE PRECISION,
            created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS audit_events (
            event_id            TEXT PRIMARY KEY,
            run_id              TEXT REFERENCES experiment_runs(run_id),
            decision_id         TEXT REFERENCES decisions(decision_id),
            event_type          TEXT NOT NULL,
            severity            TEXT NOT NULL DEFAULT 'info',
            symbol              TEXT,
            message             TEXT NOT NULL,
            details             JSONB DEFAULT '{}'::jsonb,
            created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS trade_events (
            event_id            TEXT PRIMARY KEY,
            run_id              TEXT NOT NULL REFERENCES experiment_runs(run_id),
            decision_id         TEXT REFERENCES decisions(decision_id),
            position_id         TEXT REFERENCES positions(position_id),
            event_type          TEXT NOT NULL,
            symbol              TEXT NOT NULL,
            direction           TEXT,
            price               DOUBLE PRECISION,
            quantity            INTEGER,
            pnl                 DOUBLE PRECISION,
            reason              TEXT DEFAULT '',
            details             JSONB DEFAULT '{}'::jsonb,
            created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_agent_turns_decision ON agent_turn_records(decision_id)
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_tool_traces_decision ON tool_call_traces(decision_id)
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_audit_events_run ON audit_events(run_id, created_at)
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_trade_events_run ON trade_events(run_id, created_at)
    """)


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
        self.agent_turns: Optional[AgentTurnRepository] = None
        self.tool_traces: Optional[ToolTraceRepository] = None
        self.audit: Optional[AuditEventRepository] = None
        self.trade_events: Optional[TradeEventRepository] = None

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
        self.agent_turns = AgentTurnRepository(self.cursor)
        self.tool_traces = ToolTraceRepository(self.cursor)
        self.audit = AuditEventRepository(self.cursor)
        self.trade_events = TradeEventRepository(self.cursor)

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
