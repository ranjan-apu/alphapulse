-- AlphaPulse audit schema: durable audit, tool traces, trade lifecycle events
-- These tables are additive to 001_initial_schema.sql and can be run independently.

-- ============================================================
-- Reproducibility columns used by replay audit records
-- ============================================================
ALTER TABLE decisions
    ADD COLUMN IF NOT EXISTS toolset_version TEXT,
    ADD COLUMN IF NOT EXISTS snapshot_set_id TEXT,
    ADD COLUMN IF NOT EXISTS validation_outcome TEXT,
    ADD COLUMN IF NOT EXISTS evaluation_labels JSONB DEFAULT '[]'::jsonb;

ALTER TABLE experiment_runs
    ADD COLUMN IF NOT EXISTS model_name TEXT,
    ADD COLUMN IF NOT EXISTS prompt_version TEXT,
    ADD COLUMN IF NOT EXISTS toolset_version TEXT;

-- ============================================================
-- Agent turn records: message-level audit for each LLM interaction
-- ============================================================
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
);

-- ============================================================
-- Tool call traces: structured I/O, latency, status per tool call
-- ============================================================
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
);

-- ============================================================
-- Audit events: engine failures, policy violations, system events
-- ============================================================
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
);

-- ============================================================
-- Trade events: entry, exit, fill, forced square-off, rejection lifecycle
-- ============================================================
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
);

-- ============================================================
-- Indexes for audit tables
-- ============================================================
CREATE INDEX IF NOT EXISTS idx_agent_turns_decision ON agent_turn_records(decision_id);
CREATE INDEX IF NOT EXISTS idx_tool_traces_decision ON tool_call_traces(decision_id);
CREATE INDEX IF NOT EXISTS idx_audit_events_run ON audit_events(run_id, created_at);
CREATE INDEX IF NOT EXISTS idx_audit_events_type ON audit_events(event_type);
CREATE INDEX IF NOT EXISTS idx_trade_events_run ON trade_events(run_id, created_at);
CREATE INDEX IF NOT EXISTS idx_trade_events_position ON trade_events(position_id);
