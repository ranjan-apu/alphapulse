-- AlphaPulse runtime schema split.
--
-- historical: current historical replay / backtest state.
-- live: future live trading state, intentionally empty after migration.
--
-- This migration assumes 001_initial_schema.sql and 002_audit_schema.sql have
-- already been applied once. Existing public tables are moved into historical
-- when historical does not already contain that table. The live schema is then
-- created from the historical schema structure.

CREATE SCHEMA IF NOT EXISTS historical;
CREATE SCHEMA IF NOT EXISTS live;

DO $$
DECLARE
    table_names TEXT[] := ARRAY[
        'stock_metadata',
        'data_snapshot_sets',
        'data_snapshots',
        'runs',
        'experiment_runs',
        'portfolio_snapshots',
        'decisions',
        'positions',
        'orders_simulated',
        'session_maps',
        'session_levels',
        'session_events',
        'memory_episodes',
        'memory_reflections',
        'calibration_stats',
        'trade_locks',
        'agent_checkpoints',
        'agent_turn_records',
        'tool_call_traces',
        'audit_events',
        'trade_events'
    ];
    table_name TEXT;
BEGIN
    FOREACH table_name IN ARRAY table_names LOOP
        IF to_regclass(format('historical.%I', table_name)) IS NULL THEN
            IF to_regclass(format('public.%I', table_name)) IS NOT NULL THEN
                EXECUTE format('ALTER TABLE public.%I SET SCHEMA historical', table_name);
            ELSE
                RAISE EXCEPTION
                    'Missing source table %. Run 001_initial_schema.sql and 002_audit_schema.sql before 003_runtime_schemas.sql.',
                    table_name;
            END IF;
        END IF;
    END LOOP;
END $$;

DO $$
DECLARE
    table_names TEXT[] := ARRAY[
        'stock_metadata',
        'data_snapshot_sets',
        'data_snapshots',
        'runs',
        'experiment_runs',
        'portfolio_snapshots',
        'decisions',
        'positions',
        'orders_simulated',
        'session_maps',
        'session_levels',
        'session_events',
        'memory_episodes',
        'memory_reflections',
        'calibration_stats',
        'trade_locks',
        'agent_checkpoints',
        'agent_turn_records',
        'tool_call_traces',
        'audit_events',
        'trade_events'
    ];
    table_name TEXT;
BEGIN
    FOREACH table_name IN ARRAY table_names LOOP
        IF to_regclass(format('live.%I', table_name)) IS NULL THEN
            EXECUTE format(
                'CREATE TABLE live.%I (LIKE historical.%I INCLUDING ALL)',
                table_name,
                table_name
            );
        END IF;
    END LOOP;
END $$;

DO $$
DECLARE
    fk RECORD;
    constraint_sql TEXT;
BEGIN
    FOR fk IN
        SELECT
            con.conname,
            rel.relname AS table_name,
            pg_get_constraintdef(con.oid) AS definition
        FROM pg_constraint con
        JOIN pg_class rel ON rel.oid = con.conrelid
        JOIN pg_namespace nsp ON nsp.oid = rel.relnamespace
        WHERE nsp.nspname = 'historical'
          AND con.contype = 'f'
    LOOP
        IF NOT EXISTS (
            SELECT 1
            FROM pg_constraint live_con
            JOIN pg_class live_rel ON live_rel.oid = live_con.conrelid
            JOIN pg_namespace live_nsp ON live_nsp.oid = live_rel.relnamespace
            WHERE live_nsp.nspname = 'live'
              AND live_rel.relname = fk.table_name
              AND live_con.conname = fk.conname
        ) THEN
            constraint_sql := replace(fk.definition, 'REFERENCES historical.', 'REFERENCES live.');
            EXECUTE format(
                'ALTER TABLE live.%I ADD CONSTRAINT %I %s',
                fk.table_name,
                fk.conname,
                constraint_sql
            );
        END IF;
    END LOOP;
END $$;

CREATE INDEX IF NOT EXISTS idx_live_decisions_run_time ON live.decisions(run_id, decision_time);
CREATE INDEX IF NOT EXISTS idx_live_decisions_symbol_time ON live.decisions(symbol, decision_time);
CREATE INDEX IF NOT EXISTS idx_live_positions_run_active ON live.positions(run_id, active);
CREATE INDEX IF NOT EXISTS idx_live_positions_symbol_active ON live.positions(symbol, active);
CREATE INDEX IF NOT EXISTS idx_live_orders_run ON live.orders_simulated(run_id);
CREATE INDEX IF NOT EXISTS idx_live_orders_status ON live.orders_simulated(order_status);
CREATE INDEX IF NOT EXISTS idx_live_portfolio_run_time ON live.portfolio_snapshots(run_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_live_session_events_time ON live.session_events(session_id, event_time);
CREATE INDEX IF NOT EXISTS idx_live_session_levels_session ON live.session_levels(session_id);
CREATE INDEX IF NOT EXISTS idx_live_memory_episodes_symbol_tags ON live.memory_episodes USING GIN (setup_tags);
CREATE INDEX IF NOT EXISTS idx_live_memory_reflections_tags ON live.memory_reflections USING GIN (tags);
CREATE INDEX IF NOT EXISTS idx_live_trade_locks_run_expires ON live.trade_locks(run_id, expires_at);
CREATE INDEX IF NOT EXISTS idx_live_calibration_run_bucket ON live.calibration_stats(run_id, bucket_key);
CREATE INDEX IF NOT EXISTS idx_live_data_snapshots_set ON live.data_snapshots(set_id);
CREATE INDEX IF NOT EXISTS idx_live_agent_turns_decision ON live.agent_turn_records(decision_id);
CREATE INDEX IF NOT EXISTS idx_live_tool_traces_decision ON live.tool_call_traces(decision_id);
CREATE INDEX IF NOT EXISTS idx_live_audit_events_run ON live.audit_events(run_id, created_at);
CREATE INDEX IF NOT EXISTS idx_live_audit_events_type ON live.audit_events(event_type);
CREATE INDEX IF NOT EXISTS idx_live_trade_events_run ON live.trade_events(run_id, created_at);
CREATE INDEX IF NOT EXISTS idx_live_trade_events_position ON live.trade_events(position_id);
