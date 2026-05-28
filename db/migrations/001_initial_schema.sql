-- AlphaPulse initial schema
-- All tables for portfolio, decisions, session memory, experiments, and data snapshots

-- Enable pgvector extension for embedding-based memory retrieval (Plan Section 4.4.1)
CREATE EXTENSION IF NOT EXISTS vector;

-- ============================================================
-- Stock metadata and corporate actions
-- ============================================================
CREATE TABLE IF NOT EXISTS stock_metadata (
    symbol              TEXT PRIMARY KEY,
    isin                TEXT,
    lot_size            INTEGER NOT NULL DEFAULT 1,
    circuit_limit_upper DOUBLE PRECISION,
    circuit_limit_lower DOUBLE PRECISION,
    adjustment_factor   DOUBLE PRECISION DEFAULT 1.0,
    yahoo_ticker        TEXT,
    expiry_cycle        TEXT,
    is_index            BOOLEAN DEFAULT FALSE,
    earnings_dates      JSONB DEFAULT '[]'::jsonb,
    split_dates         JSONB DEFAULT '[]'::jsonb,
    dividend_dates      JSONB DEFAULT '[]'::jsonb,
    bonus_dates         JSONB DEFAULT '[]'::jsonb,
    notes               TEXT,
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================================
-- Data snapshot versioning for experiment reproducibility
-- ============================================================
CREATE TABLE IF NOT EXISTS data_snapshot_sets (
    set_id              TEXT PRIMARY KEY,
    symbol              TEXT NOT NULL,
    source              TEXT NOT NULL,
    adjusted_for_splits     BOOLEAN DEFAULT FALSE,
    adjusted_for_dividends  BOOLEAN DEFAULT FALSE,
    notes               TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS data_snapshots (
    snapshot_id         TEXT PRIMARY KEY,
    set_id              TEXT NOT NULL REFERENCES data_snapshot_sets(set_id),
    timeframe           TEXT NOT NULL,    -- 'weekly', 'daily', 'intraday_15min'
    period_start        DATE NOT NULL,
    period_end          DATE NOT NULL,
    candle_count        INTEGER,
    first_candle        TIMESTAMPTZ,
    last_candle         TIMESTAMPTZ,
    data_hash           TEXT NOT NULL,    -- SHA-256 of sorted OHLCV rows
    yfinance_period     TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================================
-- Experiment runs
-- ============================================================
CREATE TABLE IF NOT EXISTS experiment_runs (
    run_id              TEXT PRIMARY KEY,
    data_snapshot_set_id TEXT REFERENCES data_snapshot_sets(set_id),
    symbol              TEXT NOT NULL,
    instrument_type     TEXT NOT NULL DEFAULT 'equity_cash',
    product_type        TEXT NOT NULL DEFAULT 'CNC',
    decision_interval   TEXT NOT NULL DEFAULT '15min',
    start_date          DATE NOT NULL,
    end_date            DATE NOT NULL,
    starting_capital    DOUBLE PRECISION NOT NULL DEFAULT 100000.0,
    max_capital_per_trade DOUBLE PRECISION NOT NULL DEFAULT 30000.0,
    risk_budget_per_trade DOUBLE PRECISION NOT NULL DEFAULT 1000.0,
    max_daily_loss      DOUBLE PRECISION NOT NULL DEFAULT 3000.0,
    max_trades_per_day  INTEGER NOT NULL DEFAULT 5,
    agent_version       TEXT,
    prompt_version      TEXT,
    toolset_version     TEXT,
    memory_mode         TEXT,
    model_name          TEXT,
    temperature         DOUBLE PRECISION,
    decision_mode       TEXT DEFAULT 'exploratory',
    random_seed         INTEGER,
    status              TEXT DEFAULT 'running',
    started_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at        TIMESTAMPTZ,
    metrics             JSONB,
    config_snapshot     JSONB,
    notes               TEXT
);

-- ============================================================
-- Portfolio snapshots (capital ledger)
-- ============================================================
CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    snapshot_id         TEXT PRIMARY KEY,
    run_id              TEXT NOT NULL REFERENCES experiment_runs(run_id),
    decision_id         TEXT,
    timestamp           TIMESTAMPTZ NOT NULL,
    starting_capital    DOUBLE PRECISION NOT NULL,
    cash_available      DOUBLE PRECISION NOT NULL,
    capital_deployed    DOUBLE PRECISION NOT NULL DEFAULT 0,
    capital_reserved    DOUBLE PRECISION NOT NULL DEFAULT 0,
    realized_pnl        DOUBLE PRECISION NOT NULL DEFAULT 0,
    unrealized_pnl      DOUBLE PRECISION NOT NULL DEFAULT 0,
    charges_paid        DOUBLE PRECISION NOT NULL DEFAULT 0,
    max_capital_per_trade DOUBLE PRECISION NOT NULL,
    max_daily_loss      DOUBLE PRECISION NOT NULL,
    daily_loss_used     DOUBLE PRECISION NOT NULL DEFAULT 0,
    trades_taken_today  INTEGER NOT NULL DEFAULT 0,
    max_trades_per_day  INTEGER NOT NULL DEFAULT 5,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================================
-- Decisions
-- ============================================================
CREATE TABLE IF NOT EXISTS decisions (
    decision_id         TEXT PRIMARY KEY,
    run_id              TEXT NOT NULL REFERENCES experiment_runs(run_id),
    symbol              TEXT NOT NULL,
    decision_time       TIMESTAMPTZ NOT NULL,
    current_price       DOUBLE PRECISION,
    context_data_hash   TEXT,
    portfolio_snapshot_before TEXT REFERENCES portfolio_snapshots(snapshot_id),
    portfolio_snapshot_after  TEXT REFERENCES portfolio_snapshots(snapshot_id),
    raw_action          TEXT,
    validated_action    TEXT,
    confidence          DOUBLE PRECISION,
    entry               DOUBLE PRECISION,
    stop                DOUBLE PRECISION,
    target              DOUBLE PRECISION,
    gross_reward_risk   DOUBLE PRECISION,
    net_reward_risk     DOUBLE PRECISION,
    expected_horizon_minutes INTEGER,
    dart_direction      TEXT,
    dart_area           TEXT,
    dart_risk           TEXT,
    dart_trigger        TEXT,
    checklist_json      JSONB,
    reason              TEXT,
    invalidation        TEXT,
    thesis_health       TEXT,
    exit_reason         TEXT,
    suggested_exit_price DOUBLE PRECISION,
    position_id         TEXT,
    is_valid            BOOLEAN,
    rejection_reason    TEXT,
    tool_calls_json     JSONB,
    memory_references   JSONB,
    reflection_ids      JSONB,
    prompt_version      TEXT,
    model_name          TEXT,
    agent_version       TEXT,
    raw_llm_responses   JSONB,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================================
-- Positions
-- ============================================================
CREATE TABLE IF NOT EXISTS positions (
    position_id         TEXT PRIMARY KEY,
    run_id              TEXT NOT NULL REFERENCES experiment_runs(run_id),
    symbol              TEXT NOT NULL,
    instrument_type     TEXT NOT NULL DEFAULT 'equity_cash',
    product_type        TEXT NOT NULL DEFAULT 'CNC',
    active              BOOLEAN NOT NULL DEFAULT TRUE,
    direction           TEXT NOT NULL,       -- 'BUY' or 'SELL'
    entry               DOUBLE PRECISION,   -- requested entry
    executed_entry      DOUBLE PRECISION,   -- after slippage
    stop                DOUBLE PRECISION,
    target              DOUBLE PRECISION,
    quantity            INTEGER NOT NULL,
    entry_time          TIMESTAMPTZ NOT NULL,
    exit_time           TIMESTAMPTZ,
    exit_price          DOUBLE PRECISION,
    exit_reason         TEXT,
    last_price          DOUBLE PRECISION,
    unrealized_pnl      DOUBLE PRECISION DEFAULT 0,
    realized_pnl        DOUBLE PRECISION DEFAULT 0,
    r_multiple_live     DOUBLE PRECISION,
    r_multiple_realized DOUBLE PRECISION,
    status              TEXT DEFAULT 'OPEN', -- OPEN, CLOSED, SQUARED_OFF
    entry_order_id      TEXT,
    exit_order_id       TEXT,
    slippage_entry      DOUBLE PRECISION DEFAULT 0,
    slippage_exit       DOUBLE PRECISION DEFAULT 0,
    charges_entry       DOUBLE PRECISION DEFAULT 0,
    charges_exit        DOUBLE PRECISION DEFAULT 0,
    charges_total       DOUBLE PRECISION DEFAULT 0,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================================
-- Orders simulated (bridges decisions to positions)
-- ============================================================
CREATE TABLE IF NOT EXISTS orders_simulated (
    order_id            TEXT PRIMARY KEY,
    run_id              TEXT NOT NULL REFERENCES experiment_runs(run_id),
    decision_id         TEXT REFERENCES decisions(decision_id),
    position_id         TEXT REFERENCES positions(position_id),
    symbol              TEXT NOT NULL,
    instrument_type     TEXT NOT NULL DEFAULT 'equity_cash',
    product_type        TEXT NOT NULL DEFAULT 'CNC',
    order_side          TEXT NOT NULL,       -- 'BUY' or 'SELL'
    order_type          TEXT NOT NULL,       -- 'ENTRY', 'STOP_LOSS', 'TARGET', 'EXIT', 'FORCED_SQUAREOFF'
    requested_price     DOUBLE PRECISION NOT NULL,
    requested_quantity  INTEGER NOT NULL,
    executed_price      DOUBLE PRECISION,
    executed_quantity   INTEGER,
    slippage_points     DOUBLE PRECISION DEFAULT 0,
    slippage_pct        DOUBLE PRECISION DEFAULT 0,
    charges_brokerage   DOUBLE PRECISION DEFAULT 0,
    charges_stt         DOUBLE PRECISION DEFAULT 0,
    charges_exchange    DOUBLE PRECISION DEFAULT 0,
    charges_sebi        DOUBLE PRECISION DEFAULT 0,
    charges_stamp       DOUBLE PRECISION DEFAULT 0,
    charges_gst         DOUBLE PRECISION DEFAULT 0,
    charges_total       DOUBLE PRECISION DEFAULT 0,
    breakeven_adjustment DOUBLE PRECISION DEFAULT 0,
    order_status        TEXT NOT NULL DEFAULT 'PENDING',
    filled_at           TIMESTAMPTZ,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================================
-- Session memory: maps
-- ============================================================
CREATE TABLE IF NOT EXISTS session_maps (
    session_id          TEXT PRIMARY KEY,
    run_id              TEXT NOT NULL REFERENCES experiment_runs(run_id),
    symbol              TEXT NOT NULL,
    session_date        DATE NOT NULL,
    opening_range_high  DOUBLE PRECISION,
    opening_range_low   DOUBLE PRECISION,
    session_high        DOUBLE PRECISION,
    session_low         DOUBLE PRECISION,
    session_vwap        DOUBLE PRECISION,
    vwap_slope          DOUBLE PRECISION,
    current_poc         DOUBLE PRECISION,
    current_vah         DOUBLE PRECISION,
    current_val         DOUBLE PRECISION,
    gap_classification  TEXT,
    gap_points          DOUBLE PRECISION,
    gap_pct             DOUBLE PRECISION,
    market_regime       TEXT,
    current_bias        TEXT,
    session_type        TEXT,
    cooldown_active     BOOLEAN DEFAULT FALSE,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================================
-- Session levels
-- ============================================================
CREATE TABLE IF NOT EXISTS session_levels (
    level_id            TEXT PRIMARY KEY,
    session_id          TEXT NOT NULL REFERENCES session_maps(session_id),
    run_id              TEXT NOT NULL REFERENCES experiment_runs(run_id),
    price               DOUBLE PRECISION NOT NULL,
    level_type          TEXT NOT NULL,       -- 'support', 'resistance', 'vwap', 'poc', 'vah', 'val', 'swing_high', 'swing_low', 'opening_range_high', 'opening_range_low'
    state               TEXT NOT NULL DEFAULT 'ACTIVE',
                                            -- ACTIVE, TESTED, REJECTED, BROKEN,
                                            -- FLIPPED_SUPPORT, FLIPPED_RESISTANCE,
                                            -- INVALIDATED, EXPIRED
    strength            INTEGER DEFAULT 0,   -- touch count, recency, confluence
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================================
-- Session events (for warm-start rebuild)
-- ============================================================
CREATE TABLE IF NOT EXISTS session_events (
    event_id            TEXT PRIMARY KEY,
    session_id          TEXT NOT NULL REFERENCES session_maps(session_id),
    run_id              TEXT NOT NULL REFERENCES experiment_runs(run_id),
    event_time          TIMESTAMPTZ NOT NULL,
    event_type          TEXT NOT NULL,       -- 'LEVEL_IDENTIFIED', 'LEVEL_TESTED', 'LEVEL_BROKEN',
                                            -- 'LEVEL_REJECTED', 'LEVEL_FLIPPED', 'LEVEL_INVALIDATED',
                                            -- 'VWAP_RECLAIM', 'VWAP_REJECTION', 'GAP_FILLED',
                                            -- 'REGIME_CHANGE', 'COOLDOWN_START', 'COOLDOWN_END',
                                            -- 'POSITION_OPENED', 'POSITION_CLOSED', 'STOP_HIT', 'TARGET_HIT'
    event_data          JSONB NOT NULL,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================================
-- Memory: episodic (past trades and outcomes)
-- ============================================================
CREATE TABLE IF NOT EXISTS memory_episodes (
    episode_id          TEXT PRIMARY KEY,
    run_id              TEXT REFERENCES experiment_runs(run_id),
    symbol              TEXT NOT NULL,
    decision_id         TEXT REFERENCES decisions(decision_id),
    action              TEXT NOT NULL,
    direction           TEXT,
    market_regime       TEXT,
    session_type        TEXT,
    gap_type            TEXT,
    structure_state     TEXT,
    vwap_relation       TEXT,
    vwap_distance_atr   DOUBLE PRECISION,
    profile_location    TEXT,
    price_location      TEXT,
    time_bucket         TEXT,
    volatility_bucket   TEXT,
    setup_tags          JSONB DEFAULT '[]'::jsonb,
    outcome_net_r       DOUBLE PRECISION,
    outcome_label       TEXT,               -- 'win', 'loss', 'breakeven', 'ambiguous'
    mfe_pct             DOUBLE PRECISION,
    mae_pct             DOUBLE PRECISION,
    thesis_json         JSONB,
    mistakes            JSONB DEFAULT '[]'::jsonb,
    confidence          DOUBLE PRECISION DEFAULT 0,
    sample_quality      INTEGER DEFAULT 1,   -- number of similar episodes backing this
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================================
-- Memory: reflections (learned rules and warnings)
-- ============================================================
CREATE TABLE IF NOT EXISTS memory_reflections (
    reflection_id       TEXT PRIMARY KEY,
    run_id              TEXT REFERENCES experiment_runs(run_id),
    symbol              TEXT NOT NULL,
    lesson              TEXT NOT NULL,
    tags                JSONB DEFAULT '[]'::jsonb,
    source_episode_ids  JSONB DEFAULT '[]'::jsonb,
    direction           TEXT,
    reflection_level    TEXT DEFAULT 'LOW',  -- HIGH, MEDIUM, LOW
    confidence          DOUBLE PRECISION DEFAULT 0,
    num_supporting_episodes INTEGER DEFAULT 1,
    last_updated        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================================
-- Calibration stats
-- ============================================================
CREATE TABLE IF NOT EXISTS calibration_stats (
    stat_id             TEXT PRIMARY KEY,
    run_id              TEXT REFERENCES experiment_runs(run_id),
    bucket_key          TEXT NOT NULL,       -- e.g. 'confidence_0.60-0.70', 'setup_breakout'
    bucket_type         TEXT NOT NULL,       -- 'confidence', 'setup_tag', 'regime', 'session_type'
    total_trades        INTEGER NOT NULL DEFAULT 0,
    wins                INTEGER NOT NULL DEFAULT 0,
    losses              INTEGER NOT NULL DEFAULT 0,
    win_rate            DOUBLE PRECISION,
    avg_net_r           DOUBLE PRECISION,
    sum_net_r           DOUBLE PRECISION DEFAULT 0,
    avg_mfe             DOUBLE PRECISION,
    avg_mae             DOUBLE PRECISION,
    confidence_interval_lower DOUBLE PRECISION,
    confidence_interval_upper DOUBLE PRECISION,
    min_samples_for_hint INTEGER NOT NULL DEFAULT 20,
    last_updated        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================================
-- Trade locks (cooldown / re-entry blocking)
-- ============================================================
CREATE TABLE IF NOT EXISTS trade_locks (
    lock_id             TEXT PRIMARY KEY,
    run_id              TEXT NOT NULL REFERENCES experiment_runs(run_id),
    symbol              TEXT NOT NULL,
    direction           TEXT,
    level_zone          TEXT,
    reason              TEXT NOT NULL,       -- 'AFTER_STOP_LOSS', 'AFTER_TARGET_HIT',
                                            -- 'AFTER_AGENT_EXIT', 'AFTER_REJECTED_SIGNAL',
                                            -- 'SAME_LEVEL_REPEATED'
    expires_at          TIMESTAMPTZ NOT NULL,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================================
-- Agent checkpoints (LangGraph persistence)
-- ============================================================
CREATE TABLE IF NOT EXISTS agent_checkpoints (
    checkpoint_id       TEXT PRIMARY KEY,
    run_id              TEXT NOT NULL REFERENCES experiment_runs(run_id),
    decision_id         TEXT REFERENCES decisions(decision_id),
    node_name           TEXT NOT NULL,
    state_json          JSONB NOT NULL,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================================
-- Indexes
-- ============================================================
CREATE INDEX IF NOT EXISTS idx_decisions_run_time ON decisions(run_id, decision_time);
CREATE INDEX IF NOT EXISTS idx_decisions_symbol_time ON decisions(symbol, decision_time);
CREATE INDEX IF NOT EXISTS idx_positions_run_active ON positions(run_id, active);
CREATE INDEX IF NOT EXISTS idx_positions_symbol_active ON positions(symbol, active);
CREATE INDEX IF NOT EXISTS idx_orders_run ON orders_simulated(run_id);
CREATE INDEX IF NOT EXISTS idx_orders_status ON orders_simulated(order_status);
CREATE INDEX IF NOT EXISTS idx_portfolio_run_time ON portfolio_snapshots(run_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_session_events_time ON session_events(session_id, event_time);
CREATE INDEX IF NOT EXISTS idx_session_levels_session ON session_levels(session_id);
CREATE INDEX IF NOT EXISTS idx_memory_episodes_symbol_tags ON memory_episodes USING GIN (setup_tags);
CREATE INDEX IF NOT EXISTS idx_memory_reflections_tags ON memory_reflections USING GIN (tags);
CREATE INDEX IF NOT EXISTS idx_trade_locks_run_expires ON trade_locks(run_id, expires_at);
CREATE INDEX IF NOT EXISTS idx_calibration_run_bucket ON calibration_stats(run_id, bucket_key);
CREATE INDEX IF NOT EXISTS idx_data_snapshots_set ON data_snapshots(set_id);
