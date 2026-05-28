# AlphaPulse — Current Application Status

> **Last updated**: 2026-05-28
> **Branch**: `feature/agent-price-action-architecture`
> **Test status**: 82 passing, 0 failing
> **Total codebase**: ~16,200 lines Python + 422 lines SQL + 4,500 lines docs

---

## 1. What This Is

AlphaPulse is a **price-action intraday trading agent** for Indian large-cap equity cash stocks. It reads multi-timeframe market data, maintains a Postgres-backed portfolio/session/memory state, makes state-aware decisions (BUY/SKIP/HOLD/EXIT), and persists everything for evaluation and learning.

**Not a backtester. Not a broker integration.** It's an agent harness — the intelligence layer that decides *what* to trade and *why*.

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        main.py                              │
│                    (Replay Loop)                            │
│  WalkForwardClock → DartAgent.decide() → validate →         │
│  DecisionTransactionService → OutcomeFeedbackService        │
└──────────────────────────┬──────────────────────────────────┘
                           │
         ┌─────────────────┼─────────────────┐
         ▼                 ▼                 ▼
┌────────────────┐ ┌──────────────┐ ┌────────────────┐
│  agent/dart.py │ │ core/tools.py│ │ db/services.py │
│  (LLM Agent)   │ │ (24 Tools)   │ │ (Postgres)     │
└───────┬────────┘ └──────┬───────┘ └───────┬────────┘
        │                 │                  │
        ▼                 ▼                  ▼
┌────────────────┐ ┌──────────────┐ ┌────────────────┐
│ agent/schema.py│ │ core/*.py    │ │ db/repository.py│
│ (Pydantic)     │ │ (Computation)│ │ (SQL Queries)   │
└────────────────┘ └──────────────┘ └────────────────┘
```

---

## 3. Module Map — Every File with Purpose

### 3.1 Agent Layer (`agent/`)

| File | Lines | Purpose | Key Classes/Functions |
|---|---|---|---|
| `agent/dart.py` | 718 | **Main LLM agent** — decision loop with incremental context | `DartAgent.decide()`, `DartAgent._get_new_candles()`, `DartAgent._build_step_user_prompt()`, `DartAgent._build_focused_memory_context()`, `DartAgent._fallback_signal()` |
| `agent/schema.py` | 238 | **Pydantic schemas** — LLM output validation | `AnalysisPlan`, `DartThesis`, `PriceActionChecklist`, `FinalSignal`, `ToolRequest`, `validate_llm_output()`, `signal_to_dict()` |
| `agent/planner.py` | 371 | **Analysis planner** — creates plan before tool use | `AgentPlanner.plan_analysis()`, `AgentPlanner.retrieve_context_memories()`, `AgentPlanner._classify_time_bucket()`, `AgentPlanner._classify_volatility()` |
| `agent/memory.py` | 464 | **4-layer memory** — working, session, episodic, reflection | `WorkingMemory`, `SessionMemory`, `MemoryEpisode`, `MemoryReflection`, `MemoryStore`, `MemoryStore.retrieve_similar_setups()`, `MemoryStore.build_retrieval_query()` |
| `agent/reflection.py` | 243 | **Reflection writer** — post-outcome lessons | `ReflectionWriter.write_reflection()`, `ReflectionWriter._determine_confidence_level()`, `ReflectionWriter._generate_lesson()` |
| `agent/graph.py` | 632 | **LangGraph orchestrator** — graph-based workflow | `create_agent_graph()`, `AgentState`, 12 node functions, 3 conditional edges |
| `agent/prompts.py` | 440 | **System/user prompts** — DART framework | `BASE_SYSTEM_PROMPT`, `STRICT_MODE_PROMPT`, `EXPLORATORY_MODE_PROMPT`, `TOOL_RESULT_PROMPT`, `FINAL_REMINDER`, `build_system_prompt()`, `build_user_prompt()`, `build_system_prompt_from_manager()` |
| `agent/prompt_manager.py` | 703 | **A/B prompt testing** — versioned variants | `PromptVariant`, `PromptManager`, `PromptComparisonResult`, `build_default_variants()` |

### 3.2 Core Computation (`core/`)

| File | Lines | Purpose | Key Classes/Functions |
|---|---|---|---|
| `core/context.py` | 403 | **Market state package** — builds complete context for LLM | `build_market_state_package()`, `format_market_state_for_prompt()`, `format_incremental_candles()` |
| `core/context_window.py` | 297 | **No-lookahead slicing** — completed candles only | `ContextWindowPolicy`, `get_completed_weekly_context()`, `get_completed_daily_context()`, `get_completed_intraday_context()`, `build_context_contract()` |
| `core/tools.py` | 732 | **24 deterministic tools** — LLM can request any | `ToolHarness`, `ToolHarness.execute()`, `ToolHarness.get_tool_descriptions()`, 24 tool methods |
| `core/summarizer.py` | 455 | **Indicators & patterns** — RSI, ATR, swings, levels | `compute_all_indicators()`, `detect_swings()`, `find_levels()`, `detect_pattern()`, `summarize_trend()`, `price_location()` |
| `core/charts.py` | 589 | **Chart generation** — matplotlib visualizations | `plot_micro_5m_chart()`, `plot_decision_zoom_chart()`, `plot_context_dashboard()`, `generate_all_charts()` |
| `core/clock.py` | 114 | **Walk-forward clock** — iterates decision points | `WalkForwardClock.iterate()`, `WalkForwardClock.total_steps()` |
| `core/charges.py` | 176 | **Indian equity charges** — brokerage, STT, GST | `EquityCashCharges`, `EquityCashMISCharges`, `ChargeResult`, `compute_charges()`, `compute_entry_charges()`, `compute_exit_charges()` |
| `core/slippage.py` | 189 | **Slippage model** — fixed/percentage/ATR | `SlippageConfig`, `apply_entry_slippage()`, `apply_exit_slippage()`, `apply_stop_slippage()`, `apply_target_slippage()`, `compute_executed_prices()` |
| `core/order_simulator.py` | 243 | **Order simulation** — fills orders with slippage+charges | `SimulatedOrder`, `OrderSimulator.simulate_entry_order()`, `OrderSimulator.simulate_exit_order()` |
| `core/position_sizing.py` | 203 | **Risk-based sizing** — risk budget + capital ceiling | `PositionSizingConfig`, `SizingResult`, `compute_position_size()` |
| `core/portfolio_state.py` | 324 | **In-memory portfolio** — for LLM inspection | `PortfolioState`, `OpenPosition`, `PortfolioStateManager`, `get_portfolio_state_tool()`, `get_open_position_tool()` |
| `core/position.py` | 164 | **Redis position tracker** (legacy) | `PositionTracker` — used by old replay path |
| `core/session_controller.py` | 171 | **Session phases** — 6 phases, entry cutoff | `SessionPhase`, `SessionConfig`, `MarketSessionController.get_phase()`, `can_open_new_position()`, `must_square_off()` |
| `core/cooldown.py` | 258 | **Cooldown & trade locks** — revenge trade prevention | `CooldownController`, `CooldownConfig`, `TradeLock`, `can_open_position()`, `add_lock()`, `record_loss()`, `record_win()` |
| `core/session_levels.py` | 496 | **Level lifecycle** — 8 states, deterministic transitions | `LevelLifecycleManager.process_candle()`, `SessionRebuilder.rebuild_from_events()`, `LevelState` enum |
| `core/market_structure.py` | 380 | **BOS/CHOCH detection** — swing analysis | `detect_market_structure()`, `MarketStructure`, `SwingPoint` |
| `core/vwap.py` | 188 | **Session VWAP** — compute + bands | `compute_session_vwap()`, `VWAPResult` |
| `core/volume_profile.py` | 240 | **Volume profile** — POC, VAH, VAL | `compute_volume_profile()`, `VolumeProfileResult`, `compare_prior_value_area()` |
| `core/confluence.py` | 267 | **Multi-factor scoring** — quality assessment | `score_confluence()`, `score_trade_location()`, `ConfluenceScore`, `TradeLocation` |
| `core/regime.py` | 226 | **Market regime** — trend/range/volatile | `detect_market_regime()`, `classify_time_bucket()`, `classify_volatility()` |
| `core/gap_context.py` | 182 | **Gap classification** — 10 gap types | `classify_gap()`, `GapContext`, `get_gap_context_dict()` |
| `core/calibration.py` | 437 | **Calibration tracking** — confidence buckets | `CalibrationTracker`, `CalibrationBucket`, `record_outcome()`, `get_calibration_hints()` |
| `core/data_snapshot.py` | 309 | **Data versioning** — SHA-256 hashing | `DataSnapshotManager`, `DataSnapshotSet`, `_hash_ohlcv()`, `compare_sets()` |
| `core/stock_metadata.py` | 268 | **Stock reference data** — corporate actions | `StockMetadataManager`, `StockMetadata`, `populate_from_yfinance()`, `adjust_prices()` |

### 3.3 Database (`db/`)

| File | Lines | Purpose | Key Classes/Functions |
|---|---|---|---|
| `db/services.py` | 948 | **Postgres services** — all business logic writes | `RunBootstrapService.create_or_resume_run()`, `DecisionTransactionService.process_decision()`, `OutcomeFeedbackService.record_feedback()`, `SessionStateService.init_session_if_needed()`, `ReplayStateService` |
| `db/repository.py` | 313 | **SQL queries** — table CRUD | `RunRepository`, `PortfolioRepository`, `PositionRepository`, `OrderRepository`, `DecisionRepository`, `SessionRepository`, `MemoryRepository`, `CalibrationRepository`, `TradeLockRepository`, `SnapshotRepository` |
| `db/unit_of_work.py` | 101 | **Transaction management** — single commit per decision | `UnitOfWork.__enter__()`, `UnitOfWork.__exit__()`, `UnitOfWork.commit()`, `UnitOfWork.rollback()` |
| `db/connection.py` | 60 | **Connection pool** — psycopg2 | `get_connection_pool()`, `get_connection()`, `test_connection()` |
| `db/migrations/001_initial_schema.sql` | 422 | **17 tables** — full Postgres schema | Tables: runs, portfolio_snapshots, positions, orders_simulated, decisions, session_maps, session_events, session_levels, memory_episodes, memory_reflections, calibration_stats, experiment_runs, data_snapshot_sets, data_snapshots, stock_metadata, agent_checkpoints, trade_locks |

### 3.4 Validation (`validation/`)

| File | Lines | Purpose | Key Classes/Functions |
|---|---|---|---|
| `validation/validator.py` | 287 | **Signal validator** — state-aware, risk-based | `TradeValidator.validate()`, `validate_signal()`, tool policy checks, session constraints |

### 3.5 Data Collection (`data/`)

| File | Lines | Purpose | Key Classes/Functions |
|---|---|---|---|
| `data/collector.py` | 249 | **yfinance data** — fetch + cache | `fetch_5m_data()`, `fetch_daily_data()`, `fetch_weekly_data()`, `collect_all_data()`, `load_cached_data()` |

### 3.6 Configuration & Entry Point

| File | Lines | Purpose | Key Classes/Functions |
|---|---|---|---|
| `config.py` | 176 | **All config** — env vars, defaults | `Config` class, 50+ fields |
| `main.py` | 703 | **Replay loop** — main entry point | `run_replay()`, `setup()`, `load_data()`, `main()` |

### 3.7 Tests (`tests/`)

| File | Lines | Tests | Purpose |
|---|---|---|---|
| `tests/test_new_modules.py` | 753 | 42 | Unit tests for all core modules |
| `tests/test_integration.py` | 571 | 26 | Integration tests for data snapshots, session levels, calibration |
| `tests/test_services.py` | 501 | 10 | Postgres service tests (transaction, rollback, charges, feedback) |
| `tests/test_persistence.py` | 111 | 1 | Repository round-trip and rollback test |
| `tests/test_incremental_context.py` | 236 | 1 | Incremental context flow (3 scenarios) |
| `tests/test_agent_workflow.py` | 101 | 1 | Memory retrieval tied to analysis plan |

---

## 4. Current Test Status

```
82 passed, 0 failed, 14 warnings in 1.55s
```

### Test Coverage by Area

| Area | Tests | Status |
|---|---|---|
| Context window (no-lookahead) | 4 | ✅ |
| Charges model | 3 | ✅ |
| Slippage model | 4 | ✅ |
| Position sizing | 4 | ✅ |
| State-aware validator | 4 | ✅ |
| Session controller | 3 | ✅ |
| Cooldown | 3 | ✅ |
| Gap context | 2 | ✅ |
| Schema validation | 6 | ✅ |
| VWAP | 1 | ✅ |
| Volume profile | 1 | ✅ |
| Market structure | 1 | ✅ |
| Regime detection | 1 | ✅ |
| Confluence scoring | 2 | ✅ |
| Order simulator | 2 | ✅ |
| Data snapshots | 5 | ✅ |
| Stock metadata | 6 | ✅ |
| Session levels | 5 | ✅ |
| Calibration | 4 | ✅ |
| Cooldown with run_id | 3 | ✅ |
| Schema extended | 2 | ✅ |
| Context extended | 1 | ✅ |
| Repository persistence | 1 | ✅ |
| Services (transaction, rollback) | 6 | ✅ |
| Incremental context | 1 | ✅ |
| Agent workflow | 1 | ✅ |

---

## 5. What's Implemented (Plan Compliance: 94.7%)

### Phase 1 — State, Context, Action Semantics ✅
- Postgres schema (17 tables)
- ContextWindowPolicy (no lookahead)
- Completed daily/weekly/intraday context
- Charges model (CNC + MIS)
- Slippage model (fixed/percentage/ATR)
- Order simulator (slippage → charges → fill)
- Risk-based position sizing
- Data snapshot versioning
- Stock metadata
- Portfolio state (Postgres-backed)
- Session controller (6 phases)
- Cooldown / trade locks
- Action semantics (BUY/SKIP/HOLD/EXIT)
- State-aware validator

### Phase 2 — Structured Output ✅
- Pydantic schemas (AnalysisPlan, FinalSignal, etc.)
- Schema validation wrapper
- Analysis plan enforcement in round 0

### Phase 3 — Memory Layer ✅
- 4 memory layers (working, session, episodic, reflection)
- Postgres persistence
- Weighted feature similarity retrieval
- 19 setup tags

### Phase 4 — Feedback + Reflection ✅
- ReflectionWriter with confidence gates
- OutcomeFeedbackService (episodes, reflections, calibration)
- Memory episodes with 12 structured fields

### Phase 5 — Price Action Tools ✅
- VWAP, Volume Profile, Market Structure, Regime, Confluence
- 24 tools in ToolHarness

### Phase 6 — Prompt v2 ✅
- Full price-action workflow (A-H)
- Scoring rubric (6 dimensions)
- 5 example signals
- State-aware action vocabulary

### Phase 7 — Agentic Planner ✅
- AnalysisPlan with 15 fields
- Deterministic memory retrieval from plan
- LangGraph orchestrator (behind feature flag)
- Tool policy enforcement

### Phase 8 — Calibration + Experiments ✅
- Confidence buckets, setup tags
- Three-tier fallback policy
- Prompt A/B testing (3 variants)
- Incremental context (full history first, deltas after)

---

## 6. What's NOT Implemented (5.3% gap)

| Gap | Section | Priority |
|---|---|---|
| `detect_supply_demand_zones` tool | §7.2 | Low |
| `get_recent_signal_outcomes` tool | §7.7 | Low |
| `classify_session_type` as standalone tool | §7.4 | Low |
| `min_setups_per_tag` configurable threshold | §6.9 | Low |
| `confidence_interval_required` calibration | §6.9 | Low |
| Circuit breaker (LLM failure) | §14.2 | Medium |
| Postgres failure handling | §14.3 | Medium |

---

## 7. How to Run

### Prerequisites
- Python 3.9+
- Postgres 16 (port 5433)
- LLM API key (DeepSeek or OpenRouter)

### Setup
```bash
# 1. Start Postgres
docker run -d --name alphapulse-pg \
  -e POSTGRES_USER=alphapulse \
  -e POSTGRES_PASSWORD=alphapulse \
  -e POSTGRES_DB=alphapulse \
  -p 5433:5432 \
  postgres:16

# 2. Apply schema
psql -h localhost -p 5433 -U alphapulse -d alphapulse \
  -f db/migrations/001_initial_schema.sql

# 3. Configure
cp .env.example .env
# Edit .env with API key, symbol, etc.

# 4. Run tests
venv/bin/python -m pytest tests/ -v

# 5. Run replay
venv/bin/python main.py
```

### Config Defaults (`.env`)
```
SYMBOL=RELIANCE
DECISION_INTERVAL=15min
DECISION_MODE=exploratory
MAX_TOOL_CALLS_PER_DECISION=3
LLM_MODEL_NAME=deepseek-v4-pro
LLM_VISION_ENABLED=true
```

---

## 8. Key Design Decisions

1. **Postgres is source of truth** — Redis only for legacy `PositionTracker`, all new state through `UnitOfWork`
2. **Single transaction per decision** — `DecisionTransactionService.process_decision()` runs inside `with UnitOfWork()`
3. **Deterministic tools own math** — LLM proposes, tools compute, validator checks
4. **No lookahead** — `ContextWindowPolicy` ensures completed candles only
5. **Incremental context** — Full history first prompt, delta updates after (saves ~39% tokens)
6. **State-aware actions** — Flat=BUY/SKIP, Open=HOLD/EXIT, SELL rejected for CNC
7. **LangGraph behind feature flag** — Default is direct workflow, graph available via `AGENT_WORKFLOW=graph`

---

## 9. Data Flow per Decision

```
WalkForwardClock.iterate()
  → DartAgent.decide(market_state_package, text, harness)
    → [First] Full context: 75+22+13 candles + portfolio + session + memory + charts
    → [Later] Incremental: new candles only + step prompt (indicators/levels)
    → Round 0: AnalysisPlan validation + focused memory retrieval
    → Round 1-N: Tool execution via ToolHarness
    → Final: Parse FinalSignal JSON
  → validate_signal() — state-aware, risk-based, session-constrained
  → DecisionTransactionService.process_decision()
    → OrderSimulator.simulate_entry_order/exit_order()
    → Position sizing + charge computation
    → Single Postgres transaction: snapshot → position → decision → order
  → OutcomeFeedbackService.record_feedback() (after replay)
    → Memory episode + reflection + calibration stats
```

---

## 10. Documentation Files

| File | Purpose |
|---|---|
| `docs/agent-price-action-evolution-plan.md` | Original 2,336-line plan — all 8 phases |
| `docs/llm-context-contract.md` | What the LLM receives — incremental architecture |
| `docs/future-architecture.md` | Future vision beyond first harness |
| `docs/application-status.md` | This file — current state + code map |
