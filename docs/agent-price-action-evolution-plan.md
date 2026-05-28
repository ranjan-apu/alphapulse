# AlphaPulse Agent Evolution Plan — Price Action Trading Harness (Single Large-Cap Stock)

> **Status**: Proposed plan  
> **Branch**: `feature/agent-price-action-architecture`  
> **Instrument**: Single Indian large-cap equity cash stock (e.g., RELIANCE, TCS, HDFC Bank, Infosys, ICICI Bank)  
> **Scope**: Agent intelligence, context, tools, memory, feedback, and evaluation. Real-time data ingestion and broker execution are intentionally deferred.  
> **Goal**: Turn the current isolated POC LLM decision loop into a coherent, feedback-aware, portfolio-aware price-action trading agent that can reason like a session trader.

---

## 1. What We Are Building

The immediate objective is **not** broker execution and **not** live market integration.

The immediate objective is to build a strong **agent harness** that can:

1. Read clean multi-timeframe market context at decision time `T`.
2. Maintain a session-level market map.
3. Remember prior predictions, levels, invalidations, and outcomes.
4. Use deterministic tools for market structure, VWAP, volume profile, liquidity, regime, and trade math.
5. Make state-aware structured decisions: BUY / SELL / SKIP when flat, HOLD / EXIT when already in a position.
6. Know portfolio state before every decision: capital available, capital deployed, open position, realized P&L, unrealized P&L, charges, and risk limits.
7. Receive delayed outcome feedback.
8. Improve its next decisions through memory and reflection.
9. Persist decisions, state, and memory through Postgres so the agent has durable continuity across a session. Redis may be used later only as an optional cache, not as the source of truth.
10. Be evaluated across prompts, tools, models, and agent configurations.

The agent should not be treated as a magic future-predictor. It should be treated as a **probabilistic market-structure reasoner** whose edge must be measured over many decisions.

---

## 1.1 First Instrument: Single Large-Cap Equity Cash Stock

Trading a single large-cap stock in the Indian equity cash segment simplifies the harness significantly versus NIFTY F&O.

**Why cash equity over NIFTY:**

| Factor | NIFTY F&O | Large-Cap Cash Equity |
|---|---|---|
| Instrument type | Futures/Options, needs expiry/strike/lot-size | `equity_cash`, constant |
| Short selling | Allowed on futures | CNC (delivery) has no short selling; MIS intraday shorts restricted |
| Action space | BUY / SELL / SKIP / HOLD / EXIT | BUY / SKIP / HOLD / EXIT (SELL deferred to intraday MIS later) |
| Lot size | Multiple (25/50/75/etc.) | 1 share |
| Charges | Asymmetric (STT varies per leg) | Well-defined, symmetric within delivery, slight variance in MIS |
| Slippage | 2-3 points on index, more on options | 0.05-0.20% on liquid large-caps |
| Corporate actions | None for index | Splits, dividends, bonuses must be handled in data pipeline |
| Multi-stock | Not applicable for NIFTY | Start with one, add later |
| Gap opens | Smaller, structure-driven | Larger, can be earnings/news-driven |

**Recommended candidates:** RELIANCE, TCS, HDFC Bank, ICICI Bank, Infosys. Pick one with tight spreads, good intraday range, and clean price-action structure.

**Single-stock, single-position:** only one active position at a time. Multi-stock portfolio logic is added later, after the single-stock harness is stable.

**Action space simplification for equity cash CNC:**

```text
FLAT state:  BUY / SKIP only (no short selling in delivery segment)
OPEN state:  HOLD / EXIT
```

SELL as an entry action is valid only for intraday MIS. For the first harness, restrict to CNC delivery with mandatory EOD square-off (or managed as delivery with position tracking).

**Corporate actions data pipeline requirement:** Historical OHLCV must provide adjusted close for context (charting, indicators) while actual trade prices use raw unadjusted values. Any backtest must handle dividend dates, split dates, and bonus dates explicitly so the agent does not see misleading price gaps in historical context.

---

## 2. Research Notes

### 2.1 FinMem: Layered Memory for Trading Agents

Source: `FinMem: A Performance-Enhanced LLM Trading Agent with Layered Memory and Character Design`, arXiv:2311.13743.

Useful idea:

- Trading agents need a memory architecture, not isolated prompts.
- FinMem uses layered memory to assimilate hierarchical financial data and improve decision-making.
- The memory module is designed to mimic how human traders retain important market context beyond short-term perception.

Application to AlphaPulse:

- Add **working memory**, **session memory**, **episodic trade memory**, and **reflection memory**.
- The DART agent should see selected memories at every decision point.
- Outcome evaluation should write back to memory.

### 2.2 ReAct: Interleaved Reasoning + Tool Use

Source: `ReAct: Synergizing Reasoning and Acting in Language Models`, arXiv:2210.03629.

Useful idea:

- Agents perform better when reasoning and tool actions are interleaved.
- Tool calls reduce hallucination by grounding reasoning in external deterministic information.

Application to AlphaPulse:

- Replace the current shallow tool loop with an explicit **analysis plan → tool use → synthesis → final decision** loop.
- The agent should request tools according to a structured price-action analysis workflow.

### 2.3 Reflexion: Learning via Verbal Feedback

Source: `Reflexion: Language Agents with Verbal Reinforcement Learning`, arXiv:2303.11366.

Useful idea:

- Agents can improve without fine-tuning by reflecting on feedback and storing lessons in episodic memory.
- Feedback can be scalar or textual.

Application to AlphaPulse:

- After every signal is evaluated, generate a compact reflection:
  - What the agent expected
  - What actually happened
  - Whether the trade thesis was valid
  - What to watch next time
- Feed relevant reflections back into future decisions.

### 2.4 FinAgent: Tool-Augmented Financial Trading Agent

Source: `A Multimodal Foundation Agent for Financial Trading: Tool-Augmented, Diversified, and Generalist`, arXiv:2402.18485.

Useful idea:

- FinAgent combines market intelligence, tool augmentation, reflection, diversified memory retrieval, and reasoning-for-action.
- The paper emphasizes that trading agents should not rely only on raw model reasoning; they need external tools, historical behavior retrieval, and feedback.
- Its dual-level reflection and diversified memory retrieval are directly relevant to AlphaPulse.

Application to AlphaPulse:

- Build a deterministic **market intelligence layer** before the LLM: structure, VWAP, volume profile, regime, liquidity, session behavior.
- Add **dual reflection**:
  - low-level reflection: trade-by-trade diagnosis
  - high-level reflection: recurring setup lessons and regime-level behavior
- Add **diversified retrieval**:
  - retrieve same-session context
  - retrieve similar historical setups
  - retrieve mistakes/warnings relevant to current setup tags

### 2.5 LangChain / LangGraph: Do Not Reinvent Agent Plumbing

Sources: LangChain structured output docs and LangGraph persistence docs.

Useful idea:

- LangChain supports structured output through provider-native schemas or tool-calling fallback, returning validated `structured_response` data.
- LangGraph provides graph-based agent workflows, checkpointing, persistent state, state history, replay/time-travel debugging, and memory stores.
- LangGraph's persistence model is especially useful for long-running agents because state can be checkpointed per step and resumed after failure.

Application to AlphaPulse:

- Use LangChain/LangGraph where they reduce generic plumbing:
  - structured output handling
  - tool binding
  - graph orchestration
  - checkpointing
  - agent state replay/debugging
  - memory retrieval patterns
- Do **not** outsource trading-critical logic:
  - trade math
  - capital ledger
  - risk validation
  - no-lookahead context slicing
  - Postgres portfolio state
  - evaluation metrics

Recommended stance:

> Use LangGraph for the agent workflow graph, but keep all market math, risk, portfolio state, and validation as deterministic AlphaPulse modules.

### 2.6 VWAP

Source: Investopedia VWAP reference.

Useful idea:

- VWAP resets at each session and is widely used on intraday charts.
- It combines price and volume and acts as a session-level reference for value, trend, and mean reversion.

Application to AlphaPulse:

- Add session VWAP and VWAP bands.
- Agent should know:
  - Price above/below VWAP
  - Distance from VWAP
  - VWAP slope
  - VWAP reclaim/rejection
  - Trend-day vs mean-reversion interpretation

### 2.7 Volume Profile

Source: TradingView volume profile documentation.

Useful idea:

- Volume profile reveals trading activity at price levels.
- Key concepts:
  - POC: highest-volume price level
  - VAH / VAL: value area high/low, often around 70% of session volume
  - HVN: high-volume nodes, accepted/fair-value areas
  - LVN: low-volume nodes, rejection/fast-move areas

Application to AlphaPulse:

- Add previous-session and current-session volume profile tools.
- Agent should reason about:
  - Price acceptance inside value
  - Rejection outside value
  - POC magnet behavior
  - LVN breakout/failure
  - Previous day value-area relationship

### 2.8 Structured Outputs

Source: OpenAI structured output documentation.

Useful idea:

- Structured outputs enforce JSON Schema adherence.
- They are superior to free-form JSON parsing because they prevent missing fields and invalid enums.

Application to AlphaPulse:

- Replace regex JSON extraction with a schema-driven response layer.
- If provider does not support strict schema mode, fall back to Pydantic validation + repair retry.

---

## 3. Current Agent Architecture and Gaps

| Component | Current Behavior | Gap |
|---|---|---|
| `DartAgent` | Single agent with max 3 tool calls | No durable memory, no planning, no reflection, no calibration |
| `ToolHarness` | Basic deterministic tools | Missing VWAP, volume profile, liquidity map, regime, session structure, confluence scoring |
| `prompts.py` | Static DART prompt | Does not force systematic price-action reasoning |
| `context.py` | Builds compact OHLCV + indicators package | Daily/weekly completeness issue; weak session context; no market map |
| `evaluator.py` | Scores outcomes after replay | Feedback is never returned to agent |
| `journal/signal.py` | Stores decisions | No memory extraction, no outcome-linked lessons |
| `validation/validator.py` | Validates trade math and thesis | Validator rejects but agent does not learn why |
| `core/position.py` | Tracks basic open/closed position state | Does not expose full capital ledger to the agent |
| State persistence | Redis currently caches some decisions/position state | Need Postgres as durable source of truth for portfolio, decisions, memory, session map, and experiments |
| Agent orchestration | Hand-written loop | Could use LangGraph-style graph workflow, checkpointing, and replay instead of custom state plumbing |

---

## 4. Correct Context Contract

The current data context must be corrected before we can trust any agent evaluation.

### 4.1 Required Context at Every Decision Time `T`

At decision time `T`, the agent should always receive:

```text
1. Weekly context:
   - Previous 3 months of completed weekly candles
   - Current incomplete week excluded unless explicitly marked partial

2. Daily context:
   - Previous 1 month of completed daily candles
   - Current incomplete day excluded unless explicitly marked partial

3. Intraday context:
   - Previous 3 trading sessions of completed 15-minute candles
   - Current closed 15-minute candle included
   - No partially forming 15-minute candle included
```

### 4.2 Why This Matters

If the current day daily candle or current week weekly candle is included during intraday replay, the agent may see future high/low/close information. That creates lookahead leakage and destroys evaluation quality.

### 4.3 Proposed Fix

Introduce a `ContextWindowPolicy`:

```python
@dataclass
class ContextWindowPolicy:
    weekly_months: int = 3
    daily_months: int = 1
    intraday_sessions: int = 3
    intraday_timeframe: str = "15min"
    include_partial_daily: bool = False
    include_partial_weekly: bool = False
    require_complete_intraday_candles: bool = True
```

Then split context functions:

```python
def get_completed_weekly_context(df_weekly, T, months=3): ...
def get_completed_daily_context(df_daily, T, months=1): ...
def get_completed_intraday_context(df_5m, T, sessions=3, timeframe="15min"): ...
```

Rules:

- Daily context only includes candles whose trading session is complete before `T.date()`.
- Weekly context only includes weeks that ended before the current week containing `T`.
- Intraday context includes only closed candles labeled `<= T`.
- If partial daily/weekly data is ever added later, it must be clearly labeled as `partial_current_day` / `partial_current_week` and separated from completed candles.

---

## 4.4 Postgres Portfolio + Decision State Contract

The agent should never decide without knowing account/portfolio state. Even in historical harness mode, we should simulate portfolio state as if the agent were trading a real account.

Postgres should become the durable source of truth for:

1. Current and historical portfolio state
2. Current and historical position state
3. Current session state
4. Current session memory
5. Decision records
6. Evaluated outcomes
7. Reflection memory
8. Experiment manifests and run metadata
9. Agent graph checkpoints if LangGraph is used

Redis may be added later as an optional low-latency cache, but it must not be the source of truth for capital, positions, or decisions.

### 4.4.1 Proposed Tables

```text
runs
portfolio_snapshots
positions
orders_simulated
decisions
session_maps
session_events
session_levels
memory_episodes
memory_reflections
calibration_stats
experiment_runs
data_snapshots
agent_checkpoints
```

For memory retrieval, use Postgres with `pgvector` when embeddings are introduced:

```text
memory_episodes.embedding vector(...)
memory_reflections.embedding vector(...)
```

### 4.4.2 Orders Simulated Table

This table bridges intent (decisions) and execution (positions). Every trade intent becomes at least one order record, and every position has an associated entry order.

```text
orders_simulated
  order_id            TEXT PRIMARY KEY
  run_id              TEXT NOT NULL
  decision_id         TEXT REFERENCES decisions(decision_id)
  position_id         TEXT REFERENCES positions(position_id)
  symbol              TEXT NOT NULL
  instrument_type     TEXT NOT NULL    -- 'equity_cash'
  product_type        TEXT NOT NULL    -- 'CNC'
  order_side          TEXT NOT NULL    -- 'BUY' or 'SELL'
  order_type          TEXT NOT NULL    -- 'ENTRY', 'STOP_LOSS', 'TARGET', 'EXIT', 'FORCED_SQUAREOFF'
  requested_price     DOUBLE PRECISION NOT NULL
  requested_quantity  INTEGER NOT NULL
  executed_price      DOUBLE PRECISION  -- with slippage applied
  executed_quantity   INTEGER
  slippage_points     DOUBLE PRECISION  -- positive = adverse
  slippage_pct        DOUBLE PRECISION
  charges_brokerage   DOUBLE PRECISION
  charges_stt         DOUBLE PRECISION
  charges_exchange    DOUBLE PRECISION
  charges_sebi        DOUBLE PRECISION
  charges_stamp       DOUBLE PRECISION
  charges_gst         DOUBLE PRECISION
  charges_total       DOUBLE PRECISION  -- sum of all charge components
  breakeven_adjustment DOUBLE PRECISION -- how much price must move to cover charges
  order_status        TEXT NOT NULL    -- 'PENDING', 'FILLED', 'CANCELLED', 'REJECTED', 'SIMULATED'
  filled_at           TIMESTAMPTZ
  created_at          TIMESTAMPTZ NOT NULL
```

The position table's `entry` should be the *executed* price from the filled ENTRY order, not the requested price from the signal. Same for stop-loss exit and target exit.

### 4.4.3 Portfolio Snapshot

Table: `portfolio_snapshots`

Shape:

```json
{
  "run_id": "2026-05-28-backtest-001",
  "mode": "backtest",
  "timestamp": "2026-05-28T10:15:00+05:30",
  "starting_capital": 100000.0,
  "cash_available": 70000.0,
  "capital_deployed": 30000.0,
  "capital_reserved": 0.0,
  "realized_pnl": 0.0,
  "unrealized_pnl": 0.0,
  "charges_paid": 0.0,
  "max_capital_per_trade": 30000.0,
  "max_daily_loss": 3000.0,
  "daily_loss_used": 0.0,
  "trades_taken_today": 1,
  "max_trades_per_day": 5
}
```

The agent receives a compact portfolio summary in every decision prompt:

```text
Portfolio State:
- Cash available: ₹70,000
- Capital deployed: ₹30,000
- Open position: BUY RELIANCE, qty=12, entry=₹2,385.50, stop=₹2,360, target=₹2,445
- Realized P&L today: ₹0
- Unrealized P&L: ₹-36
- Charges paid today: ₹0
- Trades today: 1 / 5
- Daily loss used: ₹0 / ₹3,000
```

### 4.4.4 Position State

Table: `positions`

Shape:

```json
{
  "position_id": "pos_20260528_1015_reliance",
  "run_id": "2026-05-28-backtest-001",
  "symbol": "RELIANCE",
  "instrument_type": "equity_cash",
  "product_type": "CNC",
  "active": true,
  "direction": "BUY",
  "entry": 2385.50,
  "executed_entry": 2386.00,
  "stop": 2360.00,
  "target": 2445.00,
  "quantity": 12,
  "entry_time": "2026-05-28T10:15:00+05:30",
  "last_price": 2382.50,
  "unrealized_pnl": -36.00,
  "r_multiple_live": -0.14,
  "status": "OPEN",
  "entry_order_id": "ord_entry_20260528_1015",
  "exit_order_id": null,
  "slippage_entry": 0.50,
  "slippage_exit": null,
  "charges_entry": 45.20,
  "charges_exit": null,
  "charges_total": 45.20
}
```

Note: `instrument_type` and `product_type` are invariant for the first harness (`equity_cash`/`CNC`). They are included so the schema is ready for F&O without a migration.

### 4.4.5 Decision State

Table: `decisions`

Stores:

- `run_id`
- `symbol`
- `decision_time`
- market state hash
- context contract
- portfolio snapshot before decision
- portfolio snapshot after decision
- position snapshot
- agent action
- raw action
- validated action
- rejection reason
- tool calls
- memory references used
- reflection IDs used
- prompt/model/agent version

### 4.4.6 Session Memory State

Tables:

```text
session_maps
session_events
session_levels
memory_episodes
memory_reflections
```

Session map stores:

- opening range
- session high/low
- session VWAP behavior
- current POC / VAH / VAL
- important active levels
- rejected levels
- accepted zones
- prior predictions
- current market regime
- current bias
- gap classification
- cooldown / trade lock state

### 4.4.7 Postgres Access Rules

- The agent can read portfolio/session/memory summaries.
- The agent can propose observations, but code decides what is written permanently.
- The validator and portfolio ledger own cash, quantity, charges, realized P&L, and deployed capital.
- The LLM must never self-report capital left or update portfolio state.
- Every accepted trade updates Postgres in a single transaction:
  1. reserve/deploy capital
  2. create position
  3. append decision event
  4. write portfolio snapshot
  5. update session memory

If any part fails, the transaction rolls back. No partial capital/position state should exist.

---

## 4.5 Action Semantics: SKIP vs HOLD

The action vocabulary must reflect actual trading state.

### Flat State: No Open Position

Allowed actions:

```text
BUY  = open a long position if validator and risk manager approve (delivery CNC)
SKIP = no trade; setup incomplete or not worth trading
```

SELL as an entry action is valid only for intraday MIS shorts. For the first CNC-only harness, SELL is excluded from the flat-state action space. It will be added in a later phase when MIS product type is supported.

When there is **no open position**, the agent must not output `HOLD`. If there is no trade, it outputs `SKIP`.

### Open Position State

Allowed actions:

```text
HOLD  = keep the existing position open
EXIT  = close the existing position early due to thesis failure / regime shift / risk deterioration
```

When there **is an open position**, the agent must not output `SKIP`. It should output `HOLD` if the trade thesis remains valid, or `EXIT` if the thesis is invalidated before stop/target.

First implementation may keep deterministic stop/target handling and add `EXIT` later, but the schema should be designed for it now.

### State Machine

```text
FLAT
  ├── BUY accepted  ──▶ LONG_OPEN
  ├── SELL accepted ──▶ SHORT_OPEN
  └── SKIP          ──▶ FLAT

LONG_OPEN / SHORT_OPEN
  ├── HOLD          ──▶ position remains open
  ├── EXIT          ──▶ FLAT
  ├── stop hit      ──▶ FLAT
  ├── target hit    ──▶ FLAT
  └── session end   ──▶ FLAT
```

---

## 4.6 Market Session Controller

A real intraday trading agent needs a deterministic session module. The agent should not decide purely from candle timestamps.

### 4.6.1 Session Phases

For Indian equity/index markets:

```text
PRE_OPEN          = before 09:15 IST; no decisions
OPENING_BUILD     = 09:15-09:30; build opening range, no new trades by default
ACTIVE_TRADING    = 09:30-entry_cutoff; BUY/SELL/SKIP allowed when flat
MANAGEMENT_ONLY   = entry_cutoff-squareoff_time; no new BUY/SELL, only HOLD/EXIT/manage open position
FORCED_SQUAREOFF  = squareoff_time-15:30; force close all positions
CLOSED            = after 15:30; no decisions
```

Suggested defaults:

```json
{
  "session_start": "09:15",
  "decision_start": "09:30",
  "new_entry_cutoff": "15:00",
  "force_squareoff_time": "15:20",
  "session_end": "15:30",
  "minimum_minutes_for_new_trade": 45
}
```

### 4.6.2 Validator Rule

A BUY/SELL can only pass when:

```text
current_phase == ACTIVE_TRADING
and minutes_to_force_squareoff >= minimum_minutes_for_new_trade
and expected_horizon_minutes <= minutes_to_force_squareoff
```

The session controller owns forced FLAT transitions. It must close any open position at `force_squareoff_time` even if the agent says HOLD.

---

## 4.7 Cooldown and Re-entry Policy

Real session traders do not immediately re-enter after every stop or exit. The harness needs deterministic behavioral brakes.

### 4.7.1 Cooldown Events

```text
AFTER_STOP_LOSS       -> cooldown N candles
AFTER_TARGET_HIT      -> cooldown M candles, usually shorter
AFTER_AGENT_EXIT      -> cooldown N candles
AFTER_REJECTED_SIGNAL -> optional small cooldown
AFTER_SCHEMA_FAILURE  -> no trade for that decision only
```

Suggested defaults:

```json
{
  "cooldown_after_stop_candles": 2,
  "cooldown_after_exit_candles": 2,
  "cooldown_after_target_candles": 1,
  "same_direction_reentry_candles": 3,
  "max_attempts_per_level_per_day": 2,
  "max_same_direction_losses_per_day": 2
}
```

### 4.7.2 Re-entry Rules

Same-direction re-entry after a stop is allowed only when at least one condition is true:

- a new structure break occurs
- price reclaims VWAP after losing it
- price retests and accepts the broken level
- volatility regime changes from chop/compression to expansion
- target path improves materially

Same-level repeated entries are blocked after `max_attempts_per_level_per_day`.

Store trade locks in Postgres:

```text
trade_locks(run_id, symbol, direction, level_zone, reason, expires_at, created_at)
```

---

## 4.8 Gap Open Handling

Gap opens are a major intraday regime and must be explicit in context.

### 4.8.1 Gap Features

Add a `gap_context` block to `MarketStatePackage`:

```json
{
  "prior_close": 22500.0,
  "today_open": 22650.0,
  "gap_points": 150.0,
  "gap_pct": 0.67,
  "gap_atr_multiple": 1.2,
  "gap_direction": "gap_up",
  "open_location_vs_prior_value": "above_vah",
  "gap_status": "unfilled",
  "gap_fill_level": 22500.0,
  "gap_type": "gap_and_go_candidate"
}
```

### 4.8.2 Gap Classification

```text
no_gap
small_gap
gap_up_inside_value
gap_down_inside_value
gap_up_above_value
gap_down_below_value
gap_and_go_candidate
gap_fade_candidate
gap_fill_in_progress
gap_filled
```

The agent should not treat the first hour of a gap day like a normal session. Gap context should influence session type, VWAP interpretation, and SKIP/BUY thresholds.

**Fallback gap classification without volume profile:** Volume profile (VAH/VAL) is a Phase 5 tool. In Phase 1, gap classification uses a simpler fallback:

```text
If no volume profile data available:
  gap_up_above_value    → use prior day high instead of VAH
  gap_down_below_value  → use prior day low instead of VAL
  gap_up_inside_value   → open between prior low and prior high
  gap_down_inside_value → open between prior low and prior high
```

Prior day high/low/preclose are always available from daily candles and are a reasonable initial proxy for value area boundaries.

---

## 4.9 Execution Cost Model: Equity Cash Charges

Indian equity cash market charges are well-defined and must be a deterministic module, not embedded in the LLM prompt.

### 4.9.1 CashMarketChargesModel

Module: `core/charges.py`

```python
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
    gst_rate: float = 0.18                  # 18% on (brokerage + exchange charges)

@dataclass
class EquityCashMISCharges:
    """Intraday (MIS) charges — lower STT and stamp duty on sell/intraday."""
    brokerage_per_order: float = 20.0
    stt_buy: float = 0.0                    # 0% on buy side (MIS)
    stt_sell: float = 0.00025               # 0.025% on sell side (MIS)
    exchange_txn_charge: float = 0.0000345
    sebi_fee_per_crore: float = 10.0
    stamp_duty_buy: float = 0.00003         # 0.003% on buy side (MIS)
    gst_rate: float = 0.18

@dataclass
class ChargeResult:
    total_charges: float
    breakeven_points: float  # how many points price must move to cover charges
    net_r_adjustment: float  # subtract this from gross reward when computing net R
    breakdown: dict          # component-level breakdown

def compute_charges(
    charges: EquityCashCharges,
    direction: str,          # 'BUY' or 'SELL'
    quantity: int,
    entry_price: float,
    exit_price: float,
) -> ChargeResult:
    turnover = quantity * (entry_price + exit_price)
    brokerage = charges.brokerage_per_order * 2  # entry + exit
    stt = (quantity * entry_price * charges.stt_buy +
           quantity * exit_price * charges.stt_sell)
    exchange = turnover * charges.exchange_txn_charge
    sebi = (turnover / 10_000_000) * charges.sebi_fee_per_crore
    stamp = quantity * entry_price * charges.stamp_duty_buy
    gst = charges.gst_rate * (brokerage + exchange)
    total = brokerage + stt + exchange + sebi + stamp + gst
    breakeven = total / quantity
    return ChargeResult(total_charges=total, breakeven_points=breakeven,
                        net_r_adjustment=breakeven, breakdown={...})
```

### 4.9.2 Integration Rule

Every `net_reward_risk` calculation must pass through `compute_charges`. The order simulator calls this when filling an order. The validator rejects any trade where `net_reward_risk < 2.0` after charge-adjusted pricing.

For the first harness, use `EquityCashCharges` (CNC delivery) as default. MIS charges can be selected per-run via config.

---

## 4.10 Slippage Model

Idealized candle-close prices do not reflect real execution. A configurable slippage model is required.

### 4.10.1 SlippageConfig

```python
@dataclass
class SlippageConfig:
    mode: str = "fixed_paise_per_share"  # or "percentage" or "atr_based"
    entry_slippage: float = 0.50          # ₹0.50 per share adverse (entry)
    exit_slippage: float = 0.50           # ₹0.50 per share adverse (exit)
    stop_slippage: float = 1.00           # ₹1.00 per share adverse (stop loss — wider)
    target_slippage: float = 0.50         # ₹0.50 per share adverse (target — smaller)
    force_squareoff_slippage: float = 0.75
```

For a ₹2,400 stock:

```text
Requested entry: ₹2,400.00 → Executed entry: ₹2,400.50 (+0.02%)
Requested stop:  ₹2,370.00 → Executed stop:  ₹2,369.00 (-0.04%)
Requested target:₹2,460.00 → Executed target:₹2,459.50 (-0.02%)
```

### 4.10.2 Application

Slippage is applied at order simulation time, before charges are computed. The executed price is used for P&L, R-multiple, and charge calculations. The requested price is stored only for audit.

The `orders_simulated` table stores both `requested_price` and `executed_price` with `slippage_points`.

---

## 4.11 Data Snapshot Versioning

Experiment reproducibility requires knowing exactly which data was used.

### 4.11.1 DataSnapshot

Table: `data_snapshots`

```text
snapshot_id     TEXT PRIMARY KEY
symbol          TEXT NOT NULL
source          TEXT NOT NULL    -- 'yahoo_finance', 'broker_backfill', etc.
period_start    DATE NOT NULL
period_end      DATE NOT NULL
timeframe       TEXT NOT NULL    -- '5m', '15m', 'daily', 'weekly'
candle_count    INTEGER
first_candle    TIMESTAMPTZ
last_candle     TIMESTAMPTZ
data_hash       TEXT NOT NULL    -- SHA-256 of sorted OHLCV rows
adjusted_for_splits  BOOLEAN
adjusted_for_dividends BOOLEAN
yfinance_period TEXT
created_at      TIMESTAMPTZ NOT NULL
```

Every experiment run references its data snapshot IDs. If raw data is ever re-sourced, re-adjusted, or the pipeline changes, the hash will differ and old experiment comparisons will flag as invalid rather than silently producing wrong conclusions.

**Corporate actions rule:** For stocks, historical context (charting, indicators) should use adjusted close to show continuous price history. Trade execution uses raw unadjusted prices. The harness must store both adjusted and unadjusted DataFrames separately and never mix them.

---

## 5. Target Agent Architecture

```text
                         Market Data <= T
                               |
                               v
                  Context Window Policy Layer
       (3mo completed weekly + 1mo completed daily +
        3 sessions completed 15m candles)
                               |
                               v
                    Market Structure Engine
        (swings, BOS/CHOCH, ranges, S/R, liquidity)
                               |
                               v
                    Auction/Volume Engine
        (VWAP, volume profile, POC, VAH/VAL, HVN/LVN)
                               |
                               v
                    Session Intelligence Engine
        (opening range, session phase, day type, regime)
                               |
                               v
                    Portfolio + Position State
        (Postgres ledger: capital left, deployed, open position,
         realized/unrealized P&L, daily limits)
                               |
                               v
                           Memory Layer
       (working + session + episodic + reflection memory)
                               |
                               v
                    Agent Planning + Tool Loop
       (analysis plan -> tool calls -> synthesis -> signal)
                               |
                               v
                    Structured Signal Schema
                               |
                               v
                    Deterministic Validator
                               |
                               v
                    Journal + Evaluation
                               |
                               v
                    Feedback + Reflection Writer
                               |
                               v
                         Memory Store
```

---

## 6. New Agent Modules

### 6.1 `agent/schema.py`

Defines strict Pydantic schemas for:

- Tool request
- Final signal
- DART thesis
- Price-action checklist
- Confluence score
- Memory items
- Reflection items

Example final signal shape:

```python
class DartThesis(BaseModel):
    direction: str
    area: str
    risk: str
    trigger: str

class PriceActionChecklist(BaseModel):
    market_regime: Literal["trend", "range", "volatile", "compression", "unclear"]
    session_type: Literal["trend_day", "range_day", "reversal_day", "inside_day", "opening_drive", "unclear"]
    structure_state: Literal["bullish_bos", "bearish_bos", "range_bound", "choch", "unclear"]
    location_quality: int  # 0-5
    trigger_quality: int   # 0-5
    risk_quality: int      # 0-5
    volume_confirmation: int # 0-5
    higher_tf_alignment: int # 0-5
    reason_to_wait: str | None

class FinalSignal(BaseModel):
    type: Literal["final_signal"]
    # FLAT state (CNC delivery): BUY, SKIP (SELL requires MIS, planned for later phase)
    # FLAT state (MIS future): BUY, SELL, SKIP
    # OPEN position state: HOLD, EXIT
    action: Literal["BUY", "SELL", "SKIP", "HOLD", "EXIT"]
    confidence: float
    dart: DartThesis
    checklist: PriceActionChecklist

    # BUY/SELL fields
    entry: float | None
    stop: float | None
    target: float | None
    gross_reward_risk: float | None
    net_reward_risk: float | None
    expected_horizon_minutes: int | None

    # HOLD/EXIT fields
    position_id: str | None
    thesis_health: Literal["valid", "weakening", "invalidated", "not_applicable"]
    exit_reason: str | None
    suggested_exit_price: float | None

    # Common fields
    invalidation: str | None
    reason: str
```

### 6.2 `agent/planner.py`

Adds an explicit planning step before tool use.

Agent flow:

1. Read compact context.
2. Produce analysis plan:
   - What is the market regime?
   - What levels matter?
   - What volume/auction information is needed?
   - Is there a trade location?
3. Request tools in batches.
4. Synthesize results.
5. Emit final signal.

This turns the current agent from “ask model for answer” into “model runs a repeatable analysis workflow.”

### 6.2.1 Action-Specific Schema Rules

The validator must enforce different required fields by action:

| Action | Required Fields | Invalid When |
|---|---|---|
| BUY | entry, stop, target, net_reward_risk, expected_horizon_minutes, invalidation | open position exists; session not ACTIVE_TRADING; capital insufficient |
| SELL | entry, stop, target, net_reward_risk, expected_horizon_minutes, invalidation | requires MIS product type (deferred); open position exists; session not ACTIVE_TRADING |
| SKIP | reason, checklist.reason_to_wait | position is open |
| HOLD | position_id, thesis_health, reason | no position is open |
| EXIT | position_id, exit_reason, suggested_exit_price, reason | no position is open |

`rewardRisk` should be removed or deprecated. Use explicit:

```text
gross_reward_risk
net_reward_risk
```

The decision gate uses `net_reward_risk >= 2.0` after charges. Gross R:R is informational only.

Partial exits and scale-outs are explicitly out of scope for the first agent harness. They should not be added until the single-position state machine is stable.

### 6.3 `agent/memory.py`

Implements four memory layers.

#### Working Memory

Scope: current decision only.

Stores:

- Active market state
- Tool outputs
- Candidate trade ideas

#### Session Memory

Scope: current trading session.

Stores:

- Opening range
- Important intraday levels
- VWAP behavior
- Failed breakouts
- Accepted/rejected zones
- Earlier predictions
- Current session bias

#### Episodic Memory

Scope: past signals and outcomes.

Stores:

- Signal
- Thesis
- Context tags
- Outcome
- Mistake label
- R multiple
- Reflection

#### Reflection Memory

Scope: learned rules and warnings.

Examples:

```json
{
  "lesson": "When price opens above prior value but immediately rejects VWAP, avoid chasing breakout longs.",
  "tags": ["open_above_value", "vwap_rejection", "failed_breakout"],
  "source_trades": ["2026-05-27T10:15"],
  "confidence": 0.72,
  "last_updated": "2026-05-27T10:45"
}
```

### 6.4 `agent/reflection.py`

After evaluator scores a decision, write reflection.

Input:

- Original signal
- Market state snapshot
- Tool outputs
- Future outcome
- Validation result

Output:

- What was correct?
- What was wrong?
- Was the level respected?
- Was the trigger late/early?
- Should similar future setups be traded or avoided?
- Tags for retrieval.

### 6.5 `agent/portfolio_state.py`

Reads and summarizes Postgres-backed portfolio and position state for the agent.

Responsibilities:

- Load cash available, deployed capital, capital reserved, realized P&L, unrealized P&L, charges, open position, and risk limits.
- Produce a compact `PortfolioStatePackage` for the prompt.
- Expose a deterministic `get_portfolio_state` tool.
- Prevent the LLM from creating or mutating ledger values directly.
- Provide validator/risk-manager with authoritative capital state.

The agent must know portfolio state before every decision. A flat-state `BUY` or `SELL` cannot be accepted unless portfolio state confirms enough available capital and daily risk capacity.

### 6.6 `agent/session_controller.py`

Owns session boundaries, phase transitions, entry cutoff, forced square-off, and warm-start/cold-start behavior.

Responsibilities:

- Determine current session phase.
- Block new entries outside `ACTIVE_TRADING`.
- Force position closure at configured square-off time.
- Provide `session_phase` to context and validator.
- Rebuild session map after restart.

### 6.7 `agent/cooldown.py`

Owns deterministic cooldown, re-entry, and trade-lock rules.

Responsibilities:

- Track stop/target/EXIT events.
- Block immediate revenge trades.
- Block repeated same-level trades.
- Enforce max trades per day and max losses per direction.
- Expose `get_cooldown_state` to the agent and validator.

### 6.8 `agent/graph.py` / LangGraph Orchestrator

Optional but recommended: implement the agent as a graph rather than a hand-written loop.

Proposed graph nodes:

```text
load_context
  -> load_portfolio_state
  -> retrieve_memory
  -> plan_analysis
  -> execute_tools
  -> synthesize_signal
  -> validate_schema
  -> validate_trade_math
  -> risk_check
  -> persist_decision
  -> update_memory
```

Why LangGraph helps:

- checkpoint every node
- replay failed decisions
- inspect state history
- support durable memory
- separate planning/tool/synthesis steps cleanly
- enable A/B experiments by swapping graph nodes

Important boundary:

> LangGraph orchestrates the workflow. AlphaPulse deterministic modules own trading logic.

### 6.9 `agent/calibration.py`

Tracks whether confidence is meaningful.

Metrics:

- Accuracy by confidence bucket
- Average R by confidence bucket
- Win rate by setup tag
- Average adverse excursion by setup type
- False breakout frequency
- SKIP missed-opportunity rate when flat
- HOLD quality while managing an open position

The agent receives calibration hints only when the statistics are reliable:

```text
Recent calibration:
- BUY signals tagged `breakout_after_compression` have +0.42 avg net R.
- SELL signals near daily support have -0.75 avg net R.
- Confidence 0.70-0.80 has only 38% win rate recently; be more selective.
```

Calibration update policy:

```json
{
  "update_after_each_session": true,
  "min_trades_per_bucket": 20,
  "min_setups_per_tag": 10,
  "confidence_interval_required": true,
  "do_not_prompt_if_sample_too_small": true
}
```

A calibration hint should not be shown to the agent unless its sample size clears the threshold. Otherwise it can mislead the model.

---

## 7. New Deterministic Tools

### 7.1 Context Tools

| Tool | Purpose |
|---|---|
| `get_context_contract` | Confirms exact weekly/daily/intraday context windows used at T |
| `get_session_summary` | Opening range, high/low, session VWAP, current phase |
| `get_prior_session_levels` | Prior day high/low/close, prior value area, prior POC |

### 7.2 Market Structure Tools

| Tool | Purpose |
|---|---|
| `detect_market_structure` | BOS, CHOCH, HH/HL, LH/LL, range state |
| `detect_liquidity_zones` | Stops above/below swing highs/lows, equal highs/lows |
| `detect_supply_demand_zones` | Impulse-origin zones, rejection zones, base zones |
| `score_level_quality` | Touch count, recency, volume, HTF confluence |
| `detect_breakout_quality` | Breakout strength, retest status, fakeout risk |

### 7.3 Auction / Volume Tools

| Tool | Purpose |
|---|---|
| `compute_session_vwap` | VWAP, slope, distance, bands |
| `compute_volume_profile` | POC, VAH, VAL, HVN, LVN for session/range |
| `compare_prior_value_area` | Current price vs previous session VAH/VAL/POC |
| `detect_volume_confirmation` | Volume expansion, dry-up, climax, absorption proxy |

### 7.4 Regime and Session Tools

| Tool | Purpose |
|---|---|
| `detect_market_regime` | Trend/range/volatile/compression |
| `classify_session_type` | Trend day, range day, reversal day, opening drive |
| `score_trade_location` | At support/resistance/VWAP/value edge/range middle |
| `score_confluence` | Combines HTF, structure, volume, VWAP, risk quality |

### 7.5 Memory Tools

| Tool | Purpose |
|---|---|
| `get_active_session_memory` | Retrieves current session market map |
| `retrieve_similar_setups` | Finds prior similar episodes by tags/context |
| `get_recent_reflections` | Retrieves relevant learned warnings/rules |
| `write_observation` | Stores agent observation/level/thesis |

### 7.6 Portfolio / Position Tools

| Tool | Purpose |
|---|---|
| `get_portfolio_state` | Returns cash available, deployed capital, P&L, charges, trades today, risk budget |
| `get_open_position` | Returns active position details, live/simulated R multiple, thesis, stop, target |
| `get_decision_history` | Returns prior decisions for the current session from Postgres |
| `get_capital_constraints` | Returns max trade size, max daily loss, max trades/day, cooldown, remaining risk budget |
| `get_session_phase` | Returns PRE_OPEN / OPENING_BUILD / ACTIVE_TRADING / MANAGEMENT_ONLY / FORCED_SQUAREOFF / CLOSED |
| `get_cooldown_state` | Returns active trade locks, re-entry eligibility, attempts per level, losses per direction |

These tools are deterministic and read from Postgres/session controller state. The LLM may inspect state but cannot mutate it.

A BUY/SELL proposal is invalid unless the agent had portfolio state in context or called `get_portfolio_state` during the tool loop.

### 7.7 Evaluation Tools

| Tool | Purpose |
|---|---|
| `get_recent_signal_outcomes` | Summarizes recent performance without future leakage beyond evaluated records |
| `get_confidence_calibration` | Gives accuracy/R by confidence and setup tag |

Important rule:

> Memory and feedback tools may only expose outcomes for signals whose evaluation horizon has already completed. They must never expose future data for the current decision.

---

## 8. Upgraded Prompt Architecture

The prompt should be split into reusable sections:

```text
1. Identity
   You are a price-action intraday trading analyst.

2. Non-negotiable constraints
   - No future data.
   - If flat and not trading, output SKIP, not HOLD.
   - HOLD is valid only when an open position exists and the thesis remains valid.
   - Code owns math, portfolio state, and risk validation.
   - Agent must know capital left and open position state before proposing a trade.
   - Use tools for levels/volume/profile/math/portfolio state.

3. Context contract
   - You receive 3mo completed weekly, 1mo completed daily, 3 sessions completed 15m.

4. DART framework
   Direction, Area, Risk, Trigger.

5. Price-action workflow
   A. Higher-timeframe bias
   B. Session context
   C. Market structure
   D. Auction/value/VWAP
   E. Liquidity and levels
   F. Trigger
   G. Risk and invalidation
   H. Trade/no-trade decision

6. Scoring rubric
   - Direction score 0-5
   - Area score 0-5
   - Risk score 0-5
   - Trigger score 0-5
   - Volume score 0-5
   - Confluence score 0-5

7. Output schema
   FinalSignal Pydantic/JSON schema with action state rules.
```

### 8.1 Decision Rule

For BUY/SELL:

```text
Only trade when:
- Direction score >= 3
- Area score >= 4
- Trigger score >= 3
- Risk score >= 4
- Net post-charge R:R >= 2
- Price is not in range middle unless setup is explicit mean reversion
```

Otherwise:
- If flat, output SKIP.
- If in an open position and thesis remains valid, output HOLD.
- If in an open position and thesis is invalidated before stop/target, output EXIT.

### 8.2 Hidden Reasoning vs Stored Reasoning

We should not rely on long free-form chain-of-thought in the final journal. Instead, store structured reasoning:

```json
{
  "higher_tf_bias": "daily bullish but weekly range-bound",
  "session_read": "price reclaimed VWAP after opening drive down",
  "structure": "bullish CHOCH above morning lower high",
  "area": "prior LVN retest near VWAP",
  "trigger": "15m close above retest level with volume expansion",
  "risk": "stop below failed retest low; target prior VAH",
  "why_not_hold": "DART complete and net RR passes validator"
}
```

This keeps the agent auditable without requiring verbose hidden reasoning.

---

## 9. Feedback Loop Design

### 9.1 Current Problem

`FeedbackEvaluator` computes outcomes, but the agent never sees them.

### 9.2 Proposed Flow

```text
Signal at T
   |
   v
Journal record saved
   |
   v
Evaluator runs after horizon/session
   |
   v
Outcome record produced
   |
   v
ReflectionWriter summarizes lesson
   |
   v
MemoryStore persists episode + reflection
   |
   v
Future decision retrieves relevant memories
```

### 9.3 Evaluation Horizons

Keep current metrics but add setup-oriented labels:

- T+15m direction correct?
- T+30m direction correct?
- Stop hit?
- Target hit?
- MFE / MAE
- Net R
- Did price respect proposed area?
- Did trigger fail immediately?
- Was entry late?
- Was SKIP a good avoid or missed opportunity when flat?
- Was HOLD correct while already in a position?

### 9.3.1 SKIP Quality Definition

SKIP is only used when the agent is flat. Its quality must be evaluated with a deterministic counterfactual opportunity model.

For each SKIP at time `T`, build two synthetic opportunities from the decision close:

```text
Synthetic long:
  entry = close_T
  stop  = nearest valid downside structure or 1 ATR
  target = entry + 2R

Synthetic short:
  entry = close_T
  stop  = nearest valid upside structure or 1 ATR
  target = entry - 2R
```

Then evaluate over a fixed horizon, e.g. 30-60 minutes or until session management cutoff.

Labels:

```text
good_skip_chop
  neither long nor short reached 1.0R favorable excursion, or both sides were noisy

missed_long_opportunity
  synthetic long hit 2R before stop and drawdown stayed acceptable

missed_short_opportunity
  synthetic short hit 2R before stop and drawdown stayed acceptable

ambiguous_skip
  both long and short had large excursions, or stop/target ordering is unclear
```

This prevents the memory loop from blindly rewarding over-filtering.

### 9.3.2 HOLD Quality Definition

HOLD is only used when a position is open.

For each HOLD at time `T`, evaluate relative to the already-open position:

```text
good_hold
  position later reaches target, improves MFE, or remains thesis-valid without hitting stop

bad_hold_should_exit
  position soon hits stop, thesis level breaks, or adverse excursion expands beyond expected risk

neutral_hold
  no meaningful progress before next decision and thesis remains unresolved

ambiguous_hold
  stop/target ordering is unclear or candle-level data cannot determine path
```

**thesis_health mapping to evaluation:** The agent reports `thesis_health` on every HOLD. This should influence the evaluation label:

```text
thesis_health = "valid"
  + stop NOT hit → good_hold or neutral_hold
  + stop hit     → possibly neutral_hold (thesis was valid, market changed)

thesis_health = "weakening"
  + stop hit     → bad_hold_should_exit (agent saw weakening, did not act)
  + stop NOT hit → good_hold or neutral_hold (weakening was transient)

thesis_health = "invalidated"
  + ANY outcome  → bad_hold_should_exit (agent knew it was invalidated, stayed anyway)
```

This prevents treating a HOLD where the agent clearly said "invalidated" the same as a HOLD where the thesis was "valid." The former should produce a stronger reflection.

This makes position-management feedback separate from flat-state no-trade feedback.

### 9.4 Memory Retrieval Design

"Retrieve similar setups" must be concrete, not a vague semantic search.

Use hybrid retrieval:

1. Structured filters
2. Weighted feature similarity
3. Optional pgvector semantic similarity over reflection text

### 9.4.1 Setup Tags

Examples:

```text
breakout
failed_breakout
vwap_reclaim
vwap_rejection
range_middle_trade
near_prior_high
near_prior_low
value_area_rejection
poc_magnet
lvn_breakout
hvn_chop
low_volume_breakout
opening_drive
late_day_trade
gap_up_above_value
gap_down_below_value
gap_fill
gap_and_go
same_level_retest
post_stop_reentry
```

### 9.4.2 Similarity Features

Each memory episode should store:

```json
{
  "symbol": "RELIANCE",
  "direction": "BUY",
  "action": "BUY",
  "market_regime": "trend",
  "session_type": "opening_drive",
  "gap_type": "gap_up_above_value",
  "structure_state": "bullish_bos",
  "vwap_relation": "above_vwap",
  "vwap_distance_atr": 0.4,
  "profile_location": "above_vah",
  "price_location": "near_support",
  "time_bucket": "morning",
  "volatility_bucket": "high",
  "setup_tags": ["vwap_reclaim", "opening_drive"],
  "outcome_net_r": 1.8
}
```

Suggested retrieval score:

```text
score =
  0.20 * regime_match
+ 0.15 * session_type_match
+ 0.15 * structure_match
+ 0.15 * profile_vwap_similarity
+ 0.10 * gap_type_match
+ 0.10 * time_bucket_match
+ 0.10 * tag_overlap
+ 0.05 * semantic_similarity
```

Return only top K memories after filtering by symbol and broad regime.

### 9.4.3 Memory Retrieval Query Construction

The retrieval query must be built at a defined point in the agent loop, not left ambiguous.

**When:** After the agent completes its analysis plan and before it starts tool execution. At this point, the MarketStatePackage is available and the agent has determined its working thesis (direction, area, possible setups).

**How:** The query vector is built deterministically from the current decision's features, not by the LLM:

```python
def build_retrieval_query(
    state: MarketStatePackage,
    analysis_plan: AnalysisPlan,
) -> dict:
    return {
        "symbol": state.symbol,
        "market_regime": state.regime,
        "session_type": state.session_type or "unclassified",
        "gap_type": state.gap_context.get("gap_type", "no_gap"),
        "structure_state": state.market_structure.get("state", "unclear"),
        "vwap_relation": state.vwap_context.get("relation", "at_vwap"),
        "vwap_distance_atr": state.vwap_context.get("distance_atr", 0),
        "profile_location": state.volume_profile.get("price_location", "no_data"),
        "price_location": state.price_location,
        "time_bucket": classify_time_bucket(state.decision_time),
        "volatility_bucket": classify_volatility(state.indicators),
        "analysis_direction": analysis_plan.direction_bias,
        "proposed_setup_tags": analysis_plan.setup_tags,
    }
```

The LLM may see the retrieved memories but cannot construct or modify the retrieval query.

### 9.4.4 Memory Decay and Staleness

Every memory has an effective weight:

```text
effective_weight = base_confidence * recency_decay * regime_similarity * sample_quality
```

Suggested defaults:

```json
{
  "episodic_half_life_days": 30,
  "reflection_half_life_days": 60,
  "min_examples_for_high_level_lesson": 5,
  "stale_after_days": 120,
  "regime_mismatch_penalty": 0.5
}
```

Old reflections are not deleted immediately; they are down-weighted and eventually archived.

### 9.4.5 Reflection Quality Gate

Do not write strong reflections from noisy outcomes.

A reflection is allowed only when:

- the evaluation horizon completed
- outcome is not ambiguous
- the market state snapshot is complete
- stop/target ordering is clear, or marked as low confidence
- setup tags are available

Reflection confidence levels:

```text
HIGH    = target/stop/order/path clearly resolved
MEDIUM  = useful but some ambiguity
LOW     = noisy; store only as episode, do not promote to lesson
SKIP    = do not write reflection
```

High-level reflections require multiple supporting episodes before being fed back as guidance.

### 9.4.5 Reflection Confidence → Calibration Feedback

When only LOW-confidence reflections exist, calibration stats become sparse, and `min_trades_per_bucket` gates block hints indefinitely. There must be a fallback policy:

```text
If HIGH_CONFIDENCE episodes >= min_trades_per_bucket:
  Use only HIGH confidence for calibration hints.

If HIGH_CONFIDENCE episodes < min_trades_per_bucket but HIGH+MEDIUM >= min_trades_per_bucket:
  Fall back to HIGH+MEDIUM combined with a reduced weight for MEDIUM (0.5x).
  Label hints: "[medium-confidence estimate, N=M+L samples]"

If HIGH+MEDIUM still < min_trades_per_bucket:
  Aggregate LOW-confidence episodes with LOW weight (0.25x).
  Label hints: "[low-confidence estimate, small sample N]"
  Show only if N >= 5 absolute minimum.

If total episodes < 5:
  Show no calibration hint. Agent operates without calibration.
```

This ensures calibration does not go silent when there are no clear outcomes.

### 9.4.6 Session Level Lifecycle

Session memory levels must be mutable, not append-only.

Level states:

```text
ACTIVE            Level identified, not yet tested by price
TESTED            Price approached level, level held
REJECTED          Price approached level, level rejected price
BROKEN            Price closed beyond level (configurable: body close beyond)
FLIPPED_SUPPORT   Resistance broken, now acting as support (price accepted above prior resistance)
FLIPPED_RESISTANCE Support broken, now acting as resistance
INVALIDATED       Level repeatedly violated with no reaction; no longer meaningful
EXPIRED           Removed from active map; session progressed past relevance
```

### 9.4.7 Level State Transition Triggers

State transitions are deterministic, not LLM-decided:

```text
ACTIVE → TESTED
  A candle high reaches within (atr * 0.3) of the level or a candle low reaches within (atr * 0.3) of the level

TESTED → REJECTED
  Price touched the level then the next completed candle closed away from it by at least body_size

TESTED → BROKEN
  A completed candle body closed beyond the level
  Threshold: candle body close must be beyond level by at least (atr * 0.2)

BROKEN → FLIPPED_SUPPORT
  Two completed candles above the broken resistance with both closes above it

BROKEN → FLIPPED_RESISTANCE
  Two completed candles below the broken support with both closes below it

Any state → INVALIDATED
  Three consecutive candles ignore the level (no reaction) or the level is violated twice in same session

Any state → EXPIRED
  Session phase moves to FORCED_SQUAREOFF or CLOSED
```

Wick-only violation does not trigger BROKEN. The level is only considered broken when a candle body closes beyond it.

Store lifecycle updates in `session_levels` and `session_events` so the agent receives a clean current market map rather than stale levels.

---

## 10. Agent Evaluation Infrastructure

To know if the agent is actually improving, we need experiment tracking.

### 10.1 Agent Configs

Each run should be tied to an immutable config:

```json
{
  "agent_version": "dart-pa-v2",
  "prompt_version": "pa-checklist-v1",
  "toolset_version": "structure-vwap-profile-v1",
  "memory_mode": "session+episodic+reflection",
  "model": "...",
  "temperature": 0.2,
  "decision_interval": "15min"
}
```

### 10.2 Baselines and A/B Testing Modes

Baselines are mandatory. Without them, we cannot tell whether the agent has edge or simply trades less.

Required baselines:

| Baseline | Meaning |
|---|---|
| Always SKIP | Measures whether trading at all adds value |
| Random seeded BUY/SELL/SKIP | Measures whether model beats randomness on same replay tape |
| Simple VWAP baseline | BUY above rising VWAP, SELL below falling VWAP, otherwise SKIP |
| Simple breakout baseline | Trade prior session high/low breakout with fixed ATR stop |
| Current DART POC | Existing agent before memory/tools upgrade |

Run comparisons:

| Experiment | Variant A | Variant B |
|---|---|---|
| Memory | No memory | Session + episodic memory |
| Tools | Basic tools | VWAP + volume profile + structure tools |
| Prompt | Current DART | Price-action checklist prompt |
| Model | DeepSeek | OpenRouter model |
| Tool Loop | 3 calls | Planned multi-step analysis |

### 10.2.1 Experiment Isolation

Every experiment must run on the same frozen replay tape:

```json
{
  "data_snapshot_id": "reliance_2026_02_01_to_2026_05_28_v1",
  "symbol": "RELIANCE",
  "start_date": "2026-02-01",
  "end_date": "2026-05-28",
  "decision_interval": "15min",
  "random_seed": 42,
  "agent_version": "dart-pa-v2",
  "prompt_version": "pa-checklist-v1",
  "toolset_version": "structure-vwap-profile-v1"
}
```

A/B tests must use matched date windows and identical input data. Sequential runs on different dates are invalid comparisons.

### 10.3 Success Metrics

Primary:

- Avg net R per valid trade
- Win rate
- Profit factor
- Max drawdown
- SKIP missed-opportunity rate
- HOLD quality while in open positions
- False breakout loss rate

Secondary:

- Signal count/day
- Rejection rate by validator
- Tool calls/decision
- Token cost/decision
- Latency/decision
- Confidence calibration error

---

## 11. Implementation Plan

### Phase 0 — Git Hygiene and Planning

Status: started.

- [x] Create feature branch: `feature/agent-price-action-architecture`
- [x] Create future architecture doc
- [x] Create agent evolution plan doc
- [ ] Commit docs separately before code changes

### Phase 1 — State, Context, and Action Semantics Foundation

Goal: ensure the agent receives correct market context, correct portfolio state, and emits state-valid actions without leakage.

Tasks:

1. Add Postgres service for AlphaPulse state in `docker-compose.yml`.
2. Add initial database migrations for runs, portfolio snapshots, positions, decisions, session maps/events/levels, memory episodes/reflections.
3. Add `ContextWindowPolicy`.
4. Rewrite daily context function to return completed daily candles only.
5. Rewrite weekly context function to return completed weekly candles only.
6. Standardize 15min intraday context to exactly 3 trading sessions.
7. Add Postgres-backed `PortfolioState` and `PositionState` readers.
8. Add `get_portfolio_state`, `get_open_position`, and `get_decision_history` deterministic tools.
9. Add `MarketSessionController` for session phase, entry cutoff, and forced square-off.
10. Add cooldown/re-entry read model.
11. Update action schema:
   - Flat: BUY / SELL / SKIP
   - Open: HOLD / EXIT
12. Make `SKIP` the default no-trade action when no position exists.
13. Make `HOLD` invalid when no position exists.
14. Add a `context_contract` block to `MarketStatePackage`:

```json
{
  "weekly": {"months": 3, "complete_only": true, "rows": 13},
  "daily": {"months": 1, "complete_only": true, "rows": 22},
  "intraday": {"sessions": 3, "timeframe": "15min", "complete_only": true, "rows": 75}
}
```

15. Add tests for no-lookahead daily/weekly context.
16. Add tests for action-state validity.
17. Add tests for Postgres portfolio-state serialization and prompt summary generation.
18. Add tests for session phase/cutoff behavior.
19. Add tests for cooldown/re-entry locks.
20. Add tests for charges model and slippage-aware order simulation.
21. Add tests for data snapshot hash consistency.

### Phase 2 — Structured Output Layer

Goal: eliminate brittle JSON parsing.

Tasks:

1. Create `agent/schema.py` with Pydantic models.
2. Validate every LLM output against schema.
3. Consider LangChain structured output (`ProviderStrategy` where supported, `ToolStrategy` fallback) before implementing custom parsing.
4. Add schema-aware fallback:
   - Try provider structured output if supported.
   - Else use JSON mode.
   - Else parse + Pydantic validation + one repair retry.
5. Journal schema validation errors.

### Phase 3 — Memory Layer

Goal: turn isolated decisions into coherent session analysis.

Tasks:

1. Create `agent/memory.py`.
2. Implement memory stores:
   - Postgres first for durable session memory and decision continuity
   - JSONL as append-only audit backup
   - pgvector later for semantic retrieval
3. Add session memory object:
   - active levels
   - prior predictions
   - VWAP behavior
   - accepted/rejected zones
4. Feed memory summary into prompt.
5. Add `write_observation` and `get_active_session_memory` tools.

### Phase 4 — Feedback + Reflection

Goal: let the agent learn from evaluated outcomes.

Tasks:

1. Add `agent/reflection.py`.
2. After evaluator completes, create reflection records.
3. Store reflections by setup tags.
4. Retrieve relevant reflections for future decisions.
5. Add metrics comparing memory-enabled vs no-memory mode.

### Phase 5 — Price Action Tools v1

Goal: upgrade deterministic market understanding.

Tasks:

1. Add `compute_session_vwap`.
2. Add `compute_volume_profile`:
   - POC
   - VAH
   - VAL
   - HVN
   - LVN
3. Add `detect_market_regime`.
4. Add `detect_market_structure`:
   - HH/HL/LH/LL
   - BOS
   - CHOCH
   - range state
5. Add `detect_liquidity_zones`.
6. Add `score_confluence`.

### Phase 6 — Prompt v2: Price Action Checklist

Goal: enforce systematic reasoning.

Tasks:

1. Rewrite prompt around:
   - HTF bias
   - session structure
   - market regime
   - VWAP / value area
   - liquidity
   - DART
   - risk math
2. Require checklist scores.
3. Require explicit reason for SKIP when flat.
4. Require explicit reason for HOLD or EXIT when in position.
5. Require explicit “what would prove me wrong?” for every trade.

### Phase 7 — Agentic Planner / LangGraph Workflow

Goal: enable deeper analysis without random tool use and without reinventing generic workflow plumbing.

Tasks:

1. Add planning step.
2. Prefer LangGraph graph nodes for context → portfolio → memory → planning → tools → synthesis → validation → persistence.
3. Use LangGraph checkpointing for replay/debugging where practical.
4. Increase tool call budget from 3 to configurable 6-8.
5. Allow batched tool requests.
6. Add a deterministic tool policy:
   - Must call trade math before BUY/SELL.
   - Must call VWAP/profile before breakout trades.
   - Must call similar setup memory when confidence > 0.65.
   - Must call `get_portfolio_state` before BUY/SELL.
   - Must call `get_open_position` before HOLD/EXIT.

### Phase 8 — Calibration and Experiments

Goal: measure whether the agent is improving.

Tasks:

1. Add agent versioning.
2. Add prompt versioning.
3. Add experiment IDs to journal.
4. Add calibration metrics.
5. Add A/B run support.
6. Generate comparison reports.

---

## 12. Recommended Immediate Next PR

The first implementation PR should be small and foundational:

```text
PR: agent-state-context-schema-foundation

Includes:
1. Postgres service for AlphaPulse state in Docker
2. Initial DB migrations for all tables including orders_simulated and data_snapshots
3. ContextWindowPolicy
4. Completed daily/weekly context fix
5. 3-session 15min context contract
6. CashMarketChargesModel (CNC delivery) with full breakdown
7. SlippageConfig and order simulation applying slippage before charges
8. orders_simulated table bridging decisions to positions with fill prices
9. Data snapshot hashing and versioning for experiment reproducibility
10. Postgres-backed PortfolioState / PositionState read model
11. get_portfolio_state / get_open_position / get_decision_history / get_cooldown_state tools
12. MarketSessionController with session phases, entry cutoff, forced square-off
13. Cooldown/re-entry policy read model
14. Action schema with BUY / SKIP / HOLD / EXIT (SELL reserved for MIS)
15. State-action validator:
   - flat state (CNC) allows BUY / SKIP
   - open position state allows HOLD / EXIT
16. Pydantic signal schema or LangChain structured output wrapper
17. Schema validation wrapper around current DartAgent
18. Tests for context leakage, portfolio/slippage/charges serialization, session boundary behavior,
   cooldown locks, action-state validity, and data snapshot hash consistency
```

Do **not** implement VWAP, volume profile, full memory retrieval, and full prompt rewrite in the same PR. Those should be separate PRs. Postgres state and action semantics should come first because every future memory and evaluation layer depends on them.

---

## 13. What “Predict the Future Properly” Means

The agent should not try to literally predict exact future candles.

A useful trading agent should estimate:

1. **Directional probability**
   - Is upside/downside more likely over the next N candles?

2. **Location quality**
   - Is price at an area where a trade makes sense?

3. **Invalidation clarity**
   - Is there a nearby level that proves the thesis wrong?

4. **Path quality**
   - Is there enough room to target before resistance/support/value?

5. **Asymmetric payoff**
   - Does the setup provide at least 2:1 net reward:risk?

6. **Regime fit**
   - Is the chosen tactic appropriate for the current market regime?

The correct output is often `SKIP` when flat and `HOLD` when already in a valid open position. The goal is not more trades; the goal is better filtering and better position management.

---

## 14. Operational Reliability Rules

### 14.1 Warm Start vs Cold Start

The harness must distinguish startup types.

```text
COLD_START_BEFORE_SESSION
  No prior same-day decisions. Build context from historical candles and start clean.

WARM_START_MID_SESSION
  Rebuild session map from Postgres decisions, session_events, session_levels, and candles.
  Restore open position and portfolio state before any decision.

RECOVERY_WITH_OPEN_POSITION
  Do not open new trades until position state is reconciled.
  Manage existing position using deterministic stop/session rules.

FAILED_RECOVERY
  Enter SAFE_MODE: no new BUY. Only deterministic risk-reducing actions allowed.
```

**Warm-start rebuild procedure:** A mid-session restart must produce the same session map that would have existed without the restart. To achieve this:

1. Load all `session_events` for the day, ordered by timestamp.
2. Replay events sequentially to rebuild `session_levels` state:
   - Level IDENTIFIED events create ACTIVE levels.
   - Level TESTED/BROKEN/REJECTED events transition level state.
   - Levels in FLIPPED or INVALIDATED state are not re-tested.
3. For gaps since the last event, replay through candle data:
   - For each missing candle, run deterministic level-check logic (same code path as live).
   - Do NOT re-run LLM/tool decisions. Only replay structural/level state.
4. Restore portfolio state from last `portfolio_snapshot`.
5. Restore position state from `positions` where `active = true`.
6. The first post-recovery decision uses the rebuilt session map.

This ensures determinism: the rebuilt map matches the original path because level transition triggers are deterministic rules applied to the same candle data.

A mid-session restart must not erase session memory.

### 14.2 LLM Failure Handling

Failures include:

- timeout
- malformed response
- schema validation failure
- provider rate limit
- API/network error
- repeated tool-call loop failure

Fallback actions:

```text
Flat state:
  fallback = SKIP

Open position:
  fallback = HOLD unless deterministic stop/session/forced-exit logic says EXIT
```

Circuit breaker:

```json
{
  "max_consecutive_llm_failures": 3,
  "failure_window_minutes": 30,
  "on_trip": "disable_new_entries",
  "allow_position_management": true
}
```

If the LLM keeps failing, the system should stop asking the model for new trades but continue deterministic position management.

### 14.3 Postgres Failure Handling

Postgres is the source of truth. If unavailable:

```text
Backtest mode:
  fail fast or pause until DB is available.

Paper/live future mode while flat:
  no new BUY/SELL decisions.

Paper/live future mode with open position:
  use last in-memory position snapshot only for risk-reducing actions.
  do not open new trades.
  persist recovery event once DB returns.
```

JSONL can remain an audit backup, but it is not the primary recovery path. Recovery should come from Postgres.

### 14.4 Partial Exits Explicitly Deferred

Partial exits, scale-outs, and trailing multi-lot management are useful but out of scope for the first agent harness.

First target:

```text
single symbol
single active position
single quantity block
one stop
one target
full EXIT only
```

Do not add partial exits until the core state machine and evaluation loop are stable.

---

## 15. Final Target Agent Behavior

At every 15-minute decision point, the future agent should be able to say:

```text
I know where price is relative to:
- 3-month weekly structure
- 1-month daily structure
- prior 3 sessions of 15m structure
- today's opening range
- VWAP
- prior POC / VAH / VAL
- current session POC / value area
- liquidity above/below swing highs/lows

I know what I predicted earlier today.
I know which of my recent setup types worked or failed.
I know whether this is a trend/range/volatile/compression session.
I know whether this setup is at a tradable area or range-middle noise.
I know current cash available, deployed capital, P&L, and daily risk left.
I know whether I am flat or already in a position.
If flat and setup is incomplete, I SKIP.
If in position and thesis remains valid, I HOLD.
If in position and thesis fails, I EXIT.
I know my invalidation and target before proposing a trade.
I use deterministic math before declaring a trade valid.
```

That is the bar for a useful price-action intraday agent.
