# LLM Context Contract — What the Agent Sends to the LLM

> **Purpose**: Document exactly what data the LLM receives at each 15-minute decision point.
> **Scope**: Full session lifecycle in `DartAgent.decide()`.
> **Rule**: The LLM never sees future data. All slices are filtered to `<= T`.
> **Architecture**: Incremental context — full history on first prompt, delta updates thereafter.

---

## 1. Incremental Context Architecture

The agent uses a **stateful conversation** within each trading session. The first decision receives the complete market history; subsequent decisions receive only newly completed candles.

### 1.1 Decision Types

| Decision | What LLM Receives | Token Impact |
|---|---|---|
| **First of session** | Full context: system prompt + ALL 110 candles + portfolio + session + memory + charts | ~10K tokens |
| **2nd-100th of session** | Incremental: only newly completed candles + compact step prompt (indicators, levels, trends) | ~1K tokens |
| **New day** | Full context again (session boundary resets) | ~10K tokens |

### 1.2 Session Lifecycle

```
Trading Day Start (09:15 IST)
  │
  ├─ Decision 1 (09:30): FULL CONTEXT
  │   [System] System prompt
  │   [User]   ALL 13 weekly + 22 daily + 75 intraday candles + portfolio + session + memory + charts
  │   [Assistant] Analysis plan → Tools → Final signal
  │
  ├─ Decision 2 (09:45): INCREMENTAL
  │   [User]   INCREMENTAL MARKET UPDATE (1 new 15min candle)
  │   [User]   STEP DECISION POINT (indicators + levels + trends + portfolio + session + memory)
  │   [Assistant] Analysis plan → Tools → Final signal
  │
  ├─ Decision 3 (10:00): INCREMENTAL
  │   [User]   INCREMENTAL MARKET UPDATE (1 new 15min candle)
  │   [User]   STEP DECISION POINT (fresh indicators + levels + trends + portfolio + session + memory)
  │   [Assistant] Analysis plan → Tools → Final signal
  │
  ├─ ... (continues for ~25 decisions per day)
  │
  └─ Decision N (15:00): INCREMENTAL
      [User]   INCREMENTAL MARKET UPDATE (1 new 15min candle)
      [User]   STEP DECISION POINT
      [Assistant] Final signal

Day Boundary (next trading day)
  │
  └─ Decision 1: FULL CONTEXT (conversation history reset)
```

### 1.3 State Tracking

The agent tracks candle timestamps to detect new data:

```python
self.conversation_history = []  # Persistent messages across decisions
self.last_weekly_time = None    # Last weekly candle date sent
self.last_daily_time = None     # Last daily candle date sent
self.last_intraday_time = None  # Last intraday candle timestamp sent
self.last_session_date = None   # Current trading date (triggers reset)
```

### 1.4 Reset Logic

```python
if T.date() != self.last_session_date:
    self.reset_session()  # Clear history + timestamps
    self.last_session_date = T.date()
    # → Full context sent for new day
```

---

## 2. Message Architecture

### 2.1 First Decision of Session

```
[System]   System prompt (identity, rules, workflow, output schema)
[User]     FULL market state:
             - ALL 13 weekly candles (raw OHLCV table)
             - ALL 22 daily candles (raw OHLCV table)
             - ALL 75 intraday candles (raw OHLCV table)
             - Indicators, swings, levels, patterns
             - Portfolio state, session state, memory
             - 2 chart images (base64)
             - Tool descriptions
[Assistant] Analysis plan (round 0)
[User]     Focused memory injection (retrieved from Postgres)
[Assistant] Tool request(s) (rounds 1-N)
[User]     Tool result(s) + optional chart images
[Assistant] Final signal (BUY/SKIP/HOLD/EXIT)
```

### 2.2 Subsequent Decisions (Same Session)

```
[User]     INCREMENTAL MARKET UPDATE (only new candles since last decision)
[User]     STEP DECISION POINT:
             - Indicators (fresh from all data)
             - Support/resistance levels
             - Swing highs/lows
             - Price pattern and location
             - Portfolio state (fresh)
             - Session state (fresh)
             - Memory summary (fresh)
             - No candle tables
[Assistant] Analysis plan (round 0)
[User]     Focused memory injection
[Assistant] Tool request(s)
[User]     Tool result(s)
[Assistant] Final signal
```

Maximum rounds per decision: `max_tool_calls + 2` (1 for analysis plan, N for tools, 1 for final signal).

---

## 3. System Prompt (Sent Once per Session)

**Source**: `agent/prompts.py:BASE_SYSTEM_PROMPT` + mode prompt

**Size**: ~275 lines, ~4,000 tokens

**Persisted in**: `conversation_history[0]` (not re-sent each decision)

### 3.1 Identity
```
You are a disciplined price-action intraday trading analyst operating under
the DART decision framework.
```

### 3.2 Action Vocabulary (State-Aware)
```
FLAT state:  BUY / SKIP only (no short selling in CNC delivery)
OPEN state:  HOLD / EXIT

CRITICAL: If flat and not trading, output SKIP. If in position and thesis
valid, output HOLD. Never output HOLD when flat. Never output SKIP when
in a position.
```

### 3.3 DART Framework
```
Direction: Higher-timeframe bias and immediate momentum.
Area:      The price zone where action matters.
Risk:      Invalidation level, stop distance, target distance.
Trigger:   Lower-timeframe confirmation.
```

### 3.4 Price-Action Workflow (8 Steps)
```
A. Higher-Timeframe Bias     — weekly/daily structure, BOS/CHOCH
B. Session Context           — day type, gap, phase, opening range
C. Market Structure          — HH/HL/LH/LL, range state
D. Auction / Value / VWAP    — above/below VWAP, POC, VAH/VAL
E. Liquidity and Levels      — support/resistance, equal highs/lows
F. Trigger                   — 15m candle close, volume expansion
G. Risk and Invalidation     — stop, target, net R:R >= 2:1
H. Decision                  — BUY / SKIP / HOLD / EXIT
```

### 3.5 Scoring Rubric (0-5 each)
```
Direction (HTF alignment)
Area (location quality)
Risk (invalidation clarity)
Trigger (entry timing)
Volume (confirmation)
Confluence (overall)
```

### 3.6 Decision Rules
```
BUY only when:
  - Direction score >= 3
  - Area score >= 4
  - Trigger score >= 3
  - Risk score >= 4
  - Net post-charge R:R >= 2.0
  - Price not in range middle
  - Session allows new entries
  - Portfolio has sufficient capital
```

### 3.7 Non-Negotiable Constraints (10 rules)
```
1. No future data.
2. Code owns math, portfolio state, and risk validation.
3. MUST call get_portfolio_state before BUY/SELL.
4. MUST call calculate_trade_math before finalizing BUY/SELL.
5. For breakout trades, call detect_market_structure and compute_volume_profile.
6. For VWAP trades, call compute_session_vwap.
7. Do NOT invent levels.
8. Preference for SKIP over marginal setups.
9. Look for alignment across timeframes.
10. When chart images are attached, use them as context.
```

### 3.8 Output Schema
5 complete JSON examples:
- BUY signal (with DART, checklist, entry/stop/target, R:R)
- SKIP signal (with reason_to_wait)
- HOLD signal (with thesis_health, position_id)
- EXIT signal (with exit_reason, suggested_exit_price)
- Tool request (with tool name, arguments, reason)

### 3.9 Mode Prompt
Either `STRICT_MODE_PROMPT` or `EXPLORATORY_MODE_PROMPT` based on `config.DECISION_MODE`.

---

## 4. Full Market State (First Decision Only)

**Source**: `core/context.py:format_market_state_for_prompt(package, include_candles=True, full_history=True)`

This is the complete historical context sent only on the first decision of each session.

### 4.1 Market State Header
```
============================================================
MARKET STATE at 2026-05-28 09:30:00+05:30
Instrument: RELIANCE (RELIANCE)
Current Price: 2405.50
```

### 4.2 Context Windows
```
Context windows:
  Weekly: last 3 months of completed weekly candles; current week excluded (13 candles)
  Daily: last 1 month of completed daily candles; current day excluded (22 candles)
  Intraday: last 3 trading sessions of completed 15min candles ending at decision time (75 candles)
Context contract:
  Weekly complete_only=True rows=13 partial_rows=0
  Daily complete_only=True rows=22 partial_rows=0
  Intraday complete_only=True sessions=3 rows=75
```

### 4.3 Latest Candle
```
LATEST 15min CANDLE: O=2403.00 H=2408.50 L=2401.20 C=2405.50 V=12500
  Color: bullish, Body: 2.50, Range: 7.30
```

### 4.4 Trend Summaries (from all candles, summarized)
```
TRENDS:
  15min (micro): bullish HH/HL since 09:45
  Daily (macro): bullish with RSI 58
  Weekly (HTF): range-bound between 2350-2450
```

### 4.5 Price-Action Pattern and Location
```
PRICE-ACTION PATTERN: consolidation_after_move
PRICE LOCATION: near resistance
```

### 4.6 Indicators (computed from all intraday candles)
```
INDICATORS:
  RSI(14): 58.2 (neutral)
  ATR(14): 12.50
  MA20 slope: 0.0023 (rising)
  MA50 slope: 0.0008 (rising)
  Momentum (10p): 0.85%
  Volume vs avg: 1.15x (average)
```

### 4.7 Support/Resistance Levels
```
SUPPORT/RESISTANCE LEVELS:
  Nearest Support: 2385
  Nearest Resistance: 2415
  Supports: 2385, 2370, 2350
  Resistances: 2415, 2430, 2450
```

### 4.8 Recent Swings
```
RECENT SWINGS:
  Most Recent Swing High: 2412
  Most Recent Swing Low: 2388
```

### 4.9 ALL Intraday Candles (75 candles — full history)
```
LAST 75 15min CANDLES (most recent first):
  Time                          Open     High      Low    Close     Volume
  09:15                    2395.00  2399.80  2393.50  2398.00      18500
  09:30                    2398.00  2402.50  2396.80  2401.20      14300
  09:45                    2401.20  2406.80  2400.50  2405.30      11200
  10:00                    2403.00  2408.50  2401.20  2405.50      12500
  ... (71 more candles covering 3 trading sessions)
```

### 4.10 ALL Daily Candles (22 candles — full history)
```
LAST 22 DAILY CANDLES:
  Date           Open     High      Low    Close    Range
  2026-04-28  2320.00  2345.00  2315.00  2340.00    30.00
  2026-04-29  2340.00  2355.00  2335.00  2350.00    20.00
  ... (20 more daily candles covering 1 month)
  2026-05-27  2418.00  2430.00  2412.00  2425.00    18.00
  2026-05-28  2425.00  2428.00  2420.00  2422.00     8.00
```

### 4.11 ALL Weekly Candles (13 candles — full history)
```
LAST 13 WEEKLY CANDLES:
  Week           Open     High      Low    Close    Range
  2026-02-23  2280.00  2310.00  2270.00  2305.00    40.00
  2026-03-02  2305.00  2330.00  2295.00  2325.00    35.00
  ... (11 more weekly candles covering 3 months)
  2026-05-12  2380.00  2410.00  2370.00  2405.00    40.00
  2026-05-19  2405.00  2430.00  2395.00  2420.00    35.00
```

### 4.12 Visual Context Pack (chart descriptions + paths)
```
VISUAL CONTEXT PACK:
  These charts were generated only from data available at decision time T.
  - context_dashboard: combined price-action, volume, indicator, and HTF view
  - decision_zoom_chart: recent 15min candles around T for trigger and levels
  - micro_5m_chart: last 3 trading sessions of 15min candles for intraday structure
  - volume_chart: volume bars and relative participation context
  - daily_context_chart: daily candles for macro bias and nearby zones
  - weekly_context_chart: weekly candles for broad regime context
  - indicator_panel: compact RSI, moving-average, ATR, and momentum panel
```

---

## 5. Incremental Market Update (Subsequent Decisions)

**Source**: `core/context.py:format_incremental_candles()`

When new candles complete between decisions, only those new candles are sent.

### 5.1 Format
```
### INCREMENTAL MARKET UPDATE

Newly Completed Intraday (15min) Candles:
  Time: 09:45:00+05:30 | O=2401.20 H=2406.80 L=2400.50 C=2405.30 V=11200

Newly Completed Daily Candles:
  Date: 2026-05-28 | O=2425.00 H=2428.00 L=2420.00 C=2422.00 V=8500 Range=8.00

Newly Completed Weekly Candles:
  Week: 2026-05-26 | O=2420.00 H=2435.00 L=2415.00 C=2430.00 V=45000 Range=20.00
```

### 5.2 Detection Logic

```python
def _get_new_candles(self, package):
    new_weekly = [w for w in package["weekly_summaries"] if w["week"] > self.last_weekly_time]
    new_daily = [d for d in package["daily_summaries"] if d["date"] > self.last_daily_time]
    new_intraday = [c for c in package["recent_intraday_candles"] if c["time"] > self.last_intraday_time]
    return new_weekly, new_daily, new_intraday
```

### 5.3 Typical Incremental Payload

| Timeframe | New candles per decision | When |
|---|---|---|
| Intraday (15m) | 1 candle | Every 15 minutes |
| Daily | 0-1 candle | Once per day (at day boundary) |
| Weekly | 0-1 candle | Once per week (at week boundary) |

---

## 6. Step Decision Point (Every Decision)

**Source**: `DartAgent._build_step_user_prompt()` + `format_market_state_for_prompt(package, include_candles=False)`

This is the compact decision prompt sent at every decision point. It contains fresh indicators, levels, and state — but NO candle tables.

### 6.1 Step Prompt Structure
```
### STEP DECISION POINT at RELIANCE

---
PORTFOLIO & POSITION STATE:
- Cash available: ₹97,500.00
- Capital deployed: ₹0.00
- Open position: NONE
- Realized P&L today: ₹0.00
- Unrealized P&L: ₹0.00
- Charges paid today: ₹0.00
- Trades today: 0 / 5
- Daily loss used: ₹0.00 / ₹3,000.00

---
SESSION STATE:
- Phase: ACTIVE_TRADING
- Opening range: High=₹2408.00, Low=₹2395.00
- Session High/Low: High=₹2412.00, Low=₹2388.00
- VWAP: ₹2402.50, Slope=0.000120
- Gap classification: no_gap
- Market regime: trend
- Current bias: bullish

---
RELEVANT MEMORIES:
Recent Episodes:
  - Setup: ['vwap_reclaim'], Action: BUY, Outcome: target_hit (1.8R)
Reflections/Lessons:
  - Lesson: Breakout above VWAP worked when structure was bullish. (confidence: 0.9)

---
<market state with indicators, levels, swings, pattern — NO candle tables>

---
Decision mode is EXPLORATORY: look for a testable BUY candidate...

Follow the Price-Action Workflow. Note: Your FIRST response MUST be a
structured 'analysis_plan' JSON block...
```

### 6.2 What's Included vs Excluded

| Included | Excluded |
|---|---|
| Indicators (RSI, ATR, MA slopes, momentum, volume) | Candle OHLCV tables |
| Support/resistance levels | Raw price data |
| Swing highs/lows | |
| Price pattern and location | |
| Trend summaries (computed from all data) | |
| Portfolio state (fresh from Postgres) | |
| Session state (fresh from Postgres) | |
| Memory summary (fresh from Postgres) | |
| Chart images (if vision enabled) | |

---

## 7. Portfolio State (Every Decision)

**Source**: Postgres `portfolio_snapshots` table via `ReplayStateService`

```
--- PORTFOLIO & POSITION STATE:
- Cash available: ₹97,500.00
- Capital deployed: ₹0.00
- Open position: NONE
- Realized P&L today: ₹0.00
- Unrealized P&L: ₹0.00
- Charges paid today: ₹0.00
- Trades today: 0 / 5
- Daily loss used: ₹0.00 / ₹3,000.00
```

When a position is open:
```
- Open position: BUY RELIANCE, qty=12, entry=₹2,385.50, stop=₹2,360.00, target=₹2,445.00
- Unrealized P&L: ₹-36.00
```

**Config defaults**:
- Starting capital: ₹100,000
- Max capital per trade: ₹30,000
- Max daily loss: ₹3,000
- Max trades per day: 5

---

## 8. Session State (Every Decision)

**Source**: Postgres `session_maps` table via `SessionStateService`

```
--- SESSION STATE:
- Phase: ACTIVE_TRADING
- Opening range: High=₹2408.00, Low=₹2395.00
- Session High/Low: High=₹2412.00, Low=₹2388.00
- VWAP: ₹2402.50, Slope=0.000120
- Gap classification: no_gap
- Market regime: trend
- Current bias: bullish
```

**Session phases** (deterministic, not LLM-decided):
- `PRE_OPEN` — before 09:15 IST
- `OPENING_BUILD` — 09:15-09:30
- `ACTIVE_TRADING` — 09:30 to entry cutoff
- `MANAGEMENT_ONLY` — entry cutoff to squareoff
- `FORCED_SQUAREOFF` — 15:20-15:30
- `CLOSED` — after 15:30

---

## 9. Memory Summary (Every Decision)

**Source**: Postgres `memory_episodes` and `memory_reflections` tables

```
--- RELEVANT MEMORIES:
Recent Episodes:
  - Setup: ['vwap_reclaim'], Action: BUY, Outcome: target_hit (1.8R)
  - Setup: ['breakout'], Action: BUY, Outcome: stop_hit (-1.0R)
Reflections/Lessons:
  - Lesson: Breakout above VWAP worked when structure was bullish. (confidence: 0.9)
  - Lesson: Avoid chasing late-day breakouts without volume. (confidence: 0.7)
```

**Initial fetch**: Generic recent 3 episodes + 3 reflections for the symbol.

**After plan validation**: Replaced with focused retrieval (see Section 11).

---

## 10. Chart Images (Every Decision)

**Source**: `core/charts.py` generated at decision time `T`

**When attached**: Only when `config.VISION_ENABLED = true` (default for OpenRouter)

### 10.1 Initial Message Charts (2 images)

| Key | Description | Content |
|---|---|---|
| `context_dashboard` | Combined dashboard | Price action, volume, indicators, higher-timeframe view |
| `decision_zoom_chart` | Trigger candle zoom | Recent 15min candles around T with entry/stop/target levels |

**Encoding**: Base64 PNG, `detail: "auto"`

### 10.2 Tool-Round Charts (0-N images)

When tools like `plot_market_view`, `plot_volume_view`, or `plot_context_dashboard` return a `chart_path`, that chart is also base64-encoded and attached in the tool result message.

---

## 11. Focused Memory Injection (After Plan Validation)

**Source**: `DartAgent._build_focused_memory_context()` — deterministic weighted retrieval from Postgres

**When**: After the LLM outputs a valid `analysis_plan` (round 0), before tool execution.

```
Plan accepted.

Retrieved relevant past episodes and lessons matching today's context:
Retrieved 3 relevant past episodes (filtered by analysis plan context):
  [0.75] BUY | Setup: ['vwap_reclaim'] | Outcome: target_hit (1.8R) | Matched on: regime, session, structure
  [0.45] BUY | Setup: ['breakout'] | Outcome: stop_hit (-1.0R) | Matched on: regime, gap
  [0.30] SKIP | Setup: [] | Outcome: missed_long_opportunity (0.0R) | Matched on: session

Relevant lessons (2):
  [0.85] "Breakout above VWAP worked when structure was bullish." (confidence: 0.9, tags: ['breakout'])
  [0.40] "Avoid chasing late-day breakouts without volume confirmation." (confidence: 0.7, tags: ['late_day_trade'])

Now execute your required tools in sequence, and synthesize your final signal when ready.
```

**Retrieval weights** (matching `MemoryStore.retrieve_similar_setups`):
```
0.20 * regime_match
0.15 * session_type_match
0.15 * structure_match
0.10 * vwap_relation
0.10 * gap_type_match
0.05 * profile_location
0.05 * price_location
0.05 * time_bucket
0.05 * volatility_bucket
0.10 * setup_tag_overlap
```

---

## 12. Tool Descriptions (First Decision Only)

**Source**: `core/tools.py:ToolHarness.get_tool_descriptions()`

24 available tools the LLM can request:

### Context Tools
| Tool | Purpose |
|---|---|
| `get_portfolio_state` | Cash, deployed capital, P&L, charges, trades today |
| `get_open_position` | Active position details, R multiple, stop, target |
| `get_decision_history` | Prior decisions for current session |
| `get_capital_constraints` | Max trade size, daily loss, cooldown |
| `get_session_phase` | Current session phase |
| `get_cooldown_state` | Active locks, re-entry eligibility |

### Market Structure Tools
| Tool | Purpose |
|---|---|
| `detect_market_structure` | BOS, CHOCH, HH/HL, LH/LL, range state |
| `detect_swings` | Swing highs and lows |
| `find_levels` | Support and resistance levels |

### Auction / Volume Tools
| Tool | Purpose |
|---|---|
| `compute_session_vwap` | VWAP, slope, distance, bands |
| `compute_volume_profile` | POC, VAH, VAL, HVN, LVN |
| `compute_indicators` | RSI, ATR, MA slopes, momentum |

### Regime and Session Tools
| Tool | Purpose |
|---|---|
| `detect_market_regime` | Trend/range/volatile/compression |
| `score_confluence` | Multi-factor confluence scoring |
| `summarize_price_action` | Trend, pattern, location summary |

### Trade Math Tools
| Tool | Purpose |
|---|---|
| `calculate_trade_math` | Position sizing, charges, R:R calculation |
| `estimate_risk` | Stop and target estimation |

### Memory Tools
| Tool | Purpose |
|---|---|
| `get_active_session_memory` | Session map, levels, events |
| `write_observation` | Store agent observation (only mutating tool) |

### Chart Tools
| Tool | Purpose |
|---|---|
| `plot_market_view` | Price action chart |
| `plot_volume_view` | Volume chart |
| `plot_context_dashboard` | Combined dashboard |

### Data Tools
| Tool | Purpose |
|---|---|
| `get_candles` | Last N candles for a timeframe |
| `resample_candles` | Resample to target timeframe |
| `get_historical_data` | Historical candles for date range |

---

## 13. Tool Results (Subsequent Messages)

**Source**: `ToolHarness.execute()` → deterministic computation

Each tool result is formatted as JSON and sent back to the LLM:

```
Tool result received:
{"vwap": 2402.5, "slope": 0.00012, "distance": 3.0, "relation": "above_vwap", ...}

Continue your analysis following the Price-Action Workflow.
Remaining tool calls: 4.
```

If the tool returns a chart, the chart image is also attached.

---

## 14. Final Reminder (When Max Tools Reached)

```
You have reached the maximum tool calls. Output your final signal now.
Follow the action vocabulary strictly:
- Flat + no trade = SKIP
- Flat + clear setup = BUY
- In position + thesis valid = HOLD
- In position + thesis invalid = EXIT
Make sure you include all required fields for your chosen action type.
```

---

## 15. Data Flow Summary

```
                    Postgres
                    ┌─────────────────────────────┐
                    │ portfolio_snapshots          │──→ Portfolio summary (every decision)
                    │ positions                    │──→ Open position details
                    │ session_maps                 │──→ Session state (every decision)
                    │ memory_episodes              │──→ Past episodes (generic + focused)
                    │ memory_reflections           │──→ Learned lessons
                    │ decisions                    │──→ Decision history
                    │ trade_locks                  │──→ Cooldown state
                    └─────────────────────────────┘

                    Historical Data (filtered <= T)
                    ┌─────────────────────────────┐
                    │ 75 intraday candles          │──→ First prompt: ALL 75 as raw table
                    │ 22 daily candles             │──→ First prompt: ALL 22 as raw table
                    │ 13 weekly candles            │──→ First prompt: ALL 13 as raw table
                    │                              │──→ Subsequent: only new candles (1-3)
                    │                              │──→ Every decision: indicators, levels, trends
                    └─────────────────────────────┘

                    Charts (generated at T)
                    ┌─────────────────────────────┐
                    │ context_dashboard.png        │──→ base64 image (every decision)
                    │ decision_zoom_chart.png      │──→ base64 image (every decision)
                    │ (tool-generated charts)      │──→ base64 image (per tool round)
                    └─────────────────────────────┘

                    Deterministic Tools (on-demand)
                    ┌─────────────────────────────┐
                    │ VWAP, Volume Profile,        │──→ JSON results
                    │ Structure, Regime,           │
                    │ Confluence, Trade Math       │
                    └─────────────────────────────┘

                              │
                              ▼
                    ┌─────────────────────────────────────────────┐
                    │              LLM (GPT/DeepSeek)             │
                    │                                             │
                    │  FIRST DECISION:                            │
                    │  - System prompt (4K tok)                   │
                    │  - Full market state (8K tok)               │
                    │    - 75 intraday candles (3K tok)           │
                    │    - 22 daily candles (1K tok)              │
                    │    - 13 weekly candles (500 tok)            │
                    │    - Indicators, levels, trends (1K tok)    │
                    │  - Portfolio (200 tok)                      │
                    │  - Session (150 tok)                        │
                    │  - Memory (300 tok)                         │
                    │  - 2 chart images (2K tok)                  │
                    │  - Tool descriptions (1K)                   │
                    │  Total: ~16K tokens                         │
                    │                                             │
                    │  SUBSEQUENT DECISIONS:                      │
                    │  - System prompt (4K tok, from history)     │
                    │  - Incremental update (100 tok)             │
                    │  - Step prompt with indicators (500 tok)    │
                    │  - Portfolio (200 tok)                      │
                    │  - Session (150 tok)                        │
                    │  - Memory (300 tok)                         │
                    │  - 2 chart images (2K tok)                  │
                    │  Total: ~5K tokens                          │
                    │                                             │
                    │  Per tool round: +500-1500 tokens           │
                    └─────────────────────────────────────────────┘
```

---

## 16. What the LLM Does NOT See

| Data | Why Not |
|---|---|
| Future candles (after T) | Lookahead prevention |
| Full candle tables on subsequent decisions | Incremental updates replace them |
| Portfolio mutations | LLM reads only; code owns writes |
| Charge calculations | Deterministic via `calculate_trade_math` tool |
| Session phase transitions | Deterministic via `get_session_phase` tool |
| Cooldown lock decisions | Deterministic via `get_cooldown_state` tool |
| Position sizing math | Deterministic via `calculate_trade_math` tool |
| Other agents' decisions | Single-agent harness |
| Live market feeds | Backtest/replay mode only |

---

## 17. Token Budget Estimate

### First Decision of Session

| Component | Estimated Tokens |
|---|---|
| System prompt | ~4,000 |
| Full market state (75+22+13 candles + indicators) | ~8,000 |
| Portfolio summary | ~200 |
| Session summary | ~150 |
| Memory summary | ~300 |
| Tool descriptions | ~1,000 |
| Chart images (2x base64) | ~2,000 |
| **First decision total** | **~15,000** |
| + 3 tool rounds | ~3,000 |
| + focused memory | ~300 |
| **First decision with tools** | **~18,000** |

### Subsequent Decisions

| Component | Estimated Tokens |
|---|---|
| System prompt (from history) | ~4,000 |
| Full candle tables (from history) | ~4,500 |
| Incremental update (1 new candle) | ~100 |
| Step prompt (indicators + levels) | ~500 |
| Portfolio summary | ~200 |
| Session summary | ~150 |
| Memory summary | ~300 |
| Chart images (2x base64) | ~2,000 |
| **Subsequent decision total** | **~7,700** |
| + 3 tool rounds | ~3,000 |
| **Subsequent decision with tools** | **~10,700** |

### Full Trading Day (25 decisions)

| Metric | Value |
|---|---|
| First decision | ~18,000 tokens |
| 24 subsequent decisions | ~10,700 × 24 = ~257,000 tokens |
| **Total tokens for day** | **~275,000** |
| **Without incremental (old approach)** | **~18,000 × 25 = ~450,000** |
| **Savings** | **~175,000 tokens (39%)** |

---

## 18. Implementation Details

### 18.1 Key Functions

| Function | File | Purpose |
|---|---|---|
| `DartAgent.decide()` | `agent/dart.py:162` | Main decision loop with incremental context |
| `DartAgent._get_new_candles()` | `agent/dart.py:100` | Detects new candles by timestamp comparison |
| `DartAgent._build_step_user_prompt()` | `agent/dart.py:129` | Builds compact step prompt (no candle tables) |
| `DartAgent.reset_session()` | `agent/dart.py:93` | Clears history on day boundary |
| `format_market_state_for_prompt()` | `core/context.py:219` | Formats market state with `include_candles`/`full_history` flags |
| `format_incremental_candles()` | `core/context.py:376` | Formats only new candles as delta update |
| `build_market_state_package()` | `core/context.py:73` | Builds package with ALL candles (was last 20/10) |

### 18.2 Configuration

| Config | Default | Effect |
|---|---|---|
| `config.DECISION_INTERVAL` | `"15min"` | Intraday candle frequency |
| `config.MICRO_DAYS` | `3` | Number of intraday sessions in context |
| `config.MACRO_MONTHS` | `1` | Months of daily context |
| `config.HTF_MONTHS` | `3` | Months of weekly context |
| `config.MAX_TOOL_CALLS_PER_DECISION` | `3` | Max tool rounds per decision |
| `config.VISION_ENABLED` | `true` (OpenRouter) | Whether to attach chart images |

### 18.3 Test Coverage

| Test | File | Verifies |
|---|---|---|
| `test_incremental_context_flow` | `tests/test_incremental_context.py` | 3-scenario flow: first prompt, incremental, day boundary |
| `test_focused_memory_uses_analysis_plan` | `tests/test_agent_workflow.py` | Memory retrieval tied to analysis plan |
