---
name: Market Agent Harness
overview: Design a lightweight POC for testing whether a DART-style LLM agent can read structured multi-timeframe price-action data and produce useful BUY, SELL, or HOLD signals in a walk-forward backtest.
todos:
  - id: define-poc-scope
    content: "Keep the scope to a hypothesis POC: one replay loop, one DART decision agent, one signal journal, and simple feedback scoring."
    status: completed
  - id: collect-sample-data
    content: Collect one public 5-minute OHLCV dataset into a local CSV for replay.
    status: in_progress
  - id: connect-api-llm
    content: Use the DeepSeek/OpenAI-compatible API chat endpoint (base URL, model name, and API key configured in .env).
    status: pending
  - id: design-agent-tools
    content: Define the tool layer the DART agent can request through the harness.
    status: pending
  - id: design-visual-context
    content: Generate human-readable charts and vision-model inputs for price action, volume, levels, and higher-timeframe context.
    status: pending
  - id: design-market-state
    content: Define the compact market state package the agent receives at each 5-minute decision point.
    status: pending
  - id: plan-walkforward-loop
    content: Specify the walk-forward replay loop so the agent only sees data available at time T.
    status: pending
  - id: define-feedback
    content: Define basic T+15/T+30 outcome scoring for each recorded signal.
    status: pending
  - id: define-trade-validations
    content: Add POC trade validation rules for 2:1 net reward-to-risk, ₹30,000 capital cap, per-order rupee charges, deterministic calculator tooling, and forced square-off by session end.
    status: pending
isProject: false
---

# Market Time-Series Agent Harness POC Plan

## Goal

Test a hypothesis: can an LLM agent, given clean multi-timeframe price-action context up to time `T`, generate useful `BUY`, `SELL`, or `HOLD` signals that can be evaluated against future candles?

This is not a production trading system. It is a research harness for observing signal quality.

## POC Assumptions

- DART is our internal project/thesis name.
- The first version is market-agnostic, but can use any convenient OHLCV dataset.
- For the first run, prefer an Indian equity or index-style instrument with a fixed intraday session, because mandatory end-of-day square-off is part of the hypothesis.
- The base decision interval is `5m`.
- The agent can only see data available at the current replay time.
- Future candles such as `T+15m`, `T+30m`, and the rest of the same trading session are used only by the evaluator after the decision is recorded.
- The first version records signals only; it does not execute broker orders.
- The LLM backend for the POC is a DeepSeek/OpenAI-compatible API (supporting OpenAI chat format). The base URL, model name, and API key are configured in `.env`.
- Tool use is harness-mediated. The LLM can request tools in JSON, then the harness executes approved tools and sends results back.
- The POC should generate visual chart context for humans and, when supported by the model endpoint, feed those images to a vision-capable LLM.
- Every actionable trade must satisfy the POC validation rules before it is counted as a valid trade.
- Each trade setup is capped at a maximum capital restriction of ₹30,000. The harness must use this to dynamically calculate transaction size rather than executing arbitrary quantities.
- No production hardening, broker WebSockets, deployment, or full test suite for this POC.
- The tech stack is Python-first, run within a virtual environment (`venv`).
- Redis is utilized via `docker-compose` to cache, store decisions, and manage state.
- Langfuse is integrated for LLM observability and logging, configured using the API keys and endpoints defined in `.env`.

## What Current Work Suggests

- [QuantAgent](https://arxiv.org/html/2509.09995) is closest to this idea for price-driven LLM trading. It decomposes analysis into indicator, pattern, trend, and risk agents, then emits traceable trade decisions.
- [FinMem](https://ar5iv.labs.arxiv.org/html/2311.13743) shows a useful memory pattern: working memory plus layered long-term memory ranked by recency, relevance, and importance.
- Trading feedback systems such as TradingGroup and AlphaLoop point toward structured decision logs, delayed outcome evaluation, reward scoring, and reflection agents.
- Practical lesson for this POC: do not ask the LLM to parse huge raw candle dumps. Give it a compact market-state summary and ask it to reason over that.

## QuantAgent Ideas To Reuse

QuantAgent gives us a useful blueprint, but we should simplify it for the POC.

Reuse:

- Price-only inputs: start with OHLCV and ignore news/sentiment.
- Compact context windows: use recent candles, not an entire history dump.
- Indicator perspective: summarize RSI, MACD or moving-average slope, momentum, volatility, and volume change.
- Pattern perspective: summarize structures such as breakout, retest, rejection, consolidation, triangle, flag, double bottom/top, or no clear pattern.
- Trend perspective: summarize higher-timeframe direction, support/resistance slope, range boundaries, and whether price is compressing or expanding.
- Risk perspective: estimate entry, invalidation, target, and reward-to-risk.
- Evaluation horizon: compare the decision against the next three 5-minute candles as the first `T+15m` score.

Change for our POC:

- QuantAgent forces `LONG` or `SHORT`; our harness must allow `HOLD`.
- QuantAgent uses separate agents; we start with one local-LLM DART agent fed by deterministic summaries.
- QuantAgent focuses on 1h/4h benchmarks; our hypothesis starts at 5m decisions with 3-day micro context plus daily/weekly trend context.
- QuantAgent uses risk-reward around `1.2-1.8`; our thesis asks for at least roughly `2:1` before actionable trades.

## First-Principles Breakdown

The central rule is:

```text
Stored historical data can contain the full future session.
The decision agent at T must only receive candles and features <= T.
The evaluator may inspect future candles only after the signal is saved.
```

A decision at time `T` needs four artifacts:

1. Clean state: what the market looked like at exactly `T`, across multiple timeframes.
2. Thesis: why a buy, sell, or hold is justified under DART.
3. Risk idea: entry area, invalidation, possible target, and approximate reward-to-risk.
4. Feedback: what actually happened after `T`, such as `T+15m`, `T+30m`, or `T+N candles`.

The harness should be built around signal records. Every five-minute replay step should produce either a trade thesis or a no-trade thesis.

## POC Architecture

```text
Historical OHLCV Data
(3 days of 5m + daily/weekly context)
        |
        v
Walk-Forward Clock
(moves one 5m candle at a time)
        |
        v
Timeframe Builder
(intraday 5m + daily + weekly)
        |
        v
Price-Action Summarizer
(trend, ranges, swings, levels, patterns, volatility)
        |
        v
Visual Market Context
(candles, levels, volume, indicators, HTF charts)
        |
        v
Market State Package
(only data <= T)
        |
        v
DART Tool Harness
(approved tools, no future leakage)
        |
        v
DART Decision Agent
(local LLM + tool results -> BUY / SELL / HOLD)
        |
        v
Signal Journal
(input snapshot, action, thesis, confidence, levels)
        |
        v
Feedback Evaluator
(looks at future same-session candles after signal)
        |
        v
POC Results
(win rate, average R, HOLD rate, setup notes)
```

## DART Working Definition

Define DART as the internal decision thesis:

- `D`: Direction, the higher-timeframe bias and immediate momentum.
- `A`: Area, the price zone where action matters, such as support, resistance, value area, prior high/low, fair value gap, range edge, or liquidity zone.
- `R`: Risk, invalidation level, stop distance, target distance, reward-to-risk, and exposure constraints.
- `T`: Trigger, the lower-timeframe confirmation that justifies acting now rather than waiting.

This gives the agent a disciplined checklist. It also makes `HOLD` natural: if any DART part is missing, the system should prefer no trade.

## DART Tool Harness

The decision agent should not be a passive prompt. It should be able to ask the harness for more context, but only through safe, deterministic tools.

The harness controls:

- Which tools exist.
- What data each tool can access.
- Whether the requested timestamp would leak future data.
- How many tool calls are allowed per decision.
- How tool results are returned to the local LLM.

This matters because the agent may ask questions like:

- "Show me the daily and weekly trend before this 5m setup."
- "Plot the recent swing highs and lows."
- "Calculate ATR and reward-to-risk for this candidate entry."
- "Give me the last 50 5m candles and the completed 15m candles."
- "Find support and resistance around current price."

For the POC, the harness should support a small whitelist of tools:

### DART Tool Harness Whitelist

- `get_candles(timeframe, lookback)`: returns candles ending at decision time `T`.
- `get_context_slice(kind, lookback)`: returns the approved 3-day 5m, daily, or weekly context ending at decision time `T`.
- `resample_candles(targetTimeframe)`: builds completed higher-timeframe candles from available lower-timeframe data when useful.
- `compute_indicators(timeframe)`: returns basic ATR, RSI, moving-average slope, momentum, and volume change.
- `detect_swings(timeframe)`: returns recent swing highs/lows.
- `find_levels(timeframe)`: returns nearby support, resistance, prior high/low, and range boundaries.
- `summarize_price_action(timeframe)`: returns trend, range, breakout/retest/rejection/consolidation labels.
- `estimate_risk(direction, entryIdea)`: returns candidate stop, target, and reward-to-risk estimate.
- `calculate_trade_math(entry_price, stop_price, target_price, direction, capital_cap=30000, order_charge=30)`: deterministically calculates quantity, deployed capital, gross risk, gross target profit, total round-trip charges, net target profit, and net reward-to-risk. This tool must use numeric parsing/validation and must not delegate arithmetic to the LLM.
- `plot_market_view(timeframe)`: saves a candle/trend/level chart for inspection and, if supported, model vision input.
- `plot_volume_view(timeframe)`: saves volume bars and unusual-volume annotations.
- `plot_context_dashboard()`: saves a combined decision-time dashboard for the human reviewer and vision model.

### Calculator Tool Rule

The agent must not perform trade arithmetic from natural-language reasoning alone. Any calculation involving quantity, deployed capital, risk, target profit, fees, net P&L, or reward-to-risk must go through `calculate_trade_math`.

Do not expose a generic bash or terminal tool to the decision agent for arithmetic. A bash tool is too broad for this use case and creates unnecessary command-execution risk. The calculator should be a narrow deterministic Python function with a strict schema, numeric input validation, and no shell access.

Important POC constraint:

```text
Every tool receives decisionTime = T.
Every tool must filter data to timestamp <= T.
No tool can query future candles during decision generation.
```

## Tool-Calling Flow

Because the LLM API is a standard chat endpoint (supporting OpenAI chat format), we can use an OpenAI-compatible API client or direct HTTP requests, with structured outputs or lightweight JSON request/response protocols.

First call:

```json
{
  "mode": "analysis_or_tool_request",
  "marketState": "...compact state..."
}
```

The model may respond with either a tool request:

```json
{
  "type": "tool_request",
  "tool": "find_levels",
  "arguments": {
    "timeframe": "daily"
  },
  "reason": "Need higher-timeframe support and resistance before deciding."
}
```

Or a final signal:

```json
{
  "type": "final_signal",
  "action": "HOLD",
  "confidence": 0.54,
  "reason": "Daily bias is bullish but 5m trigger is unclear."
}
```

If a tool is requested, the harness executes it, appends the result, and asks the model to continue. Limit this to a small number of rounds, such as `3` tool calls per decision, so the POC remains cheap and debuggable.

## Graph And Chart Context

Charts are now part of the POC input, not only a debugging artifact. The harness should create a visual market context pack at each decision point.

For the first POC:

- Generate chart images for human inspection.
- Generate text summaries from the same chart data for text-only fallback.
- Feed chart images to the LLM when the local endpoint supports vision input.
- Store chart file paths in the signal journal.

The visual context pack should include:

- `micro_5m_chart`: last 3 trading days of 5-minute candles, with current session emphasized.
- `decision_zoom_chart`: recent 5-minute candles around decision time `T`, with support, resistance, entry, stop, and target candidates.
- `volume_chart`: volume bars, average volume, and unusual-volume markers.
- `daily_context_chart`: daily candles for the last 2 weeks to 1 month with major zones.
- `weekly_context_chart`: weekly candles for the last 3 months with broad trend/regime.
- `indicator_panel`: compact RSI, moving-average slope, ATR/volatility, and momentum view.

For a text-only LLM, the harness should send the derived summaries and chart file paths. For a vision-capable LLM, the harness should include the generated images in the request payload if the API supports image attachments. The exact image payload format should be isolated behind the LLM client so the rest of the harness does not depend on one provider's format.

Human-review requirement:

- Every saved signal should be inspectable without rerunning the harness.
- Each journal entry should link to the visual context pack used for the decision.
- A later dashboard can show the chart, the DART thesis, validation result, and future outcome side by side.

## Minimal Market State Package

At every 5-minute replay step, create a compact `MarketStatePackage`.

Include only what is useful for the POC:

- Instrument and decision timestamp.
- Latest closed 5m candle.
- Micro action context: last 3 trading days of 5-minute candles, compacted into summaries plus recent raw candles.
- Macro trend context: last 2 weeks to 1 month of daily candles.
- HTF landscape context: last 3 months of weekly candles.
- Visual context pack paths for the current decision.
- Current price location: near support, resistance, range middle, breakout area, pullback area, or no clear area.
- Trend summary per timeframe: bullish, bearish, ranging, volatile, or unclear.
- Recent price-action pattern: breakout, retest, rejection, consolidation, pullback, failed breakout, or none.
- Candidate risk levels: possible entry, invalidation, target, and reward-to-risk when available.
- Optional basic indicators: ATR, moving average slope, volume change, RSI.

Keep the first context window small enough for the local model:

- Summarize the full 3-day 5m micro context, but include only the most recent raw 5m candles needed for the immediate trigger.
- Include daily candle summaries for the last 2 weeks to 1 month.
- Include weekly candle summaries for the last 3 months.
- Derived summaries should be preferred over dumping all raw rows.

## Market State Context Layer

For an Indian intraday session, resetting context at each morning open would make the agent blind during the most important part of the day. The POC should therefore separate current-session replay from historical context.

```text
Historical Data Slice
(3 days of 5m + daily/weekly context)
        |
        v
Walk-Forward Replay Loop
        |
        v
Market State Package Created
        |
        v
DART Agent (Local LLM)
        |
        v
Deterministic Validator
(2:1 RR, charges, intraday session limit)
        |
        v
EOD Outcome Evaluator & Journal
```

Context layers:

- `Micro Action`: last 3 trading days of 5-minute candles to capture immediate intraday momentum, yesterday's levels, opening range behavior, and recent structural levels.
- `Macro Trend`: last 2 weeks to 1 month of daily candles to capture major support/resistance zones and daily bias.
- `HTF Landscape`: last 3 months of weekly candles to keep the agent grounded in the broader regime without filling the token window.

Leakage rule:

```text
At decision time T, each context layer may include only candles with timestamp <= T.
Daily and weekly candles must be completed candles unless explicitly marked as partial current-period context.
```

## LLM API & Observability Integration

Use an OpenAI-compatible API client library or HTTP request library to call the endpoint configured via environment variables in `.env` (supporting OpenAI chat format).

Integrate **Langfuse** for full observability, tracking LLM calls, prompts, costs, and tool execution traces. The Langfuse credentials (public key, secret key, host) are read from `.env`.

For the POC, the agent call should send:

- `system_prompt` / developer message: the DART rules, output schema, and instruction to use only provided data.
- `messages`: chat history representing the query with the serialized `MarketStatePackage`.
- `images`, if supported by the model endpoint: generated chart images from the visual context pack.

Expected response shape:

```json
{
  "action": "BUY",
  "confidence": 0.68,
  "dart": {
    "direction": "Daily bullish, weekly neutral, 5m retest",
    "area": "prior breakout level acting as support",
    "risk": "entry near 104.1, invalidation below 103.4, target 105.8",
    "trigger": "5m rejection candle after retest"
  },
  "entry": 104.1,
  "stop": 103.4,
  "target": 105.8,
  "rewardRisk": 2.43,
  "reason": "Aligned higher-timeframe bias with lower-timeframe trigger.",
  "invalidation": "Close below retest support."
}
```

For `HOLD`, `entry`, `stop`, `target`, and `rewardRisk` can be `null`, but the reason should be specific.

## Signal Record

The agent should output a simple structured signal:

- action: `BUY`, `SELL`, or `HOLD`
- confidence: `0..1`
- DART thesis: direction, area, risk, trigger
- entry idea, if actionable
- stop/invalidation idea, if actionable
- target idea, if actionable
- reward-to-risk estimate, if actionable
- deterministic sizing output, including quantity and actual deployed capital, if actionable
- reason for `HOLD`, if no trade
- short explanation of what would invalidate the signal

## Trade Validation Rules

Before a `BUY` or `SELL` signal is accepted as an actionable trade, the harness should validate it deterministically.

Required checks:

- The signal must include entry, stop, target, and direction.
- The harness must dynamically calculate order sizing at decision time `T` using the allocation cap:

```text
quantity = floor(30000 / entry)
```

- The calculated quantity must be greater than `0`; the instrument price cannot exceed ₹30,000 for an actionable trade under this POC cap.
- The validation engine must check the net reward-to-risk ratio after transaction fees. A trade is only valid if:

```text
Net Target Profit >= 2 * Gross Risk
```

- The setup must have a clear price-action reason, not only an indicator reason.
- The target must remain valid after applying the POC order-charge and sizing model.
- The trade must be closed no later than the end of the trading session.
- No position can remain open after the session boundary.
- There is no `T+100` limit in the session-constrained version because an Indian trading day has only about 75 five-minute candles.

If any validation fails, convert the signal to one of these journal outcomes:

- `REJECTED_RISK_REWARD`
- `REJECTED_MISSING_LEVELS`
- `REJECTED_CHARGES`
- `REJECTED_NO_PRICE_ACTION_THESIS`
- `REJECTED_SESSION_END_CONSTRAINT`

This lets us separate the agent's raw opinion from trades that the harness would actually consider.

## Order Charge & Sizing Rule

For the POC, the harness uses a fixed capital pool of ₹30,000 per trade and applies a flat fee of ₹30 per executed order leg, or ₹60 total round-trip.

This creates a practical baseline: a ₹60 round-trip charge on a ₹30,000 deployment is fixed friction of `0.20%`.

### Mathematical Model

1. **Position Sizing:**

```text
quantity = floor(30000 / entry_price)
actual_deployed_capital = quantity * entry_price
```

2. **PnL Metrics:**

```text
gross_buy_value = quantity * entry_price
gross_sell_value = quantity * exit_price

total_order_charges = 60  (₹30 entry + ₹30 exit)
net_pnl = gross_sell_value - gross_buy_value - total_order_charges
```

3. **Validation Metrics:**

For a long trade:

```text
gross_risk = quantity * (entry_price - stop_price)
gross_target_profit = quantity * (target_price - entry_price)
net_target_profit = gross_target_profit - total_order_charges
net_reward_to_risk = net_target_profit / gross_risk
```

For a short trade:

```text
gross_risk = quantity * (stop_price - entry_price)
gross_target_profit = quantity * (entry_price - target_price)
net_target_profit = gross_target_profit - total_order_charges
net_reward_to_risk = net_target_profit / gross_risk
```

A trade is valid only when:

```text
net_target_profit >= 2 * gross_risk
```

### Operational Example: Asset at ₹1,200

- **Quantity:** `floor(30000 / 1200) = 25 shares`.
- **Actual Deployment:** `25 * 1200 = ₹30,000`.
- **Stop Loss Setup:** invalidation at ₹1,180.
- **Risk:** ₹20 per share; total gross risk = `25 * 20 = ₹500`.
- **Target Setup:** profit target at ₹1,240.
- **Target Profit:** ₹40 per share; total gross profit = `25 * 40 = ₹1,000`.
- **Harness Friction Check:**
  - Net profit = `₹1,000 - ₹60 = ₹940`.
  - Net reward-to-risk = `₹940 / ₹500 = 1.88:1`.
- **Outcome:** this trade would be `REJECTED_CHARGES` by the validator because the flat fee drags the true risk-reward below the mandatory `2:1` threshold.

POC implication:

- The agent may propose an entry, stop, and target, but the harness owns the final arithmetic.
- A trade that looks like `2:1` before charges may be rejected after order charges.
- The reported profit/loss should include both gross P&L and net P&L after charges.
- The fixed ₹30 per-leg charge is a POC simplification, not a complete brokerage/tax model.

## Feedback Scoring

After the signal is recorded, the evaluator can inspect future candles.

For each signal, score:

- What happened at `T+15m`.
- What happened at `T+30m`.
- Maximum favorable movement after entry idea.
- Maximum adverse movement after entry idea.
- Whether target or stop would have been touched.
- Approximate result in `R` multiple.
- Whether `HOLD` avoided a bad trade or missed a good one.
- Whether the trade was force-closed at session end.
- Quantity and actual deployed capital under the ₹30,000 cap.
- Gross result before order charges.
- Net result after order charges.
- Net reward-to-risk after charges.

This is enough to see whether the hypothesis has signal. We do not need perfect broker-grade execution modeling yet.

## Main Hurdles For The POC

- Lookahead bias: the agent must never see `T+15m` data during the decision at `T`.
- Candle alignment: intraday, daily, and weekly context must use only candles available at decision time.
- Price-action summarization: raw candles need to become levels, trend, range, and trigger context.
- LLM consistency: output should be structured enough to compare across many replay steps.
- Too many trades: the agent may over-signal unless `HOLD` is treated as a good outcome.
- Noisy 5m data: the POC should expect many unclear cases.
- Evaluation ambiguity: if stop and target are both touched within the same candle, we need a simple rule for the POC.
- Charge and sizing modeling: the ₹30,000 capital cap plus fixed ₹30 per-order charge can reject setups that appear valid before fees.
- Session-end square-off: trades near the end of a replay session may not have enough time to reach target before forced exit.
- Morning context continuity: the 09:15/09:30 open needs previous days' 5m context plus daily/weekly context, not a clean empty intraday slate.

## Build Steps For The POC

### Step 1: Pick A Dataset

Use one instrument and historical 5m OHLCV data. A CSV is enough. The dataset only needs to be clean enough to replay.

Default for first execution:

- Use an Indian equity/index-style instrument with a fixed intraday session.
- Use the Indian equity session around `09:15` to `15:30`; the POC can optionally begin taking decisions after `09:30` to avoid the first few volatile candles.
- Save it under a local data folder, for example `data/raw/indian_sample_5m.csv`.
- Prefer a simple public source first, such as Yahoo Finance if it provides enough 5m history for the selected instrument.
- Also collect or derive daily and weekly context for the same instrument.

Output:

- Sorted candles.
- Known timezone.
- No duplicate timestamps.
- Clear open, high, low, close, volume columns.

### Step 2: Build The Walk-Forward Replay

Move through the historical candles one 5m step at a time.

At each `T`:

- Slice candles only up to `T`.
- Build the market context from 3 days of 5m data plus daily and weekly candles available up to `T`.
- Create the `MarketStatePackage`.
- Send it to the DART agent.
- Save the signal.

### Step 3: Run The DART Agent

Start with one agent, not a complex multi-agent system.

The prompt should force the agent to answer:

- What is the direction?
- Where is the important area?
- Where is the risk/invalidation?
- What is the trigger?
- Is this `BUY`, `SELL`, or `HOLD`?
- If actionable, what are the entry, stop, target, and direction?
- Use the deterministic calculator tool for quantity, deployed capital, gross risk, net target profit, and net reward-to-risk.
- Does the trade satisfy `2:1` after order charges under the ₹30,000 capital cap?
- Can the trade reasonably resolve before the session ends?

Internally, the prompt can borrow QuantAgent's structure:

- First summarize indicators.
- Then summarize pattern/structure.
- Then summarize trend.
- Then decide through DART.
- Return only JSON.

The agent may request tools before making the final decision, but the final answer must still be a structured signal.

### Step 4: Save Every Signal

Store both trades and no-trades.

Each record should contain:

- Timestamp.
- Input market-state snapshot.
- Tool requests and tool results used.
- Chart paths generated for the decision.
- Agent output.
- Parsed action.
- Entry/stop/target if present.
- Raw explanation.

### Step 5: Score Future Outcomes

For each saved signal, look forward through the same trading session.

The first score should mimic QuantAgent's simple next-three-candle evaluation:

- For `BUY`, count how many of the next three closes are above the decision close.
- For `SELL`, count how many of the next three closes are below the decision close.
- Separately compute whether the proposed stop or target was touched.
- Keep `HOLD` records so we can inspect whether the agent avoided chop or missed clean moves.
- Continue tracking until stop, target, or session end.
- If neither stop nor target is touched before the close deadline, square off at the last allowed candle.

Compute simple metrics:

- Win rate.
- Average R.
- Average net R after charges.
- HOLD rate.
- Number of actionable signals.
- Number of rejected signals by validation reason.
- Number of forced square-offs.
- Number of rejected/unclear setups.
- Best and worst setup types.
- Whether the explanation matched what actually happened.

### Step 6: Inspect The Hypothesis

After one run, inspect examples:

- Best `BUY` signals.
- Best `SELL` signals.
- Worst signals.
- Good `HOLD` calls.
- Missed opportunities.
- Repeated reasoning mistakes.

This tells us whether the agent has a useful decision pattern or is mostly producing noise.

## Optional Memory For Later In The POC

Persistent memory is useful, but should not be first.

Add it only after the no-memory baseline exists:

- Store prior signal/outcome pairs.
- Retrieve similar past setups before a new decision.
- Compare results with memory enabled vs memory disabled.
- Watch carefully for future leakage.

## Opinionated Recommendation

For the first POC, do the simplest possible version:

- One dataset.
- One replay loop.
- One DART decision agent.
- One signal journal.
- One future-outcome evaluator.
- One deterministic trade validator.
- One results summary.

Do not start with broker integration, complex memory, reinforcement learning, prompt rewriting, or a multi-agent debate system. Those are useful only after the basic hypothesis shows promise.

## NOT in Scope Yet

- Broker WebSocket or order execution.
- Production deployment.
- Full test suite.
- Multi-instrument portfolio logic.
- News, sentiment, and fundamentals.
- Reinforcement learning or fine-tuning.
- Automatic prompt rewriting.
- Complex long-term memory.

## Decisions To Make Next

- Confirm the first Indian equity/index asset and data source, such as pulling a clean 5m CSV for a high-volume stock like Reliance.
- Confirm setup of the python `venv`, Docker Compose for Redis, and Langfuse integration configurations.
- Confirm whether the first run should use `3 days of 5m + 1 month daily + 3 months weekly` context.
- Confirm the API model name and configuration parameters.
