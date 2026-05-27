# Market Agent Harness Architecture

This document describes the architecture of the DART market-agent harness defined in [plan.md](plan.md) and reflected in the current Python implementation.

## Purpose

The harness is a research POC for testing whether an LLM agent can read structured multi-timeframe price-action context and produce useful `BUY`, `SELL`, or `HOLD` signals in a walk-forward backtest.

The system is intentionally not a production trading platform. It does not place broker orders. It replays historical OHLCV data, gives the decision agent only information available at decision time `T`, records the decision, and then evaluates the result against future candles after the decision is saved.

## High-Level Flow

```text
Historical OHLCV Data
        |
        v
Walk-Forward Clock
        |
        v
Market State Builder
        |
        v
Chart and Visual Context Generator
        |
        v
DART Tool Harness
        |
        v
DART Decision Agent
        |
        v
Deterministic Trade Validator
        |
        v
Signal Journal
        |
        v
Feedback Evaluator
        |
        v
POC Metrics and Artifacts
```

## Core Architecture Rule

The main architectural boundary is the leakage rule:

```text
At decision time T, the agent and every agent-accessible tool may only see data with timestamp <= T.
The evaluator may inspect future candles only after the signal has been recorded.
```

This rule separates decision generation from outcome evaluation and keeps the walk-forward test honest.

## Main Components

### 1. Configuration

Module: `config.py`

The config layer reads `.env` values and centralizes runtime settings:

- LLM provider, base URL, API key, model, and vision options.
- Instrument symbol and name.
- Indian intraday session boundaries.
- Data fetch and context-window sizes.
- Decision interval.
- Capital cap, charges, and minimum reward-to-risk rules.
- Redis, Langfuse, output, chart, and journal paths.

### 2. Data Layer

Modules:

- `data/collector.py`
- `data/raw/`

The data layer fetches or loads OHLCV candles, caches them locally, and provides resampling utilities. The current harness uses cached 5-minute, daily, and weekly data, then resamples the intraday source to the configured decision interval.

### 3. Walk-Forward Clock

Module: `core/clock.py`

The clock is responsible for replaying historical market data one eligible decision point at a time. Each decision point provides:

- `decision_time`
- current closed candle
- session start
- session end

The rest of the system treats this timestamp as the boundary for all agent-visible context.

### 4. Market State Context Layer

Module: `core/context.py`

The market state builder creates a compact `MarketStatePackage` for each decision time `T`.

It includes:

- Instrument and timestamp metadata.
- Latest closed intraday candle.
- Recent intraday candles for trigger context.
- Daily and weekly summaries.
- Indicator summaries.
- Swing highs and lows.
- Support and resistance levels.
- Price-action pattern and price-location labels.
- Trend summaries for intraday, daily, and weekly context.
- Chart paths.
- Session boundary metadata.

The package favors summaries over large raw candle dumps so the LLM gets compact, decision-relevant context.

### 5. Price-Action Summarizer

Module: `core/summarizer.py`

The summarizer owns deterministic market feature extraction:

- Indicators such as ATR, RSI, moving-average slope, momentum, and volume change.
- Swing detection.
- Support and resistance detection.
- Trend summarization.
- Pattern labels such as breakout, retest, rejection, consolidation, pullback, or unclear.
- Candidate risk estimation.

This keeps low-level market math outside the LLM.

### 6. Visual Context Layer

Module: `core/charts.py`

The chart layer generates inspectable visual artifacts for each decision:

- Micro intraday chart.
- Decision zoom chart.
- Volume chart.
- Daily context chart.
- Weekly context chart.
- Indicator panel.
- Combined context dashboard.

These files support human review and can be attached to the LLM request when the configured endpoint supports vision input.

### 7. DART Tool Harness

Module: `core/tools.py`

The tool harness mediates all tool calls requested by the LLM. It enforces:

- A fixed whitelist of tools.
- A maximum tool-call count per decision.
- Timestamp filtering to `<= T`.
- Structured JSON responses.
- No generic shell access.

Important tools include:

- `get_candles`
- `resample_candles`
- `compute_indicators`
- `detect_swings`
- `find_levels`
- `summarize_price_action`
- `estimate_risk`
- `calculate_trade_math`
- `plot_market_view`
- `plot_volume_view`
- `plot_context_dashboard`
- `get_historical_data`

The calculator tool is intentionally deterministic. Trade arithmetic such as quantity, deployed capital, charges, risk, target profit, and net reward-to-risk should be calculated by code, not by model reasoning.

### 8. DART Decision Agent

Modules:

- `agent/dart.py`
- `agent/prompts.py`

The DART agent wraps the OpenAI-compatible chat API. It sends the market state package, prompt instructions, tool descriptions, and optional chart images to the configured LLM.

The agent loop supports two response types:

- `tool_request`: the model asks the harness for additional deterministic context.
- `final_signal`: the model emits a final `BUY`, `SELL`, or `HOLD` decision.

The expected signal contains:

- action
- confidence
- DART thesis
- entry, stop, and target for actionable trades
- reward-to-risk idea
- reason
- invalidation condition

DART means:

- `D`: Direction
- `A`: Area
- `R`: Risk
- `T`: Trigger

If any part of the thesis is weak or missing, the agent should prefer `HOLD`.

### 9. Trade Validation

Module: `validation/validator.py`

The validator applies deterministic POC trade rules after the agent returns a signal.

For `BUY` or `SELL`, the signal must include:

- entry
- stop
- target
- valid direction
- clear trigger thesis

The validator calculates:

```text
quantity = floor(30000 / entry_price)
```

It then checks gross risk, target profit, fixed round-trip charges, and net reward-to-risk. A trade is accepted only if net target profit is at least `2 * gross risk`.

Rejected trades are journaled with explicit outcomes such as:

- `REJECTED_RISK_REWARD`
- `REJECTED_MISSING_LEVELS`
- `REJECTED_CHARGES`
- `REJECTED_NO_PRICE_ACTION_THESIS`
- `REJECTED_SESSION_END_CONSTRAINT`
- `REJECTED_PRICE_EXCEEDS_CAPITAL`

This separates the model's raw opinion from trades the harness would actually count.

### 10. Position State

Module: `core/position.py`

The position tracker keeps replay state about whether a valid position is open. It prevents overlapping decisions from opening conflicting positions and can use Redis when available.

The replay closes any remaining open position at the end of the run, matching the POC's session-bound square-off assumption.

### 11. Signal Journal

Module: `journal/signal.py`

The journal is an append-only JSONL record of every decision. It stores:

- decision timestamp
- instrument and price
- final action
- original model action
- confidence
- DART thesis
- trade levels
- validation result
- sizing output
- tool-call log
- chart paths
- compact market-state snapshot

The journal is the main audit trail for reviewing what the agent knew and why it acted.

### 12. Feedback Evaluator

Module: `journal/evaluator.py`

The evaluator runs after signals are recorded. It can inspect future same-session candles to score outcomes:

- `T+15m` price movement.
- `T+30m` price movement.
- Max favorable excursion.
- Max adverse excursion.
- Stop or target touch.
- Square-off result.
- Gross and net PnL.
- R multiple and net R multiple.

This component is intentionally downstream from the journal so it cannot influence the agent's decision.

### 13. Observability

Module: `observability/langfuse_integration.py`

Langfuse tracing captures:

- root decision spans
- LLM generations
- tool-call spans
- validation spans
- final decision metadata

Redis is used for lightweight state and decision caching when available.

## Runtime Orchestration

Module: `main.py`

`main.py` wires the complete POC together:

1. Validate config and create output directories.
2. Check Langfuse and Redis availability.
3. Load cached data or fetch fresh market data.
4. Resample intraday data to the configured decision interval.
5. Iterate through decision points using the walk-forward clock.
6. Build the market state package.
7. Generate charts.
8. Run the DART agent and tool loop.
9. Validate the signal.
10. Open position state when a valid trade is produced.
11. Save the signal journal entry.
12. Log tracing data.
13. Evaluate outcomes after replay.
14. Save summary metrics and evaluation artifacts.

## Data and Control Boundaries

```text
Agent-visible path:
data <= T -> market state -> tools <= T -> LLM -> raw signal

Harness-owned path:
raw signal -> validator -> journal -> position state

Evaluator-only path:
journaled signal -> future same-session candles -> metrics
```

The LLM proposes decisions, but the harness owns data access, arithmetic, validation, journaling, and evaluation.

## Output Artifacts

The harness writes artifacts under `outputs/`:

- `outputs/charts/`: chart images and dashboards.
- `outputs/journal/signal_journal.jsonl`: decision log.
- `outputs/journal/journal_summary.json`: aggregate journal summary.
- `outputs/journal/evaluation_results.json`: post-replay evaluation.

These artifacts let a human inspect each signal without rerunning the agent.
