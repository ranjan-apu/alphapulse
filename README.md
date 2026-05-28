# AlphaPulse — DART Market Agent Harness

**AlphaPulse** is a research harness for building and evaluating an **LLM-powered intraday price-action trading agent**. It tests whether a structured LLM agent, given clean multi-timeframe price-action data at decision time `T`, can generate useful `BUY` / `SELL` / `SKIP` / `HOLD` / `EXIT` signals in a walk-forward backtest.

> **Status**: Research POC — not a production trading system. Does not place broker orders.

---

## Table of Contents

- [Why AlphaPulse?](#why-alphapulse)
- [Architecture Overview](#architecture-overview)
- [The DART Framework](#the-dart-framework)
- [Current State (POC)](#current-state-poc)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [CLI Usage](#cli-usage)
- [Project Structure](#project-structure)
- [Roadmap](#roadmap)
- [Research Foundations](#research-foundations)
- [License](#license)

---

## Why AlphaPulse?

Human price-action traders read structure, levels, volume, and auction dynamics to decide when to enter, exit, or stay out. AlphaPulse asks:

> **Can an LLM agent replicate this reasoning process — and can we measure whether it actually has edge?**

The harness:
- Replays historical market data through a walk-forward clock.
- Feeds the agent only information available **at** decision time (no future leakage).
- Exposes deterministic tools for structure, VWAP, volume profile, risk math.
- Validates every signal against hard rules (2:1 net R:R, capital cap, session constraints).
- Evaluates outcomes against future candles and writes back feedback.

---

## Architecture Overview

```
Historical OHLCV Data → Walk-Forward Clock → Market State Builder
  → Chart Generator → DART Tool Harness → DART Decision Agent (LLM)
  → Trade Validator → Signal Journal → Feedback Evaluator → Metrics
```

The system is organized into layers:

| Layer | Directory | Purpose |
|---|---|---|
| **Agent** | `agent/` | LLM orchestration, prompts, structured schema, memory, reflection |
| **Core** | `core/` | Tools, context builder, summarizer, charts, position tracker, clock |
| **Data** | `data/` | Yahoo Finance collector, cache manager, resampling |
| **Validation** | `validation/` | Trade signal validation (R:R, session time, capital) |
| **Journal** | `journal/` | Append-only decision journal, outcome evaluation |
| **Observability** | `observability/` | Langfuse tracing integration |
| **Config** | `config.py` | Centralized typed configuration from `.env` |

### Data Flow

1. **Walk-Forward Clock** generates decision points at each `15min` candle boundary.
2. **Market State Builder** collects multi-timeframe context (intraday, daily, weekly) filtered to `<= T`.
3. **Chart Generator** produces visual context (micro, zoom, volume, daily, weekly, indicator panel, dashboard).
4. **Tool Harness** exposes 12+ deterministic tools the LLM can request (candles, indicators, swings, levels, VWAP, volume profile, risk estimation, trade math).
5. **DART Agent** reasons through the DART framework, requests tools, and produces a structured signal.
6. **Validator** checks the signal against hard rules: net R:R >= 2:1, capital cap, session time remaining, price-action thesis.
7. **Journal** records every decision with full audit trail.
8. **Evaluator** scores each signal against future candles (stop/target touch, R-multiple, T+15m/T+30m outcome).

---

## The DART Framework

Every decision is structured around four axes:

| Component | Question |
|---|---|
| **D**irection | What is the higher-timeframe bias and immediate momentum? |
| **A**rea | Where is price relative to key levels, VWAP, and value? |
| **R**isk | What is the invalidation level, stop distance, and reward-to-risk? |
| **T**rigger | Is there a lower-timeframe confirmation to act now? |

---

## Current State (POC)

The current harness is a **proof of concept** that demonstrates:

- Walk-forward replay on NIFTY 50 index data from Yahoo Finance.
- Basic 12-tool DART Tool Harness (candles, indicators, swings, levels, patterns, risk estimation, trade math, charting, historical data).
- DART Decision Agent with tool-calling loop (up to 3 calls per decision).
- Trade validator with `₹30,000` capital cap, `₹60` round-trip charges, and `2:1` net R:R minimum.
- Post-hoc feedback evaluator (T+15m, T+30m, stop/target touch, R multiples).
- Langfuse v3 observability with span hierarchy.
- Redis position tracking with in-memory fallback.

### Known Gaps (being addressed)

| Area | Gap | Phase |
|---|---|---|
| **Memory** | No inter-decision memory; each decision is isolated | Phase 3 |
| **Feedback** | Evaluator results never reach the agent | Phase 4 |
| **Context** | Daily/weekly context may include partial candles (lookahead risk) | Phase 1 |
| **Tools** | No VWAP, volume profile, market structure, or liquidity tools | Phase 5 |
| **Schema** | Free-form JSON parsing instead of structured output | Phase 2 |
| **State** | No persistent portfolio/position Postgres state | Phase 1 |
| **Testing** | Zero test coverage | Phase 1 |
| **Action Semantics** | `HOLD` used for both flat and open states (should be `SKIP` vs `HOLD`/`EXIT`) | Phase 1 |
| **CI/CD** | No linting, type checking, or CI pipeline | — |

Detailed analysis in [`docs/agent-price-action-evolution-plan.md`](docs/agent-price-action-evolution-plan.md).

---

## Quick Start

### Prerequisites

- Python 3.10+
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (optional, for Redis/Langfuse/Postgres)
- API key for an OpenAI-compatible provider (DeepSeek, OpenRouter, etc.)

### Setup

```bash
# Clone
git clone https://github.com/ranjan-apu/alphapulse.git
cd alphapulse

# Copy environment template
cp .env.example .env
# Edit .env with your API keys

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start infrastructure (optional)
docker-compose up -d
```

### Run a Replay

```bash
# Basic run
python main.py

# Quick test (5 steps)
python main.py --quick

# Specific date
python main.py --date=2026-05-27

# Limit steps
python main.py --max-steps=20

# Force data refresh
python main.py --refresh-data
```

---

## Configuration

All configuration is through `.env` file. See [`.env.example`](.env.example) for available options.

Key settings:

| Variable | Default | Description |
|---|---|---|
| `DEEPSEEK_API_KEY` | — | LLM provider API key |
| `MODEL_NAME` | `deepseek-chat` | LLM model |
| `DECISION_INTERVAL` | `15min` | Intraday decision timeframe |
| `DECISION_MODE` | `exploratory` | `strict` or `exploratory` |
| `CAPITAL_CAP` | `30000` | Max capital per trade (₹) |
| `SYMBOL` | `^NSEI` | Trading instrument |

---

## CLI Usage

```text
python main.py [options]

Options:
  --max-steps=N      Stop after N decision points
  --date=YYYY-MM-DD  Replay a specific date only
  --quick            Run 5 steps (alias for --max-steps=5)
  --refresh-data     Force fresh data fetch from Yahoo Finance
  --skip-data        Use cached data even if stale
```

---

## Project Structure

```
alphapulse/
├── agent/
│   ├── dart.py              # DART Decision Agent (LLM orchestrator)
│   ├── prompts.py           # System/user prompts
│   ├── schema.py            # Pydantic schemas (future)
│   ├── memory.py            # Memory layer (future)
│   ├── reflection.py        # Feedback reflection (future)
│   ├── planner.py           # Analysis planner (future)
│   └── calibration.py       # Confidence calibration (future)
├── core/
│   ├── clock.py             # Walk-forward replay clock
│   ├── context.py           # Market state package builder
│   ├── summarizer.py        # Indicators, swings, levels, patterns
│   ├── tools.py             # DART Tool Harness (12+ tools)
│   ├── charts.py            # Chart generation (7 chart types)
│   └── position.py          # Position tracker
├── data/
│   ├── collector.py         # Yahoo Finance data fetcher
│   └── raw/                 # Cached OHLCV data
├── validation/
│   └── validator.py         # Trade signal validator
├── journal/
│   ├── signal.py            # Decision journal (JSONL)
│   └── evaluator.py         # Post-hoc outcome evaluator
├── observability/
│   └── langfuse_integration.py  # Langfuse tracing
├── docs/
│   ├── agent-price-action-evolution-plan.md   # Detailed agent evolution plan
│   └── future-architecture.md                 # Live trading architecture design
├── outputs/
│   ├── charts/              # Generated chart images
│   └── journal/             # Decision records and evaluation results
├── config.py                # Centralized configuration
├── main.py                  # Entry point
├── requirements.txt         # Python dependencies
├── docker-compose.yml       # Redis + Langfuse + Postgres
├── .env.example             # Environment variable template
└── README.md                # This file
```

---

## Roadmap

The agent evolution is organized into 9 phases:

| Phase | Focus | Key Deliverables |
|---|---|---|
| **0** | Planning | Architecture docs, feature branch |
| **1** | State & Context | Postgres state, context window fix, action semantics (BUY/SKIP/HOLD/EXIT), charges model, slippage, risk-based sizing, session controller, cooldown policy |
| **2** | Structured Output | Pydantic signal schemas, schema validation, structured output mode |
| **3** | Memory | Working/session/episodic/reflection memory, Postgres-backed memory store |
| **4** | Feedback & Reflection | Reflection writer, memory retrieval, calibration |
| **5** | Price Action Tools | VWAP, volume profile, market structure, liquidity zones, confluence scoring |
| **6** | Prompt v2 | Price-action checklist prompt with systematic reasoning |
| **7** | Agentic Workflow | LangGraph planner, enhanced tool loop (6-8 calls) |
| **8** | Experimentation | A/B testing, versioning, comparison reports |

See [`docs/agent-price-action-evolution-plan.md`](docs/agent-price-action-evolution-plan.md) for the full detailed plan.

### Future Live Trading Architecture

The long-term design evolves the POC harness into a production-grade live intraday trading system with WebSocket data ingestion, broker gateway, risk manager, circuit breaker, and Telegram alerts.

See [`docs/future-architecture.md`](docs/future-architecture.md) for the complete live system design.

---

## Research Foundations

AlphaPulse draws on several lines of academic and industry research:

| Paper / Concept | Key Idea |
|---|---|
| **FinMem** (arXiv:2311.13743) | Layered memory architecture for financial LLM agents |
| **ReAct** (arXiv:2210.03629) | Interleaved reasoning and tool use reduces hallucination |
| **Reflexion** (arXiv:2303.11366) | Agents improve via verbal feedback without fine-tuning |
| **FinAgent** (arXiv:2402.18485) | Tool-augmented multimodal agent with dual-level reflection |
| **VWAP** | Session-level volume-weighted price reference |
| **Volume Profile** | Market profile: POC, VAH, VAL, HVN, LVN |
| **LangGraph** | Graph-based agent orchestration with persistence |

---

## License

This project is for research and educational purposes.
