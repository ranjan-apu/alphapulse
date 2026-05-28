# AlphaPulse — Future Architecture: Live Intraday Trading Agent

> **Status**: Proposed Design Document  
> **Target**: Evolve from the current backtest POC harness into a production-grade live intraday trading system  
> **Guiding Principle**: The LLM proposes, the system disposes. Code owns arithmetic, risk, and execution. The LLM owns direction, area, and trigger reasoning.

---

## Table of Contents

1. [Architectural Shift: POC → Live](#1-architectural-shift-poc--live)
2. [System Overview](#2-system-overview)
3. [Layer 1: Data Ingestion](#3-layer-1-data-ingestion)
4. [Layer 2: Market State Engine](#4-layer-2-market-state-engine)
5. [Layer 3: Decision Agent](#5-layer-3-decision-agent)
6. [Layer 4: Broker Gateway & Order Executor](#6-layer-4-broker-gateway--order-executor)
7. [Layer 5: Risk Manager](#7-layer-5-risk-manager)
8. [Layer 6: Monitoring & Alerts](#8-layer-6-monitoring--alerts)
9. [Layer 7: Backtesting Engine (Shared Core)](#9-layer-7-backtesting-engine-shared-core)
10. [System Runtime Architecture](#10-system-runtime-architecture)
11. [Data & Control Flow](#11-data--control-flow)
12. [Migration Path: POC → Live](#12-migration-path-poc--live)
13. [Directory Structure](#13-directory-structure)
14. [Key Design Decisions](#14-key-design-decisions)
15. [Out of Scope (For Now)](#15-out-of-scope-for-now)

---

## 1. Architectural Shift: POC → Live

| Dimension | Current (POC Harness) | Future (Live Trading) |
|---|---|---|
| **Time Model** | Walk-forward replay clock | Real wall clock, event-driven |
| **Data Source** | Yahoo Finance (polling, CSV) | WebSocket feeds (Zerodha, Polygon, etc.) |
| **Execution** | Journal-only (no orders) | Broker gateway with order management |
| **Risk** | Post-hoc validation (2:1 R:R check) | Pre-trade & real-time risk manager with circuit breaker |
| **State** | In-memory, Redis for caching | Persistent, crash-recoverable |
| **Agent Loop** | Synchronous, blocking | Async, non-blocking with timeout |
| **Position Mgmt** | Open/close only | Full lifecycle: open → monitor → trail → close |
| **Backtesting** | The entire system | Shared core with pluggable data source & clock |
| **Observability** | Langfuse traces | Langfuse + Telegram alerts + real-time dashboard |
| **Reliability** | Single-shot run | Supervised process with graceful shutdown & recovery |

---

## 2. System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        AlphaPulse Live System                        │
│                                                                     │
│  ┌────────────────────┐    ┌────────────────┐    ┌───────────────┐  │
│  │   Data Ingestion   │───▶│ Market State   │───▶│   Decision    │  │
│  │   (WebSockets)     │    │   Engine       │    │   Agent (LLM) │  │
│  └────────────────────┘    └───────┬────────┘    └───────┬───────┘  │
│                                    │                      │          │
│                                    ▼                      ▼          │
│  ┌────────────────────┐    ┌────────────────┐    ┌───────────────┐  │
│  │   Broker Gateway   │◀───│  Risk Manager  │◀───│   Order       │  │
│  │   (Zerodha/Angel)  │    │  (Circuit      │    │   Executor    │  │
│  │                    │    │   Breaker)     │    │               │  │
│  └─────────┬──────────┘    └────────────────┘    └───────────────┘  │
│            │                                                         │
│            ▼                                                         │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │              Monitoring & Observability                      │    │
│  │  (Telegram Alerts, Real-time Dashboard, Langfuse, Journal)   │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

                        Shared Core (Backtest ↔ Live)
┌─────────────────────────────────────────────────────────────────────┐
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐  ┌─────────┐  │
│  │ DART Agent  │  │ Trade Calc   │  │ Validator    │  │ Journal │  │
│  │ (Prompts +  │  │ (Math)       │  │ (Signal      │  │ (JSONL) │  │
│  │  Tools)     │  │              │  │  Checks)     │  │         │  │
│  └─────────────┘  └──────────────┘  └──────────────┘  └─────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Layer 1: Data Ingestion

### 3.1 Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                         Data Ingestion Layer                         │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐            │
│  │ WebSocket    │   │ WebSocket    │   │ REST Poller  │            │
│  │ Provider A   │   │ Provider B   │   │ (Fallback)   │            │
│  │ (e.g.,       │   │ (e.g.,       │   │              │            │
│  │  Zerodha     │   │  Polygon.io) │   │ - yfinance   │            │
│  │  Kite Con-   │   │              │   │ - Alpha      │            │
│  │  nect)       │   │              │   │   Vantage    │            │
│  └──────┬───────┘   └──────┬───────┘   └──────┬───────┘            │
│         │                  │                   │                     │
│         └──────────────────┼───────────────────┘                     │
│                            ▼                                         │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                    Tick Normalizer                             │  │
│  │  - Unified Tick schema: {symbol, price, volume, timestamp}    │  │
│  │  - Deduplication (same-tick guard)                            │  │
│  │  - Timestamp standardization to UTC                           │  │
│  └──────────────────────────┬────────────────────────────────────┘  │
│                             ▼                                        │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                Tick → OHLCV Aggregator                        │  │
│  │                                                                │  │
│  │  Configurable timeframes:                                      │  │
│  │  ┌────────┬──────────┬───────────┬──────────┐                 │  │
│  │  │ 1 min  │  5 min   │  15 min   │  1 hour  │                 │  │
│  │  ├────────┼──────────┼───────────┼──────────┤                 │  │
│  │  │ 1-tick │  Session │  Daily    │  Weekly  │                 │  │
│  │  └────────┴──────────┴───────────┴──────────┘                 │  │
│  │                                                                │  │
│  │  Algorithm:                                                    │  │
│  │  1. On each tick, update current candle:                       │  │
│  │     - open = first tick price                                  │  │
│  │     - high = max(high, tick price)                             │  │
│  │     - low = min(low, tick price)                               │  │
│  │     - close = latest tick price                                │  │
│  │     - volume += tick volume                                    │  │
│  │  2. On timeframe boundary, close candle & start new one        │  │
│  │  3. Emit CandleClosed event to subscribers                     │  │
│  └──────────────────────────┬────────────────────────────────────┘  │
│                             ▼                                        │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                   Candle Buffer (Ring Buffer)                  │  │
│  │                                                                │  │
│  │  Per timeframe, in-memory ring buffer:                         │  │
│  │  - Last N candles (configurable, default 200)                  │  │
│  │  - O(1) append, O(1) slice for "last X candles"               │  │
│  │  - Redis snapshot every M minutes for crash recovery            │  │
│  │  - On restart, restore from Redis + backfill missing period    │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### 3.2 Interface

```python
class DataSource(ABC):
    """Abstract data source — all providers implement this."""

    @abstractmethod
    async def connect(self):
        """Establish connection (WebSocket handshake, auth, etc.)."""

    @abstractmethod
    async def subscribe(self, symbols: list[str], on_tick: Callable[[Tick], None]):
        """Subscribe to real-time tick data for given symbols."""

    @abstractmethod
    async def backfill(self, symbol: str, timeframe: str,
                       start: datetime, end: datetime) -> list[Candle]:
        """Fetch historical candles for warm-up context."""

    @abstractmethod
    async def disconnect(self):
        """Gracefully close connection."""


@dataclass
class Tick:
    symbol: str
    price: float
    volume: int
    timestamp: datetime


@dataclass
class Candle:
    symbol: str
    timeframe: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    timestamp: datetime       # open time
    closed_at: datetime       # close time
    is_complete: bool         # False = still forming
```

### 3.3 Key Design Decisions

- **Abstract `DataSource` interface** → swap providers without touching anything else
- **Build candles from ticks** → no polling latency, exact session boundaries
- **Ring buffer** → bounded memory, fast slice operations, no unbounded growth
- **Redis persistence** → survive a crash without losing all context
- **Backfill on startup** → agent always has full historical context even after restart

---

## 4. Layer 2: Market State Engine

### 4.1 Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                       Market State Engine                            │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────┐    ┌──────────────────┐    ┌───────────────┐  │
│  │ Candle Aggregator│    │ Indicator Engine │    │ Level Engine  │  │
│  │                  │    │                  │    │               │  │
│  │ - Builds higher  │    │ - Incremental    │    │ - Swing       │  │
│  │   timeframe      │    │   updates on     │    │   detection   │  │
│  │   candles from   │    │   new candle     │    │ - S/R levels  │  │
│  │   lower TF       │    │ - RSI, ATR, MA   │    │ - Pattern     │  │
│  │                  │    │   slopes, volume │    │   recognition │  │
│  │ - E.g., 5m → 15m│    │ - No recalcula-  │    │ - Multi-TF    │  │
│  │   15m → 1h      │    │   tion from      │    │   confluence  │  │
│  └────────┬─────────┘    │   scratch        │    └───────┬───────┘  │
│           │              └────────┬─────────┘            │          │
│           └──────────────────────┼───────────────────────┘          │
│                                  ▼                                   │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │              Market State Builder                              │  │
│  │                                                                 │  │
│  │  On CandleClosed event:                                         │  │
│  │  1. Update higher-TF candles (if needed)                       │  │
│  │  2. Update indicators incrementally                            │  │
│  │  3. Update swing points and S/R levels                         │  │
│  │  4. Detect price-action pattern                                │  │
│  │  5. Compile MarketStatePackage                                 │  │
│  │  6. Publish to subscribers (event bus)                         │  │
│  │                                                                 │  │
│  │  Cache: previous state is cached so delta is cheap             │  │
│  └──────────────────────────┬────────────────────────────────────┘  │
│                             ▼                                        │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │              State Publisher (Pub/Sub)                         │  │
│  │                                                                 │  │
│  │  Events emitted:                                                │  │
│  │  ┌────────────────────┬─────────────────────────────────────┐  │  │
│  │  │ Event              │ Payload                             │  │  │
│  │  ├────────────────────┼─────────────────────────────────────┤  │  │
│  │  │ candle_closed      │ Candle (timeframe, OHLCV)           │  │  │
│  │  │ state_updated      │ MarketStatePackage (full snapshot)  │  │  │
│  │  │ level_updated      │ New S/R levels detected             │  │  │
│  │  │ pattern_detected   │ Pattern label + confidence          │  │  │
│  │  │ session_start      │ Session metadata                    │  │  │
│  │  │ session_end        │ Final state                         │  │  │
│  │  └────────────────────┴─────────────────────────────────────┘  │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### 4.2 Incremental Indicator Updates

Rather than recalculating indicators from all historical data on every candle:

```python
class IncrementalRSI:
    """Wilder-smoothed RSI updated incrementally per new candle."""

    def __init__(self, period: int = 14):
        self.period = period
        self.avg_gain = None
        self.avg_loss = None
        self.prev_close = None
        self.values: list[float] = []
        self._warmup: list[float] = []

    def update(self, close: float) -> float:
        if self.prev_close is None:
            self.prev_close = close
            return 50.0

        delta = close - self.prev_close
        gain = max(delta, 0)
        loss = max(-delta, 0)
        self.prev_close = close

        if self.avg_gain is None:
            # Warmup phase: accumulate for SMA initialization
            self._warmup.append((gain, loss))
            if len(self._warmup) >= self.period:
                gains = [g for g, _ in self._warmup]
                losses = [l for _, l in self._warmup]
                self.avg_gain = sum(gains) / self.period
                self.avg_loss = sum(losses) / self.period
                self._warmup = []
        else:
            # Wilder smoothing
            self.avg_gain = (self.avg_gain * (self.period - 1) + gain) / self.period
            self.avg_loss = (self.avg_loss * (self.period - 1) + loss) / self.period

        if self.avg_loss == 0:
            rsi = 100.0
        else:
            rs = self.avg_gain / self.avg_loss
            rsi = 100.0 - (100.0 / (1.0 + rs))

        self.values.append(rsi)
        return rsi
```

### 4.3 Key Design Decisions

- **Incremental computation** — O(1) per new candle instead of O(N). Critical for real-time.
- **Event-driven publishing** — subscribers don't poll; they react to events.
- **Full snapshot on state_updated** — new subscribers get the complete picture immediately.
- **Pre-computed context** — by the time the LLM is called, the state is already built.

---

## 5. Layer 3: Decision Agent

### 5.1 Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                          Decision Agent                              │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                  Agent Orchestrator                           │   │
│  │                                                                │   │
│  │  On state_updated event:                                       │   │
│  │  1. If circuit breaker is active → skip                       │   │
│  │  2. If cooldown active → skip                                 │   │
│  │  3. If position open → delegate to Position Monitor           │   │
│  │  4. Prepare LLM context from MarketStatePackage               │   │
│  │  5. Call LLM (async, with timeout)                             │   │
│  │  6. Parse & validate response                                 │   │
│  │  7. If BUY/SELL → pass to Risk Manager                        │   │
│  │  8. Journal decision                                          │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌────────────────────┐    ┌──────────────────────────────────┐     │
│  │  Position Monitor  │    │  Tool Harness (live version)     │     │
│  │                    │    │                                   │     │
│  │  - Tracks open     │    │  - Same tools as backtest        │     │
│  │    positions       │    │  - Data sources are real-time    │     │
│  │  - On each tick:   │    │  - No historical replay filter   │     │
│  │    check stop      │    │  - charting produces live images │     │
│  │    check target    │    │                                   │     │
│  │    check time      │    └──────────────────────────────────┘     │
│  │  - On hit: send    │                                             │
│  │    exit order      │    ┌──────────────────────────────────┐     │
│  │  - Manage trailing  │    │  Context Pre-computer           │     │
│  │    stop             │    │                                   │     │
│  └────────────────────┘    │  - Pre-compute context in        │     │
│                            │    background on each candle      │     │
│  ┌────────────────────┐    │  - Agent has zero-latency access │     │
│  │  Cooldown Manager  │    │    to latest state               │     │
│  │                    │    └──────────────────────────────────┘     │
│  │  - Min time between│                                             │
│  │    signals         │    ┌──────────────────────────────────┐     │
│  │  - Max trades per  │    │  LLM Call Manager               │     │
│  │    session         │    │                                   │     │
│  │  - Max trades per  │    │  - Async HTTP with timeout       │     │
│  │    instrument      │    │  - Retry with backoff (max 2)    │     │
│  └────────────────────┘    │  - Token usage tracking          │     │
│                            │  - Fallback HOLD on timeout      │     │
│                            └──────────────────────────────────┘     │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### 5.2 Agent Decision Flow

```
                    ┌──────────────────┐
                    │ New Candle Closed │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Position Open?   │──── Yes ──▶ Position Monitor
                    └────────┬─────────┘              (check stop/target)
                             │ No                      │
                             ▼                         ▼
                    ┌──────────────────┐         ┌──────────┐
                    │ Cooldown Active? │──Yes──▶ │ SKIP     │
                    └────────┬─────────┘         └──────────┘
                             │ No
                             ▼
                    ┌──────────────────┐
                    │ Circuit Breaker? │──Yes──▶ │ SKIP     │
                    └────────┬─────────┘         └──────────┘
                             │ No
                             ▼
                    ┌──────────────────────────────────────┐
                    │ Prepare LLM Context from pre-computed│
                    │ MarketStatePackage                   │
                    └────────────────┬─────────────────────┘
                                     ▼
                    ┌──────────────────────────────────────┐
                    │ Call LLM (async, 10s timeout)        │
                    └────────────────┬─────────────────────┘
                                     │
                    ┌────────────────▼─────────────────────┐
                    │ Parse Response                       │
                    │  - Valid JSON?                       │
                    │  - Has action field?                 │
                    │  - Is action BUY/SELL/HOLD?          │
                    └────────────────┬─────────────────────┘
                                     │
              ┌──────────────────────┼──────────────────────┐
              ▼                      ▼                      ▼
     ┌────────────────┐   ┌──────────────────┐   ┌────────────────┐
     │     HOLD       │   │   BUY / SELL     │   │ Parse Error    │
     │  → Journal     │   │  → Pre-trade     │   │ → HOLD + error │
     │  → Skip        │   │    Risk Checks   │   │ → Journal      │
     └────────────────┘   └────────┬─────────┘   └────────────────┘
                                   │
                          ┌────────▼────────┐
                          │ Risk Manager    │
                          │ Pass?           │
                          └────────┬────────┘
                                   │
                    ┌──────────────┼──────────────┐
                    ▼              ▼              ▼
            ┌────────────┐ ┌────────────┐ ┌────────────┐
            │ Accepted   │ │ Rejected   │ │ Rejected   │
            │ → Send     │ │ → Journal  │ │ → Journal  │
            │   Order    │ │   with     │ │   with     │
            │ → Open Pos │ │   reason   │ │   reason   │
            └────────────┘ └────────────┘ └────────────┘
```

### 5.3 Key Design Decisions

- **Async LLM call** — the agent never blocks the event loop. A slow LLM doesn't delay tick processing.
- **Timeout-protected** — if the LLM takes more than 5-10s, emit HOLD and continue. Missed decisions are better than stale ones.
- **Pre-computed context** — the MarketStatePackage is ready before the LLM is called. No time spent building context.
- **Position monitor runs separately** — it checks stop/target on every tick using deterministic code, not the LLM.
- **Same ToolHarness as backtest** — tools like `calculate_trade_math`, `find_levels` work identically live and in backtest. Only the data source changes.

---

## 6. Layer 4: Broker Gateway & Order Executor

### 6.1 Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                       Broker Gateway Layer                           │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                   Broker Abstraction                          │   │
│  │                                                                │   │
│  │                    ┌─────────────────┐                        │   │
│  │                    │  BrokerGateway  │  (Interface)           │   │
│  │                    │  (ABC)          │                        │   │
│  │                    └──────┬──────────┘                        │   │
│  │                           │                                    │   │
│  │              ┌────────────┼────────────┐                      │   │
│  │              ▼            ▼            ▼                      │   │
│  │      ┌────────────┐┌────────────┐┌────────────┐              │   │
│  │      │ Zerodha    ││ Angel One ││ Paper      │              │   │
│  │      │ KiteCon-   ││ SmartAPI  ││ Broker     │              │   │
│  │      │ nect       ││           ││ (Simulator)│              │   │
│  │      └────────────┘└────────────┘└────────────┘              │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    Order Manager                              │   │
│  │                                                                │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐    │   │
│  │  │ Order Router │  │ Order Book   │  │ Fill             │    │   │
│  │  │              │  │              │  │ Reconciliation   │    │   │
│  │  │ Routes order │  │ - pending    │  │                  │    │   │
│  │  │ to correct   │  │ - filled     │  │ - Compare local  │    │   │
│  │  │ exchange/    │  │ - cancelled  │  │   position with  │    │   │
│  │  │ product type │  │ - rejected   │  │   broker's view  │    │   │
│  │  │              │  │              │  │ - Auto-retry     │    │   │
│  │  └──────────────┘  └──────────────┘  │   mismatches     │    │   │
│  │                                       └──────────────────┘    │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    Position Sizer                             │   │
│  │                                                                │   │
│  │  Strategies (configurable):                                    │   │
│  │  - Fixed capital per trade (e.g., ₹30,000)                    │   │
│  │  - % of account equity (e.g., 10% of ₹2L = ₹20,000)          │   │
│  │  - Risk-based: quantity = max_risk_per_trade / risk_per_share  │   │
│  │  - Fixed quantity (for testing)                               │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### 6.2 BrokerGateway Interface

```python
class BrokerGateway(ABC):
    """Abstract broker interface — all brokers implement this."""

    @abstractmethod
    async def connect(self) -> bool:
        """Authenticate and establish session."""

    @abstractmethod
    async def place_order(self, order: Order) -> OrderResult:
        """Place an order (MARKET, LIMIT, SL-M)."""

    @abstractmethod
    async def modify_order(self, order_id: str, updates: dict) -> OrderResult:
        """Modify an existing order (price, quantity, etc.)."""

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order."""

    @abstractmethod
    async def get_positions(self) -> list[Position]:
        """Get current open positions."""

    @abstractmethod
    async def get_order_book(self) -> list[OrderResult]:
        """Get all orders for the day."""

    @abstractmethod
    async def get_account_summary(self) -> AccountSummary:
        """Get margin, equity, available cash."""

    @abstractmethod
    async def subscribe_orders(self, on_update: Callable[[OrderUpdate], None]):
        """Subscribe to real-time order updates via WebSocket."""


@dataclass
class Order:
    symbol: str
    side: Literal["BUY", "SELL"]
    order_type: Literal["MARKET", "LIMIT", "SL", "SL-M"]
    quantity: int
    price: float | None = None      # for LIMIT orders
    trigger_price: float | None = None  # for SL/SL-M orders
    validity: str = "DAY"
    tag: str | None = None          # for tracking (e.g., "dart_signal_001")


@dataclass
class OrderResult:
    order_id: str
    status: Literal["PENDING", "OPEN", "FILLED", "PARTIALLY_FILLED",
                    "CANCELLED", "REJECTED"]
    filled_quantity: int = 0
    average_price: float | None = None
    reject_reason: str | None = None
    order_timestamp: datetime | None = None
    fill_timestamp: datetime | None = None
```

### 6.3 Paper Broker (Simulator)

The paper broker is critical — it simulates fills against live market data:

```python
class PaperBroker(BrokerGateway):
    """
    Simulated broker that fills orders against real-time market data.
    Same code path as live broker — the agent doesn't know the difference.
    """

    def __init__(self, initial_capital: float = 100_000):
        self.capital = initial_capital
        self.positions: dict[str, Position] = {}
        self.orders: dict[str, OrderResult] = {}
        self.order_id_counter = 0

    async def place_order(self, order: Order) -> OrderResult:
        # Validate: sufficient capital, correct product type, etc.
        # If MARKET: fill immediately at next tick
        # If LIMIT: fill when tick price crosses limit
        # Return OrderResult with status "PENDING" or "FILLED"
        ...

    async def on_tick(self, tick: Tick):
        """Called by Data Ingestion on every tick."""
        # Check pending LIMIT orders → fill if price matches
        # Check SL orders → convert to market order on trigger
        # Update open position P&L
        ...
```

### 6.4 Key Design Decisions

- **Paper broker is a first-class citizen** — the same code runs in paper and live mode. Only the `BrokerGateway` implementation changes.
- **Order book tracks everything** — all order state is in memory and persisted to Redis/DB.
- **Fill reconciliation** — periodically compare local order state with broker's state. Detect missed fills.
- **Position sizer is pluggable** — different sizing strategies for different account types or risk profiles.

---

## 7. Layer 5: Risk Manager

### 7.1 Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                          Risk Manager                                │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  The Risk Manager is the ONLY layer that can veto a decision.       │
│  It has the final say before any order reaches the broker.          │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    Pre-Trade Checks                          │   │
│  │                                                                │   │
│  │  When a BUY/SELL signal arrives:                              │   │
│  │                                                                │   │
│  │  □ Account Sufficient Margin                                  │   │
│  │    - Check required margin vs available                       │   │
│  │                                                                │   │
│  │  □ Daily Loss Limit Not Hit                                   │   │
│  │    - Track intrasession P&L                                   │   │
│  │    - If loss exceeds threshold (e.g., ₹5,000) → block         │   │
│  │                                                                │   │
│  │  □ Max Drawdown Not Hit                                       │   │
│  │    - Track peak capital → current equity                      │   │
│  │    - If drawdown exceeds threshold → block                     │   │
│  │                                                                │   │
│  │  □ Max Concurrent Positions Not Exceeded                      │   │
│  │    - Hard limit on open positions                             │   │
│  │                                                                │   │
│  │  □ Max Exposure % Not Exceeded                                │   │
│  │    - Total deployed capital / account equity ≤ limit          │   │
│  │                                                                │   │
│  │  □ Not in Cooldown                                            │   │
│  │    - Min time since last trade                                │   │
│  │    - Max trades per session                                   │   │
│  │                                                                │   │
│  │  □ Instrument Not Blacklisted                                 │   │
│  │    - Circuit filter, low liquidity, etc.                      │   │
│  │                                                                │   │
│  │  □ Session Time Remaining                                     │   │
│  │    - Enough time for trade to resolve                         │   │
│  │    - Enforce EOD square-off                                   │   │
│  │                                                                │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    Post-Trade Monitoring                      │   │
│  │                                                                │   │
│  │  After a position is opened:                                   │   │
│  │  - Track running P&L per position                             │   │
│  │  - Track running P&L per session                              │   │
│  │  - Update drawdown from peak                                  │   │
│  │  - If daily loss limit hit → trigger circuit breaker          │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    Circuit Breaker                             │   │
│  │                                                                │   │
│  │  States:                                                       │   │
│  │  ┌─────────┐    ┌──────────┐    ┌────────────┐               │   │
│  │  │ NORMAL  │───▶│ WARNING  │───▶│ TRIPPED    │               │   │
│  │  │         │    │ (alerts  │    │ (squares   │               │   │
│  │  │         │    │  sent)   │    │  off all,  │               │   │
│  │  │         │◀───│          │◀───│  halts     │               │   │
│  │  └─────────┘    └──────────┘    │  trading)  │               │   │
│  │                                  └────────────┘               │   │
│  │                                                                │   │
│  │  Trigger conditions:                                           │   │
│  │  - Daily loss limit exceeded                                   │   │
│  │  - Max drawdown exceeded                                       │   │
│  │  - Broker connection lost                                      │   │
│  │  - Manual kill switch (Telegram command)                       │   │
│  │  - N consecutive losses (configurable)                         │   │
│  │                                                                │   │
│  │  Recovery:                                                     │   │
│  │  - Manual reset only (no auto-recovery)                       │   │
│  │  - Requires human confirmation via Telegram/dashboard          │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### 7.2 Risk Manager Interface

```python
class RiskManager:
    """
    Gatekeeper for all trading decisions.
    Veto power over every order before it reaches the broker.
    """

    def __init__(self, config: RiskConfig):
        self.config = config
        self.daily_pnl = 0.0
        self.peak_capital = config.initial_capital
        self.current_equity = config.initial_capital
        self.open_positions: dict[str, Position] = {}
        self.trades_today = 0
        self.state = CircuitBreakerState.NORMAL

    async def check_pre_trade(self, signal: Signal,
                              account: AccountSummary) -> RiskVerdict:
        """
        Evaluate a signal against all risk rules.
        Returns ACCEPT or REJECT with reason.
        """
        checks = [
            self._check_margin(signal, account),
            self._check_daily_loss_limit(),
            self._check_max_drawdown(),
            self._check_max_positions(),
            self._check_exposure(signal, account),
            self._check_cooldown(),
            self._check_blacklist(signal),
            self._check_session_end(signal),
        ]
        failures = [r for r in checks if r is not None]
        if failures:
            return RiskVerdict.REJECT(", ".join(failures))
        return RiskVerdict.ACCEPT

    async def check_post_trade(self, position: Position, tick: Tick):
        """Continuous monitoring of open positions."""
        # Update running P&L
        # Check if daily loss limit hit
        # Update drawdown
        ...

    def trip_circuit_breaker(self, reason: str):
        """Emergency stop — squares off all positions, halts trading."""
        ...

    def reset_daily(self):
        """End-of-day reset for daily counters."""
        self.daily_pnl = 0.0
        self.trades_today = 0
        # Do NOT reset peak_capital — that's rolling
```

### 7.3 Key Design Decisions

- **Hard limits in code** — the LLM cannot override risk limits. The Risk Manager is the final authority.
- **Circuit breaker is automatic and manual** — trips automatically on thresholds, can also be triggered via Telegram command.
- **Daily loss cap is non-negotiable** — once hit, trading stops for the day regardless of subsequent signals.
- **Peak drawdown is rolling** — it tracks from the highest equity point, not from start of day.

---

## 8. Layer 6: Monitoring & Alerts

### 8.1 Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                    Monitoring & Observability                        │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐  │
│  │ Telegram Bot     │  │ Real-time        │  │ Audit Logger     │  │
│  │                  │  │ Dashboard        │  │                  │  │
│  │ - Order fill     │  │ (Streamlit)      │  │ - Every signal   │  │
│  │   notifications  │  │                  │  │ - Every order    │  │
│  │ - Signal alerts  │  │ - Equity curve   │  │ - Every error    │  │
│  │ - Error alerts   │  │ - Open positions │  │ - JSONL + SQLite │  │
│  │ - Kill switch    │  │ - P&L by trade   │  │ - Queryable      │  │
│  │ - Daily summary  │  │ - Agent status   │  │                  │  │
│  │                  │  │ - Risk metrics   │  │                  │  │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘  │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    Langfuse Tracing                           │   │
│  │  - Root spans per decision                                   │   │
│  │  - Generation spans for LLM calls                            │   │
│  │  - Tool call spans                                           │   │
│  │  - Validation & risk check spans                             │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    Journal (JSONL)                             │   │
│  │  - Append-only, same format as current POC                   │   │
│  │  - Every decision + context snapshot + outcome               │   │
│  │  - Can be replayed into evaluator later                     │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### 8.2 Alert Levels

| Priority | Channel | Examples |
|---|---|---|
| 🔴 Critical | Telegram (immediate) + Log | Circuit breaker tripped, broker disconnect, daily loss limit |
| 🟡 Warning | Telegram (1 min delay) + Log | Max drawdown warning, N consecutive losses, margin low |
| 🔵 Info | Telegram (summary) + Log | Order filled, position closed, session summary |
| ⚪ Debug | Log only | Signal details, tool calls, indicator values |

---

## 9. Layer 7: Backtesting Engine (Shared Core)

### 9.1 Architecture

The key insight: **the agent, tools, validator, and trade calculator should be identical in backtest and live mode.** Only the clock, data source, and broker change.

```
┌──────────────────────────────────────────────────────────────────────┐
│                        Shared Core Modules                           │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │ dart_agent   │  │ trade_calc   │  │ validator    │              │
│  │              │  │              │  │              │              │
│  │ - prompts    │  │ - sizing     │  │ - signal     │              │
│  │ - tool       │  │ - charges    │  │   structure  │              │
│  │   harness    │  │ - risk math  │  │ - level      │              │
│  │ - LLM client │  │ - RR calc    │  │   checks     │              │
│  └──────────────┘  └──────────────┘  └──────────────┘              │
│                                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │ summarizer   │  │ charts       │  │ journal      │              │
│  │              │  │              │  │              │              │
│  │ - indicators │  │ - chart gen  │  │ - JSONL      │              │
│  │ - swings     │  │ - dashboard  │  │ - append     │              │
│  │ - patterns   │  │              │  │ - summary    │              │
│  │ - levels     │  │              │  │              │              │
│  └──────────────┘  └──────────────┘  └──────────────┘              │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘

                ┌────────────────────┬────────────────────┐
                │                    │                    │
                ▼                    ▼                    ▼
┌───────────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│  Backtest Runner     │  │  Paper Trading   │  │  Live Trading    │
│                       │  │  Runner          │  │  Runner          │
│  - WalkForwardClock  │  │  - Real clock    │  │  - Real clock    │
│  - CSV / DB data     │  │  - DataSource    │  │  - DataSource    │
│  - No broker         │  │  - PaperBroker   │  │  - Real broker   │
│  - Evaluator         │  │  - RiskManager   │  │  - RiskManager   │
│                       │  │  - Alerts        │  │  - Alerts        │
└───────────────────────┘  └──────────────────┘  └──────────────────┘
```

### 9.2 Pluggable Clock

```python
class Clock(ABC):
    """Abstract clock — provides decision timestamps."""

    @abstractmethod
    async def next_decision(self) -> DecisionPoint | None:
        """Get the next decision point. None when complete."""


class WalkForwardClock(Clock):
    """Historical replay clock (uses current POC logic)."""
    ...


class LiveClock(Clock):
    """Real wall clock — emits events at candle boundaries."""
    async def next_decision(self) -> DecisionPoint:
        # Wait until the next decision-interval boundary
        # (e.g., top of each 15-min candle)
        now = datetime.now()
        next_candle = ceil_to_interval(now, config.DECISION_INTERVAL)
        await sleep_until(next_candle)
        return DecisionPoint(time=next_candle, ...)
```

---

## 10. System Runtime Architecture

### 10.1 Process Topology

```
┌──────────────────────────────────────────────────────────────────────┐
│                    Main Supervisor Process                           │
│                    (asyncio event loop)                              │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                   Orchestrator                                 │   │
│  │  - Initializes all components                                 │   │
│  │  - Manages startup/shutdown sequence                          │   │
│  │  - Handles SIGINT/SIGTERM gracefully                          │   │
│  │  - Routes events between components                            │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │ Data Ingest  │  │ Market State │  │ LLM Agent    │              │
│  │ (asyncio     │  │ (asyncio     │  │ (asyncio     │              │
│  │  task)       │  │  task)       │  │  task)       │              │
│  │              │  │              │  │              │              │
│  │ WebSocket    │  │ Computes     │  │ Listens for  │              │
│  │ → Tick queue │  │ indicators   │  │ state events │              │
│  │ → Candle     │  │ → emits      │  │ → calls LLM  │              │
│  │   aggregator │  │   state      │  │ → sends      │              │
│  │              │  │   events     │  │   signals    │              │
│  └──────────────┘  └──────────────┘  └──────────────┘              │
│                                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │ Risk Manager │  │ Broker       │  │ Position     │              │
│  │ (async,      │  │ Gateway      │  │ Monitor      │              │
│  │  sync core)  │  │ (asyncio)    │  │ (asyncio     │              │
│  │              │  │              │  │  task)       │              │
│  │ - Pre-trade  │  │ - Place      │  │              │              │
│  │   checks     │  │   orders     │  │ - Check stop │              │
│  │ - Post-trade │  │ - Track      │  │   /target on │              │
│  │   monitoring │  │   fills      │  │   every tick │              │
│  │ - Circuit    │  │ - Subscribe  │  │ - Trail stop │              │
│  │   breaker    │  │   orders     │  │   logic      │              │
│  └──────────────┘  └──────────────┘  └──────────────┘              │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │              Message Bus (asyncio.Queue)                       │   │
│  │                                                                │   │
│  │  Events:                                                       │   │
│  │  - tick.{symbol}             → Tick                           │   │
│  │  - candle_closed.{tf}        → Candle                         │   │
│  │  - state_updated             → MarketStatePackage             │   │
│  │  - signal.{action}           → Signal                         │   │
│  │  - order.{status}            → OrderUpdate                    │   │
│  │  - risk.alert.{level}        → RiskAlert                      │   │
│  │  - error.{component}         → Error                          │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │ Telegram     │  │ Dashboard    │  │ Journal      │              │
│  │ Notifier     │  │ (if enabled) │  │ (always on)  │              │
│  │              │  │              │  │              │              │
│  │ Subscribes   │  │ Subscribes   │  │ Subscribes   │              │
│  │ to events    │  │ to events    │  │ to events    │              │
│  └──────────────┘  └──────────────┘  └──────────────┘              │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### 10.2 Startup Sequence

```
1. Parse config & validate .env
2. Initialize logging
3. Initialize Redis/DB
4. Create DataSource (based on mode: live/paper/backtest)
5. Create BrokerGateway (based on mode)
6. Initialize RiskManager with config
7. Create MarketStateEngine (subscribes to candles from DataSource)
8. Create Agent (subscribes to state events)
9. Create PositionMonitor (subscribes to ticks)
10. Create Telegram notifier, Dashboard, Journal
11. Backfill historical context (last N days of candles)
12. Enter main event loop:
    - Process events from bus
    - Heartbeat check every 5s
    - Watchdog for stale components
13. On SIGINT/SIGTERM:
    - Square off all positions
    - Flush journal
    - Flush Langfuse
    - Disconnect broker
    - Disconnect data source
    - Save state to Redis
    - Exit
```

---

## 11. Data & Control Flow

### 11.1 Normal Flow (No Position)

```
Tick ──▶ Data Ingestion ──▶ Candle Aggregator ──▶ CandleClosed event
                                                       │
                                                       ▼
                                              MarketStateEngine
                                              (updates indicators,
                                               levels, patterns)
                                                       │
                                                       ▼
                                              StateUpdated event
                                                       │
                                            ┌──────────┴──────────┐
                                            ▼                     ▼
                                     Agent receives        Journal records
                                     market state          state snapshot
                                            │
                                            ▼
                                     LLM called (async)
                                            │
                                            ▼
                                     Signal returned
                                            │
                                            ▼
                                     Validator (structure check)
                                            │
                                            ▼
                                     Risk Manager (pre-trade)
                                            │
                                    ┌───────┴───────┐
                                    ▼               ▼
                              Accepted         Rejected
                                    │               │
                                    ▼               ▼
                            BrokerGateway    Journal (reason)
                              (place order)
                                    │
                                    ▼
                              OrderUpdate event
                                    │
                                    ▼
                              Position Monitor
                              (starts tracking)
```

### 11.2 Position Management Flow

```
Every Tick ──▶ Position Monitor
                    │
                    │  Check: is stop hit?
                    │  Check: is target hit?
                    │  Check: is session ending?
                    │  Check: should trail stop?
                    │
                    ├── Stop hit ──▶ Exit order (MARKET)
                    │                   │
                    │                   ▼
                    │              Journal (loss record)
                    │
                    ├── Target hit ──▶ Exit order (LIMIT/MARKET)
                    │                    │
                    │                    ▼
                    │               Journal (win record)
                    │
                    ├── Session end ──▶ Square off
                    │                      │
                    │                      ▼
                    │                 Journal (sq-off record)
                    │
                    └── Trail check ──▶ Modify SL order
                                         │
                                         ▼
                                    OrderUpdate event
```

### 11.3 Error & Circuit Breaker Flow

```
                    ┌──────────────────┐
                    │ Component Error  │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Log + Alert      │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │ Can recover?     │
                    └────────┬─────────┘
                             │
               ┌─────────────┼─────────────┐
               ▼             ▼             ▼
         Auto-retry   Graceful degrade  Trip circuit
         (max 3x)     (disable agent,   breaker
                       keep position    (square off,
                        monitor)         halt trading)
               │             │             │
               ▼             ▼             ▼
          Success or     Alert admin    Alert admin
          re-raise                       (requires manual
                                         reset next day)
```

---

## 12. Migration Path: POC → Live

### Phase 1: Foundation (Week 1-2)

```
Target: Make the project maintainable and testable.
No behavioral changes. No new features.

Output:
├── pyproject.toml          # Installable package
├── Makefile                # Convenience commands
├── tests/                  # Unit tests for core math
├── src/alphapulse/         # Package structure
│   ├── trade_calc.py       # Shared trade math (extracted)
│   ├── config.py           # Same config, improved validation
│   └── ...                 # Existing modules, restructured
```

**Tasks:**
1. Add `pyproject.toml` with dependencies and entry points
2. Replace `print()` with `logging.getLogger(__name__)`
3. Extract `TradeCalculator` from `core/tools.py` and `validation/validator.py` into a shared module
4. Add `pytest` and write tests for `TradeCalculator`, indicator math, and validator
5. Add `Makefile` with targets: `install`, `test`, `run-backtest`, `lint`
6. Add `.pre-commit-config.yaml` (ruff, black)
7. Add strict type checking stub file

### Phase 2: Backtesting Parity (Week 3)

```
Target: Shared core modules that work in both backtest and live mode.
Add PaperBroker for simulated trading.

Output:
├── src/alphapulse/
│   ├── core/
│   │   ├── clock.py        # WalkForwardClock (backtest) + LiveClock (future)
│   │   ├── data_source.py  # Abstract DataSource interface
│   │   └── broker.py       # Abstract BrokerGateway + PaperBroker
│   ├── agent/              # Unchanged shared logic
│   ├── risk/               # RiskManager (initially same as validator)
│   └── backtest/
│       └── runner.py       # BacktestRunner using shared core
```

**Tasks:**
1. Define `DataSource` abstract interface
2. Define `BrokerGateway` abstract interface
3. Implement `PaperBroker` (fills against live/historical ticks)
4. Implement `LiveClock` (real wall clock)
5. Make `WalkForwardClock` implement the same `Clock` interface
6. Create `BacktestRunner` that uses shared Agent + Validator + PaperBroker
7. Validate that backtest results match current POC output

### Phase 3: Live Data (Week 4-5)

```
Target: Real-time data ingestion, incremental indicators, event-driven state.

Output:
├── src/alphapulse/
│   ├── data/
│   │   ├── zerodha.py      # Zerodha Kite Connect DataSource
│   │   ├── polygon.py      # Polygon.io DataSource
│   │   ├── aggregator.py   # Tick → OHLCV aggregator
│   │   └── buffer.py       # Ring buffer
│   ├── state/
│   │   ├── engine.py       # MarketStateEngine (event-driven)
│   │   └── indicators.py   # Incremental indicator implementations
│   └── live/
│       └── runner.py       # LiveRunner using shared core
```

**Tasks:**
1. Implement `ZerodhaDataSource` (WebSocket for ticks, REST for backfill)
2. Implement tick → OHLCV aggregator with configurable timeframes
3. Implement incremental indicators (RSI, ATR, MA slopes)
4. Build ring buffer for candle storage
5. Build event-driven MarketStateEngine
6. Test with paper trading against live data feed
7. Backfill on startup so agent has full context

### Phase 4: Broker Integration (Week 6-7)

```
Target: Real broker connectivity, order management, position monitoring.

Output:
├── src/alphapulse/
│   ├── broker/
│   │   ├── zerodha.py      # Zerodha Kite Connect BrokerGateway
│   │   ├── order_manager.py # Order book, routing, retry
│   │   └── position_sizer.py # Sizing strategies
│   ├── monitor/
│   │   └── position.py     # Stop/target tracker, trailing logic
│   └── live/
│       └── runner.py       # Updated with broker
```

**Tasks:**
1. Implement `ZerodhaBrokerGateway` using kiteconnect library
2. Implement order manager with retry + reconciliation
3. Implement position monitor (stop/target checks on every tick)
4. Add trailing stop logic
5. Implement paper trading mode for all broker operations
6. Test in paper mode for 1 week

### Phase 5: Production Hardening (Ongoing)

```
Target: Risk manager, alerts, dashboard, session recovery.

Output:
├── src/alphapulse/
│   ├── risk/
│   │   ├── manager.py      # RiskManager with all checks
│   │   └── circuit_breaker.py # Circuit breaker state machine
│   ├── monitoring/
│   │   ├── telegram.py     # Telegram bot
│   │   ├── dashboard.py    # Streamlit dashboard
│   │   └── alerts.py       # Alert routing
│   └── recovery/
│       └── state.py        # Redis state persistence & recovery
```

**Tasks:**
1. Implement full `RiskManager` with all pre-trade checks
2. Implement circuit breaker (auto + manual)
3. Add Telegram bot for alerts and kill switch
4. Build Streamlit dashboard
5. Add Redis state persistence for crash recovery
6. Add graceful shutdown handler
7. Add daily session summary report
8. Run in shadow mode alongside manual trading

---

## 13. Directory Structure

```
alphapulse/
├── pyproject.toml              # Package metadata, dependencies, entry points
├── Makefile                    # install, test, run-backtest, run-live, lint
├── .pre-commit-config.yaml     # ruff, black, mypy
├── docker-compose.yml          # Redis + Langfuse (same as now)
├── .env.example                # Template for .env
├── README.md                   # Updated for live trading
│
├── src/
│   └── alphapulse/
│       ├── __init__.py
│       ├── config.py           # Unified config (backtest + live)
│       ├── trade_calc.py       # Shared trade math (extracted from tools + validator)
│       │
│       ├── core/               # Shared between backtest and live
│       │   ├── __init__.py
│       │   ├── clock.py        # Clock ABC + WalkForwardClock + LiveClock
│       │   ├── data_source.py  # DataSource ABC
│       │   ├── broker.py       # BrokerGateway ABC + PaperBroker
│       │   ├── summarizer.py   # Same as current (indicators, patterns, levels)
│       │   ├── charts.py       # Same as current (chart generation)
│       │   ├── position.py     # Position state tracking (Redis-backed)
│       │   └── tools.py        # ToolHarness (data source agnostic)
│       │
│       ├── agent/              # DART Agent — shared between backtest and live
│       │   ├── __init__.py
│       │   ├── dart.py         # DartAgent (async, timeout, retry)
│       │   └── prompts.py      # System/user prompts (same as current)
│       │
│       ├── validation/
│       │   ├── __init__.py
│       │   └── validator.py    # Signal structure validation (uses trade_calc)
│       │
│       ├── risk/               # New — live-only (but can be used in backtest)
│       │   ├── __init__.py
│       │   ├── manager.py      # RiskManager (pre-trade checks, post-trade monitoring)
│       │   └── circuit_breaker.py # Circuit breaker state machine
│       │
│       ├── data/               # Live data ingestion
│       │   ├── __init__.py
│       │   ├── zerodha.py      # Zerodha DataSource
│       │   ├── polygon.py      # Polygon.io DataSource
│       │   ├── yahoo.py        # Yahoo Finance DataSource (migrate current collector)
│       │   ├── aggregator.py   # Tick → OHLCV aggregator
│       │   └── buffer.py       # Ring buffer with Redis persistence
│       │
│       ├── state/              # Market state engine
│       │   ├── __init__.py
│       │   ├── engine.py       # Event-driven MarketStateEngine
│       │   └── indicators.py   # Incremental indicator implementations
│       │
│       ├── broker/             # Broker gateway implementations
│       │   ├── __init__.py
│       │   ├── zerodha.py      # Zerodha Kite Connect
│       │   ├── order_manager.py # Order book, routing, retry
│       │   └── position_sizer.py # Sizing strategies
│       │
│       ├── monitor/            # Position monitoring
│       │   ├── __init__.py
│       │   └── position.py     # Stop/target/trail manager
│       │
│       ├── monitoring/         # Alerts and dashboards
│       │   ├── __init__.py
│       │   ├── telegram.py     # Telegram bot
│       │   ├── dashboard.py    # Streamlit real-time dashboard
│       │   └── alerts.py       # Alert routing and deduplication
│       │
│       ├── journal/
│       │   ├── __init__.py
│       │   ├── signal.py       # Same as current (JSONL journal)
│       │   └── evaluator.py    # Same as current (post-session evaluation)
│       │
│       ├── observability/
│       │   ├── __init__.py
│       │   └── langfuse_integration.py  # Same as current
│       │
│       ├── backtest/
│       │   ├── __init__.py
│       │   └── runner.py       # BacktestRunner (uses WalkForwardClock + PaperBroker)
│       │
│       └── live/
│           ├── __init__.py
│           ├── runner.py       # LiveRunner (uses LiveClock + real DataSource + Broker)
│           └── recovery.py     # State persistence & recovery
│
├── tests/
│   ├── __init__.py
│   ├── test_trade_calc.py     # Tests for shared trade math
│   ├── test_summarizer.py     # Tests for indicators, patterns, levels
│   ├── test_validator.py      # Tests for signal validation
│   ├── test_aggregator.py     # Tests for Tick → OHLCV
│   ├── test_indicators.py     # Tests for incremental indicators
│   ├── test_paper_broker.py   # Tests for PaperBroker
│   ├── test_risk_manager.py   # Tests for RiskManager
│   ├── test_agent.py          # Tests for DartAgent (with mock LLM)
│   └── conftest.py            # Shared fixtures
│
├── data/
│   └── raw/                   # Cached data (same as current)
│
├── outputs/
│   ├── charts/                # Chart artifacts
│   ├── journal/               # Signal journal
│   └── logs/                  # Log files
│
├── docs/
│   ├── architecture.md        # Current POC architecture (updated)
│   └── future-architecture.md # This document
│
└── scripts/
    ├── run_backtest.py         # Entry point for backtest
    ├── run_paper.py            # Entry point for paper trading
    └── run_live.py             # Entry point for live trading
```

---

## 14. Key Design Decisions

### 14.1 Agent Architecture

| Decision | Rationale |
|---|---|
| **Async LLM calls** | Don't block candle processing. A slow LLM response should not delay tick handling. |
| **LLM timeout (5-10s)** | If the LLM doesn't respond in time, emit HOLD and continue. Better to skip one candle than to freeze the system. |
| **Pre-computed context** | MarketStatePackage is built incrementally and cached. The agent receives it instantly, not after computation. |
| **Position monitor is separate** | Stop/target checks are deterministic code running on every tick. The LLM is not involved in position management after entry. |
| **Same ToolHarness in backtest and live** | Tools are data-source agnostic. Only the underlying data changes. This ensures backtest → live parity. |

### 14.2 Data Architecture

| Decision | Rationale |
|---|---|
| **Abstract DataSource interface** | Swap providers without changing agent/validator/risk code. Test with Yahoo Finance, go live with Zerodha. |
| **Build candles from ticks** | No polling latency. Exact session boundaries. Lower timeframe candles are always available. |
| **Ring buffer** | Bounded memory (O(N) for N candles, not O(∞)). Fast slice operations. Predictable performance. |
| **Redis persistence** | Survive crashes. Restore candle buffer and position state on restart. |

### 14.3 Risk Architecture

| Decision | Rationale |
|---|---|
| **Hard limits in code** | The LLM cannot override risk limits. These are enforced before any order reaches the broker. |
| **Circuit breaker is automatic + manual** | Auto-trip on drawdown/loss limits. Manual trip via Telegram for human override. |
| **Daily loss cap is final** | Once hit, trading stops for the day regardless of confidence. This is a non-negotiable hard rule. |
| **Post-trade monitoring** | Risk monitoring continues after trade entry. Daily P&L is tracked in real time. |

### 14.4 Operational Architecture

| Decision | Rationale |
|---|---|
| **Event-driven with message bus** | Loose coupling between components. Easy to add/remove subscribers (dashboard, alerts, journal). |
| **Graceful shutdown** | Square off positions, flush journal, save state. No orphaned positions on crash. |
| **Shadow mode first** | Run alongside manual trading for 1-2 weeks before going live with real capital. |
| **Paper broker is a first-class citizen** | Same code path as live. Diff only in config. This catches bugs before they cost real money. |

---

## 15. Out of Scope (For Now)

These are intentionally deferred to keep the first live version focused:

- **Multi-instrument portfolio logic** — Start with one instrument (NIFTY/BANKNIFTY), add more later.
- **News/sentiment/fundamentals** — Price-action only for v1. Add alternative data as a research project.
- **Multi-agent debate systems** — One DART agent is enough for the first live version.
- **Machine learning models** — No neural networks, no reinforcement learning. LLM + deterministic tools only.
- **Reinforcement learning from trading outcomes** — That's a separate research project.
- **Complex order types** — MARKET and LIMIT are enough for v1. Iceberg, TWAP, VWAP can come later.
- **Multi-timeframe simultaneous signals** — One decision per candle close on the configured timeframe.
- **Historical backtest optimization** — The current POC walk-forward is sufficient for v1.
- **Full test suite coverage** — Focus on core math, risk, and broker interaction tests. Not every utility function.
- **Web-based management UI** — Telegram + Streamlit dashboard is sufficient for v1.

---

## Appendix: Glossary

| Term | Definition |
|---|---|
| **DART** | Direction → Area → Risk → Trigger. The decision framework used by the agent. |
| **DataSource** | Abstract interface for market data providers (live WebSocket or historical). |
| **BrokerGateway** | Abstract interface for order execution providers. |
| **PaperBroker** | Simulated broker that fills orders against live market data. |
| **MarketStatePackage** | Snapshot of all market context at a decision point: candles, indicators, levels, patterns. |
| **Circuit Breaker** | Safety mechanism that squares off all positions and halts trading when risk limits are breached. |
| **Position Monitor** | Continuous tracking of open positions — checks stop/target/trail on every tick. |
| **Tick → OHLCV Aggregator** | Real-time builder that creates candlesticks from individual trade ticks. |
| **Ring Buffer** | Fixed-size in-memory circular buffer for candle storage, with Redis persistence. |
| **Incremental Indicator** | O(1) per-update indicator computation (vs O(N) recalculation from scratch). |
