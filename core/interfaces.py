"""
Formal data contracts for the AlphaPulse engine.

Defines the interfaces specified in the Historical Agent Engine plan:
- ContextDeliveryMode: bootstrap vs incremental vs reset context delivery
- HistoricalDataRequest: structured request for get_historical_data
- AgentTurnRecord: message-level audit for each LLM turn
- ToolCallRecord: structured tool call I/O, latency, status
- AuditEvent: engine failures, policy violations, and system events
- TradeEvent: entry, exit, fill, forced square-off, rejection lifecycle
- EngineDecisionResult: validated end-to-end decision payload
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class ContextDeliveryMode(str, Enum):
    BOOTSTRAP = "bootstrap"
    INCREMENTAL = "incremental"
    RESET = "reset"


@dataclass
class HistoricalDataRequest:
    timeframe: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    start_days_ago: Optional[int] = None
    end_days_ago: Optional[int] = None
    max_candles: int = 60


@dataclass
class ToolCallRecord:
    round_num: int
    tool_name: str
    arguments: Dict[str, Any]
    reason: str
    result: Dict[str, Any]
    status: str = "success"
    error: Optional[str] = None
    latency_ms: Optional[float] = None


@dataclass
class AgentTurnRecord:
    turn_number: int
    role: str
    raw_output: str
    parsed_type: Optional[str] = None
    schema_valid: bool = True
    schema_errors: List[str] = field(default_factory=list)
    tool_calls: List[ToolCallRecord] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class AuditEvent:
    event_type: str
    severity: str = "info"
    run_id: Optional[str] = None
    decision_id: Optional[str] = None
    symbol: Optional[str] = None
    message: str = ""
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)


class TradeEventType(str, Enum):
    ENTRY_REQUESTED = "entry_requested"
    ENTRY_FILLED = "entry_filled"
    EXIT_REQUESTED = "exit_requested"
    EXIT_FILLED = "exit_filled"
    FORCED_SQUARE_OFF = "forced_square_off"
    STOP_HIT = "stop_hit"
    TARGET_HIT = "target_hit"
    REJECTED = "rejected"


@dataclass
class TradeEvent:
    event_type: TradeEventType
    run_id: str
    symbol: str
    decision_id: Optional[str] = None
    position_id: Optional[str] = None
    direction: Optional[str] = None
    price: Optional[float] = None
    quantity: Optional[int] = None
    pnl: Optional[float] = None
    reason: str = ""
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class EngineDecisionResult:
    decision_id: str
    run_id: str
    symbol: str
    decision_time: datetime
    raw_action: str
    validated_action: str
    is_valid: bool
    rejection_reason: Optional[str] = None
    confidence: float = 0.0
    entry: Optional[float] = None
    stop: Optional[float] = None
    target: Optional[float] = None
    context_data_hash: Optional[str] = None
    prompt_version: Optional[str] = None
    model_name: Optional[str] = None
    toolset_version: Optional[str] = None
    snapshot_set_id: Optional[str] = None
    validation_outcome: Optional[str] = None
    evaluation_labels: List[str] = field(default_factory=list)
    tool_calls: List[ToolCallRecord] = field(default_factory=list)
    agent_turns: List[AgentTurnRecord] = field(default_factory=list)
    audit_events: List[AuditEvent] = field(default_factory=list)
    raw_llm_responses: List[str] = field(default_factory=list)
