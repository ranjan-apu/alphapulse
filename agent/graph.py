"""
LangGraph-based agent workflow orchestrator (Section 6.8).

Implements the agent as a graph rather than a hand-written loop.
LangGraph orchestrates the workflow, while AlphaPulse deterministic
modules own trading logic (math, risk, portfolio state, validation).

Graph nodes:
  load_context -> load_portfolio_state -> retrieve_memory -> plan_analysis
  -> execute_tools -> synthesize_signal -> validate_schema
  -> validate_trade_math -> risk_check -> persist_decision -> update_memory

Why LangGraph:
- checkpoint every node
- replay failed decisions
- inspect state history
- support durable memory
- separate planning/tool/synthesis steps cleanly
- enable A/B experiments by swapping graph nodes
"""
import json
from typing import Dict, Any, List, Optional, TypedDict, Annotated
from datetime import datetime
import operator

# LangGraph may not be installed in all environments; graceful degradation
try:
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.memory import MemorySaver
    HAS_LANGGRAPH = True
except ImportError:
    HAS_LANGGRAPH = False


# ---- Agent State ----

class AgentState(TypedDict, total=False):
    """State object passed between graph nodes."""
    # Input
    decision_time: str
    symbol: str
    run_id: str
    market_state_package: dict
    market_state_text: str

    # Portfolio
    portfolio_state: dict
    has_position: bool
    open_position: Optional[dict]

    # Session
    session_phase: dict
    cooldown_state: dict
    gap_context: dict

    # Memory
    retrieved_memories: dict
    memory_context: str
    analysis_plan: dict

    # Tool execution
    tool_requests: List[dict]
    tool_results: List[dict]
    tool_call_count: int
    max_tool_calls: int

    # Signal
    raw_signal: dict
    validated_signal: dict
    final_signal: dict

    # Validation
    schema_valid: bool
    schema_errors: List[str]
    trade_math: dict
    risk_check_passed: bool
    risk_check_errors: List[str]

    # Persistence
    decision_id: str
    position_id: Optional[str]

    # Flow control
    should_continue_tools: bool
    error: Optional[str]


# ---- Graph Nodes ----

def create_agent_graph(
    checkpointer=None,
    max_tool_calls: int = 6,
):
    """
    Create the LangGraph agent workflow.

    Args:
        checkpointer: LangGraph checkpointer (MemorySaver or PostgresSaver)
        max_tool_calls: Maximum tool calls per decision

    Returns:
        Compiled StateGraph ready for invocation
    """
    if not HAS_LANGGRAPH:
        raise ImportError(
            "langgraph is required for graph-based orchestration. "
            "Install with: pip install langgraph"
        )

    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("load_context", _node_load_context)
    workflow.add_node("load_portfolio_state", _node_load_portfolio_state)
    workflow.add_node("retrieve_memory", _node_retrieve_memory)
    workflow.add_node("plan_analysis", _node_plan_analysis)
    workflow.add_node("execute_tools", _node_execute_tools)
    workflow.add_node("synthesize_signal", _node_synthesize_signal)
    workflow.add_node("validate_schema", _node_validate_schema)
    workflow.add_node("validate_trade_math", _node_validate_trade_math)
    workflow.add_node("risk_check", _node_risk_check)
    workflow.add_node("persist_decision", _node_persist_decision)
    workflow.add_node("update_memory", _node_update_memory)
    workflow.add_node("handle_error", _node_handle_error)

    # Set entry point
    workflow.set_entry_point("load_context")

    # Define edges
    workflow.add_edge("load_context", "load_portfolio_state")
    workflow.add_edge("load_portfolio_state", "retrieve_memory")
    workflow.add_edge("retrieve_memory", "plan_analysis")
    workflow.add_edge("plan_analysis", "execute_tools")

    # Conditional edge from execute_tools
    workflow.add_conditional_edges(
        "execute_tools",
        _should_continue_tools,
        {
            "continue": "execute_tools",  # Loop back for more tools
            "synthesize": "synthesize_signal",
            "error": "handle_error",
        },
    )

    workflow.add_edge("synthesize_signal", "validate_schema")
    workflow.add_conditional_edges(
        "validate_schema",
        _should_proceed_after_validation,
        {
            "proceed": "validate_trade_math",
            "retry": "synthesize_signal",
            "error": "handle_error",
        },
    )
    workflow.add_edge("validate_trade_math", "risk_check")
    workflow.add_conditional_edges(
        "risk_check",
        _should_proceed_after_risk,
        {
            "proceed": "persist_decision",
            "reject": "persist_decision",  # Still persist rejected decisions
            "error": "handle_error",
        },
    )
    workflow.add_edge("persist_decision", "update_memory")
    workflow.add_edge("update_memory", END)
    workflow.add_edge("handle_error", END)

    # Compile with checkpointing
    if checkpointer is None:
        checkpointer = MemorySaver()

    return workflow.compile(checkpointer=checkpointer)


# ---- Node Implementations ----

def _node_load_context(state: AgentState) -> AgentState:
    """Load and validate market context at decision time."""
    # Context is already loaded by the harness and passed via state
    state["error"] = None

    market_state = state.get("market_state_package", {})
    if not market_state:
        state["error"] = "No market state package provided"
        return state

    # Validate context rows
    context_counts = market_state.get("context_row_counts", {})
    if context_counts.get("intraday", 0) < 5:
        state["error"] = "Insufficient intraday context"

    return state


def _node_load_portfolio_state(state: AgentState) -> AgentState:
    """Load portfolio and position state from Postgres/manager."""
    # These should be populated by the harness before invoking the graph
    portfolio = state.get("portfolio_state", {})
    if not portfolio:
        state["portfolio_state"] = {
            "cash_available": 100000.0,
            "capital_deployed": 0.0,
            "has_position": False,
            "message": "Portfolio state not loaded; using defaults",
        }

    state["has_position"] = state.get("open_position") is not None
    return state


def _node_retrieve_memory(state: AgentState) -> AgentState:
    """Retrieve relevant memories for the current decision."""
    # Memory retrieval is handled by AgentPlanner before graph invocation
    # Results are passed via state
    memories = state.get("retrieved_memories", {})
    memory_text = state.get("memory_context", "")

    if not memories and not memory_text:
        state["memory_context"] = ""

    return state


def _node_plan_analysis(state: AgentState) -> AgentState:
    """Create analysis plan from market state."""
    # Analysis plan is created by AgentPlanner before graph invocation
    plan = state.get("analysis_plan", {})
    if not plan:
        state["analysis_plan"] = {
            "direction_bias": "neutral",
            "setup_tags": [],
            "areas_of_interest": [],
            "planned_tools": [],
            "reason": "No analysis plan generated",
        }
    return state


def _node_execute_tools(state: AgentState) -> AgentState:
    """
    Execute tool calls requested by the LLM.

    Tool execution is handled by ToolHarness. The graph node coordinates
    the request/response cycle.
    """
    tool_requests = state.get("tool_requests", [])
    tool_results = state.get("tool_results", [])
    tool_count = state.get("tool_call_count", 0)
    max_calls = state.get("max_tool_calls", 6)

    # If we have pending tool requests, they would be executed here
    # In practice, the harness handles tool execution outside the graph
    # and feeds results back in

    if tool_count >= max_calls:
        state["should_continue_tools"] = False
    else:
        state["should_continue_tools"] = len(tool_requests) > 0

    return state


def _node_synthesize_signal(state: AgentState) -> AgentState:
    """Synthesize tool results into a final trading signal."""
    # The LLM generates the final signal after tool execution
    # This node records the signal for validation
    raw_signal = state.get("raw_signal", {})
    if not raw_signal:
        state["raw_signal"] = {
            "action": "SKIP",
            "reason": "No signal produced by agent",
            "confidence": 0.0,
        }
    return state


def _node_validate_schema(state: AgentState) -> AgentState:
    """Validate the signal against Pydantic schema."""
    raw_signal = state.get("raw_signal", {})

    try:
        from agent.schema import validate_llm_output
        is_valid, parsed, errors = validate_llm_output(raw_signal)

        state["schema_valid"] = is_valid
        state["schema_errors"] = errors

        if is_valid and parsed:
            from agent.schema import signal_to_dict
            state["validated_signal"] = signal_to_dict(parsed)
        else:
            state["validated_signal"] = raw_signal

    except Exception as e:
        state["schema_valid"] = False
        state["schema_errors"] = [str(e)]
        state["validated_signal"] = raw_signal

    return state


def _node_validate_trade_math(state: AgentState) -> AgentState:
    """Validate trade math (R:R, sizing, charges)."""
    signal = state.get("validated_signal", {})
    action = signal.get("action", "SKIP")

    if action not in ("BUY", "SELL"):
        state["trade_math"] = {"actionable": False, "note": "Not an entry signal"}
        return state

    entry = signal.get("entry")
    stop = signal.get("stop")
    target = signal.get("target")

    if not all([entry, stop, target]):
        state["trade_math"] = {
            "actionable": False,
            "errors": ["Missing entry/stop/target"]
        }
        return state

    try:
        from core.position_sizing import compute_position_size
        result = compute_position_size(
            float(entry), float(stop), float(target), action,
            total_charges=60.0,
        )
        state["trade_math"] = {
            "actionable": result.actionable,
            "quantity": result.quantity,
            "gross_risk": result.gross_risk,
            "net_reward_risk": result.net_reward_risk,
            "errors": result.errors,
            "warnings": result.warnings,
        }
    except Exception as e:
        state["trade_math"] = {
            "actionable": False,
            "errors": [str(e)]
        }

    return state


def _node_risk_check(state: AgentState) -> AgentState:
    """Perform risk validation (session phase, capital, cooldown, daily limits)."""
    errors = []
    signal = state.get("validated_signal", {})
    action = signal.get("action", "SKIP")

    if action in ("SKIP", "HOLD", "EXIT"):
        state["risk_check_passed"] = True
        state["risk_check_errors"] = []
        return state

    # Check session phase
    session_phase = state.get("session_phase", {})
    if not session_phase.get("can_open_new", True):
        errors.append("Session phase does not allow new entries")

    # Check cooldown
    cooldown = state.get("cooldown_state", {})
    if cooldown.get("active_locks"):
        errors.append(f"Cooldown active: {len(cooldown['active_locks'])} lock(s)")

    # Check portfolio
    portfolio = state.get("portfolio_state", {})
    if not portfolio.get("can_trade", True):
        errors.append("Portfolio state does not allow trading")

    # Check trade math
    trade_math = state.get("trade_math", {})
    if not trade_math.get("actionable", False):
        errors.extend(trade_math.get("errors", []))

    state["risk_check_passed"] = len(errors) == 0
    state["risk_check_errors"] = errors
    return state


def _node_persist_decision(state: AgentState) -> AgentState:
    """Persist the decision to Postgres/journal."""
    import uuid

    state["decision_id"] = f"dec_{uuid.uuid4().hex[:12]}"

    # In production, this writes to decisions table and portfolio_snapshots
    # For now, the journal handles persistence

    return state


def _node_update_memory(state: AgentState) -> AgentState:
    """Update working/session memory with the current decision."""
    # Memory updates are handled by MemoryStore after the decision is made
    return state


def _node_handle_error(state: AgentState) -> AgentState:
    """Handle errors gracefully."""
    error = state.get("error", "Unknown error")

    # Fallback action based on position state
    if state.get("has_position"):
        state["final_signal"] = {
            "action": "HOLD",
            "confidence": 0.0,
            "reason": f"Error: {error}. Holding position.",
        }
    else:
        state["final_signal"] = {
            "action": "SKIP",
            "confidence": 0.0,
            "reason": f"Error: {error}. Skipping.",
        }

    return state


# ---- Conditional Edge Functions ----

def _should_continue_tools(state: AgentState) -> str:
    """Determine whether to continue tool execution or synthesize."""
    if state.get("error"):
        return "error"

    if state.get("should_continue_tools", False):
        tool_count = state.get("tool_call_count", 0)
        max_calls = state.get("max_tool_calls", 6)
        if tool_count < max_calls:
            return "continue"

    return "synthesize"


def _should_proceed_after_validation(state: AgentState) -> str:
    """After schema validation, decide next step."""
    if state.get("error"):
        return "error"

    if not state.get("schema_valid", False):
        errors = state.get("schema_errors", [])
        # One retry for schema failures
        if not state.get("_schema_retry_attempted"):
            state["_schema_retry_attempted"] = True
            return "retry"

    return "proceed"


def _should_proceed_after_risk(state: AgentState) -> str:
    """After risk check, decide next step."""
    if state.get("error"):
        return "error"

    signal = state.get("validated_signal", {})
    action = signal.get("action", "")

    if action in ("SKIP", "HOLD", "EXIT"):
        return "proceed"

    if not state.get("risk_check_passed", False):
        return "reject"

    return "proceed"


# ---- Convenience Function ----

def run_agent_graph(
    market_state_package: dict,
    market_state_text: str,
    portfolio_state: dict,
    open_position: Optional[dict],
    session_phase: dict,
    cooldown_state: dict,
    gap_context: dict,
    retrieved_memories: dict,
    memory_context: str,
    analysis_plan: dict,
    symbol: str = "",
    run_id: str = "",
    max_tool_calls: int = 6,
) -> dict:
    """
    Run the full agent graph synchronously.

    Args:
        All context objects needed for the decision

    Returns:
        Final AgentState dict with decision result
    """
    if not HAS_LANGGRAPH:
        raise ImportError("langgraph is required for graph orchestration")

    graph = create_agent_graph(max_tool_calls=max_tool_calls)

    initial_state: AgentState = {
        "decision_time": str(market_state_package.get("decision_time", "")),
        "symbol": symbol,
        "run_id": run_id,
        "market_state_package": market_state_package,
        "market_state_text": market_state_text,
        "portfolio_state": portfolio_state,
        "has_position": open_position is not None,
        "open_position": open_position,
        "session_phase": session_phase,
        "cooldown_state": cooldown_state,
        "gap_context": gap_context,
        "retrieved_memories": retrieved_memories,
        "memory_context": memory_context,
        "analysis_plan": analysis_plan,
        "tool_requests": [],
        "tool_results": [],
        "tool_call_count": 0,
        "max_tool_calls": max_tool_calls,
        "raw_signal": {},
        "validated_signal": {},
        "final_signal": {},
        "schema_valid": False,
        "schema_errors": [],
        "trade_math": {},
        "risk_check_passed": False,
        "risk_check_errors": [],
        "should_continue_tools": False,
        "error": None,
    }

    config = {"configurable": {"thread_id": f"decision_{datetime.now().timestamp()}"}}

    result = graph.invoke(initial_state, config)
    return result
