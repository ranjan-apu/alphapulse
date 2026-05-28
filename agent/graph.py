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
    """Load portfolio and position state."""
    # Try to use PortfolioStateManager if available in state
    psm = state.get("_portfolio_manager")
    if psm:
        portfolio = psm.get_portfolio_state()
        state["portfolio_state"] = {
            "cash_available": portfolio.cash_available,
            "capital_deployed": portfolio.capital_deployed,
            "realized_pnl": portfolio.realized_pnl,
            "unrealized_pnl": portfolio.unrealized_pnl,
            "charges_paid": portfolio.charges_paid,
            "trades_taken_today": portfolio.trades_taken_today,
            "max_trades_per_day": portfolio.max_trades_per_day,
            "daily_loss_used": portfolio.daily_loss_used,
            "max_daily_loss": portfolio.max_daily_loss,
            "can_trade": portfolio.can_trade()[0],
        }
        pos = psm.get_open_position()
        if pos:
            state["open_position"] = {
                "position_id": pos.position_id,
                "symbol": pos.symbol,
                "direction": pos.direction,
                "entry": pos.entry,
                "executed_entry": pos.executed_entry,
                "stop": pos.stop,
                "target": pos.target,
                "quantity": pos.quantity,
                "unrealized_pnl": pos.unrealized_pnl,
                "r_multiple_live": pos.r_multiple_live,
            }
            state["has_position"] = True
        else:
            state["has_position"] = False
        return state
    
    # Fallback to pre-loaded state
    portfolio = state.get("portfolio_state", {})
    if not portfolio:
        state["portfolio_state"] = {
            "cash_available": 0.0,
            "capital_deployed": 0.0,
            "has_position": False,
            "error": "Portfolio state not available",
        }
    state["has_position"] = state.get("open_position") is not None
    return state


def _node_retrieve_memory(state: AgentState) -> AgentState:
    """Retrieve relevant memories using MemoryStore/AgentPlanner."""
    planner = state.get("_planner")
    memory_store = state.get("_memory_store")
    analysis_plan = state.get("analysis_plan", {})
    symbol = state.get("symbol", "")
    
    if planner and memory_store and analysis_plan:
        try:
            from agent.schema import AnalysisPlan
            plan = AnalysisPlan(**analysis_plan) if isinstance(analysis_plan, dict) else analysis_plan
            memories = planner.retrieve_context_memories(symbol, plan)
            state["retrieved_memories"] = memories
            state["memory_context"] = planner.format_memory_context(memories)
            return state
        except Exception as e:
            state["memory_context"] = f"Memory retrieval failed: {e}"
            return state
    
    # Fallback
    state["memory_context"] = state.get("memory_context", "")
    return state


def _node_plan_analysis(state: AgentState) -> AgentState:
    """Create analysis plan using AgentPlanner."""
    planner = state.get("_planner")
    if planner:
        try:
            plan = planner.plan_analysis(
                market_state=state.get("market_state_package", {}),
                portfolio_state=state.get("portfolio_state", {}),
                session_phase=state.get("session_phase", {}),
                has_position=state.get("has_position", False),
            )
            state["analysis_plan"] = plan.model_dump()
            return state
        except Exception as e:
            pass
    
    # Fallback
    plan = state.get("analysis_plan", {})
    if not plan:
        state["analysis_plan"] = {
            "direction_bias": "neutral",
            "setup_tags": [],
            "areas_of_interest": [],
            "planned_tools": ["get_portfolio_state"],
            "reason": "Default plan",
        }
    return state


def _node_execute_tools(state: AgentState) -> AgentState:
    """
    Execute tool calls through ToolHarness.
    
    The LLM's tool requests arrive via state['tool_requests'].
    Each request is dispatched to ToolHarness and results stored.
    """
    harness = state.get("_tool_harness")
    pending_requests = state.get("tool_requests", [])
    tool_results = state.get("tool_results", [])
    tool_count = state.get("tool_call_count", 0)
    max_calls = state.get("max_tool_calls", 6)
    
    if harness and pending_requests:
        for req in pending_requests:
            if tool_count >= max_calls:
                break
            tool_name = req.get("tool", "")
            tool_args = req.get("arguments", {})
            if tool_name:
                result = harness.execute(tool_name, tool_args)
                tool_results.append({
                    "tool": tool_name,
                    "arguments": tool_args,
                    "result": result,
                })
                tool_count += 1
        state["tool_results"] = tool_results
        state["tool_call_count"] = tool_count
        state["tool_requests"] = []  # Clear processed requests
    
    if tool_count >= max_calls:
        state["should_continue_tools"] = False
    else:
        state["should_continue_tools"] = len(state.get("tool_requests", [])) > 0

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
    """Persist the decision via journal or Postgres."""
    import uuid
    state["decision_id"] = f"dec_{uuid.uuid4().hex[:12]}"
    
    journal = state.get("_journal")
    if journal:
        try:
            journal.record(
                decision_time=state.get("decision_time", ""),
                market_state_package=state.get("market_state_package", {}),
                agent_result={
                    "raw_responses": [],
                    "tool_calls": state.get("tool_results", []),
                    "final_signal": state.get("validated_signal", {}),
                },
                validation_result={
                    "is_valid": state.get("risk_check_passed", False),
                    "rejection_reason": "; ".join(state.get("risk_check_errors", [])),
                    "action": state.get("validated_signal", {}).get("action", "SKIP"),
                },
                chart_paths={},
            )
        except Exception:
            pass
    
    return state


def _node_update_memory(state: AgentState) -> AgentState:
    """Update working/session memory with the current decision."""
    memory_store = state.get("_memory_store")
    if memory_store and state.get("validated_signal"):
        # Update working memory with tool outputs
        if memory_store.working:
            for tr in state.get("tool_results", []):
                memory_store.working.add_tool_output(tr["tool"], tr["result"])
    
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
