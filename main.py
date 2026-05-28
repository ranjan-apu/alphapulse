"""
Main orchestration script for the Market Agent Harness POC.
Runs the complete walk-forward replay loop with DART agent decisions.
"""
import sys
import json
import time
from datetime import datetime

from config import config
from data.collector import collect_all_data, load_cached_data, cache_is_fresh, resample_to_timeframe
from core.clock import WalkForwardClock
from core.context import build_market_state_package, format_market_state_for_prompt
from core.tools import ToolHarness
from core.charts import generate_all_charts
from core.position import PositionTracker
from agent.dart import DartAgent
from validation.validator import validate_signal
from journal.signal import SignalJournal
from journal.evaluator import FeedbackEvaluator
from observability.langfuse_integration import create_tracer
import logging

# Suppress OpenTelemetry batch export errors when Langfuse server is unavailable
logging.getLogger("opentelemetry").setLevel(logging.CRITICAL)


def setup():
    """Initialize directories and validate config."""
    print("\n" + "=" * 70)
    print("  MARKET AGENT HARNESS POC")
    print("  DART Decision Framework - Walk-Forward Backtest")
    print("=" * 70)

    config.validate()
    config.ensure_dirs()

    print(f"\n  Instrument: {config.INSTRUMENT_NAME} ({config.SYMBOL})")
    print(f"  LLM Provider: {config.LLM_PROVIDER}")
    print(f"  LLM Model:  {config.MODEL_NAME}")
    print(f"  Base URL:   {config.BASE_URL}")
    print(f"  Vision:     {'enabled' if config.VISION_ENABLED else 'disabled'}")
    print(f"  Startup:    {datetime.now().astimezone().isoformat(timespec='seconds')}")
    print(f"  Data cache: {config.FETCH_MONTHS}mo daily/weekly, {config.FETCH_5M_DAYS}d requested 5m")
    print(f"  Decision TF:{config.DECISION_INTERVAL}")
    print(f"  Decision Mode: {config.DECISION_MODE}")
    print(f"  Capital Cap: ₹{config.CAPITAL_CAP:,.0f}")
    print(f"  Order Charges: ₹{config.TOTAL_ORDER_CHARGES} round-trip")
    print(f"  Min R:R:    {config.MIN_REWARD_TO_RISK}:1 (net)")

    # Check Langfuse
    import urllib.request
    try:
        req = urllib.request.Request(f"{config.LANGFUSE_BASE_URL}/api/public/health")
        urllib.request.urlopen(req, timeout=3)
        print(f"  Langfuse:   Connected ({config.LANGFUSE_BASE_URL})")
    except Exception:
        print(f"  Langfuse:   Server not reachable at {config.LANGFUSE_BASE_URL}")
        print(f"             Run: docker-compose up -d langfuse-server langfuse-db")
        print(f"             (Tracing will work in offline/batch mode)")

    # Check Redis
    try:
        import redis
        r = redis.Redis(host=config.REDIS_HOST, port=config.REDIS_PORT, db=config.REDIS_DB)
        r.ping()
        print(f"  Redis:      Connected ({config.REDIS_HOST}:{config.REDIS_PORT})")
        redis_available = True
        redis_client = r
    except Exception as e:
        print(f"  Redis:      Not available ({e})")
        print(f"             Run: docker-compose up -d redis")
        redis_available = False
        redis_client = None

    return redis_available, redis_client


def load_data(force_refresh: bool = False, skip_refresh: bool = False):
    """Load or collect data."""
    print(f"\n{'─' * 60}")
    print("  DATA COLLECTION")
    print(f"{'─' * 60}")

    # Try cached first, but refresh on startup when cache is stale unless asked not to.
    cached = load_cached_data()
    has_cache = all(k in cached for k in ["5m", "daily", "weekly"])
    fresh = cache_is_fresh(cached)

    if has_cache and not force_refresh and (fresh or skip_refresh):
        metadata = cached.get("metadata") or {}
        print("  Using cached data from data/raw/")
        if metadata.get("fetched_at"):
            print(f"  Cache fetched_at: {metadata['fetched_at']}")
        for tf in ["5m", "daily", "weekly"]:
            df = cached[tf]
            print(f"    {tf}: {len(df)} candles from {df.index[0]} to {df.index[-1]}")
        return cached

    if has_cache and not fresh:
        print(f"  Cache older than {config.CACHE_MAX_AGE_HOURS}h; refreshing from Yahoo Finance...")
    elif force_refresh:
        print("  Refresh requested; fetching fresh data from Yahoo Finance...")
    else:
        print("  No complete cache found; fetching fresh data from Yahoo Finance...")
    data = collect_all_data()
    return data


def run_replay(
    data: dict,
    max_steps: int = None,
    redis_avail: bool = False,
    redis_client=None,
    replay_date=None,
):
    """
    Run the complete walk-forward replay loop.

    Args:
        data: Dict with '5m', 'daily', 'weekly' DataFrames
        max_steps: Limit number of steps for quick testing (None = all)
        redis_avail: Whether Redis is available
    """
    df_5m = resample_to_timeframe(data["5m"], config.DECISION_INTERVAL)
    df_daily = data["daily"]
    df_weekly = data["weekly"]

    # Initialize components
    clock = WalkForwardClock(df_5m)
    agent = DartAgent()
    journal = SignalJournal()
    evaluator = FeedbackEvaluator(df_5m)
    tracer = create_tracer()
    position = PositionTracker(redis_client)
    position.reset()

    total_steps = clock.total_steps()
    if replay_date:
        total_steps = sum(
            1 for point in clock.iterate()
            if point["decision_time"].date() == replay_date
        )
    print(f"\n{'─' * 60}")
    print(f"  WALK-FORWARD REPLAY")
    print(f"{'─' * 60}")
    print(f"  Total eligible decision points: {total_steps}")
    if replay_date:
        print(f"  Replay date: {replay_date}")
    print(f"  {config.DECISION_INTERVAL} data range: {df_5m.index[0]} to {df_5m.index[-1]}")
    print(f"  Max steps limit: {max_steps or 'None (all)'}")
    print(f"  Max tool calls per decision: {config.MAX_TOOL_CALLS_PER_DECISION}")
    print(f"  Agent context: {config.HTF_MONTHS}mo weekly + {config.MACRO_MONTHS}mo daily + {config.MICRO_DAYS} trading sessions of {config.DECISION_INTERVAL}")

    # ---- REPLAY LOOP ----
    step_count = 0
    start_time = time.time()

    print(f"\n{'─' * 60}")
    print("  STARTING REPLAY")
    print(f"{'─' * 60}\n")

    for decision_point in clock.iterate():
        T = decision_point["decision_time"]
        session_start = decision_point["session_start"]
        session_end = decision_point["session_end"]

        if replay_date and T.date() != replay_date:
            continue

        if not position.should_evaluate(T):
            continue

        if max_steps and step_count >= max_steps:
            print(f"\n  Reached max steps limit ({max_steps}). Stopping.")
            break
        step_count += 1

        position.record_evaluation(T)
        position.increment_signal_count()
        has_pos = position.has_position()

        # Progress
        print(f"\n[{step_count}/{max_steps or total_steps}] {T} "
              f"| Price: {decision_point['candle_T']['close']:.2f}", end="", flush=True)

        # ---- Create Langfuse root span (trace) ----
        trace_name = f"decision_{T.strftime('%Y%m%d_%H%M')}"
        root_span = tracer.create_root_span(
            name=trace_name,
            input_data={
                "step": step_count,
                "decision_time": str(T),
                "instrument": config.SYMBOL,
                "current_price": float(decision_point['candle_T']['close']),
            },
            metadata={
                "step": step_count,
                "decision_time": str(T),
                "instrument": config.SYMBOL,
            },
        )

        # ---- Build Market State Package ----
        try:
            package = build_market_state_package(
                T=T,
                data_5m=df_5m,
                data_daily=df_daily,
                data_weekly=df_weekly,
                chart_paths={},  # Will be filled after tool calls
            )
            package["session_start"] = str(session_start)
            package["session_end"] = str(session_end)
        except Exception as e:
            print(f"\n  [Error building market state: {e}] - skipping")
            continue

        # ---- Generate initial charts (before tool calls) ----
        # We'll generate basic charts even without agent input
        initial_charts = {}
        try:
            initial_charts = generate_all_charts(
                df_5m, df_daily, df_weekly, T, None, None, None
            )
            package["chart_paths"] = initial_charts
        except Exception as e:
            print(f" [Chart warning: {e}]", end="")

        # ---- Format market state for LLM ----
        market_state_text = format_market_state_for_prompt(package)

        # ---- Initialize Tool Harness ----
        harness = ToolHarness(df_5m, df_daily, df_weekly, T)

        # ---- Run DART Agent ----
        try:
            agent_result = agent.decide(package, market_state_text, harness)
        except Exception as e:
            print(f"\n  [Agent error: {e}]")
            current_position = position.get_position() or {}
            agent_result = {
                "raw_responses": [str(e)],
                "tool_calls": [],
                "final_signal": {
                    "type": "final_signal",
                    "action": "HOLD" if has_pos else "SKIP",
                    "confidence": 0.0,
                    "dart": {"direction": "", "area": "", "risk": "", "trigger": ""},
                    "checklist": {
                        "market_regime": "unclear",
                        "session_type": "unclear",
                        "structure_state": "unclear",
                        "location_quality": 0,
                        "trigger_quality": 0,
                        "risk_quality": 0,
                        "volume_confirmation": 0,
                        "reason_to_wait": "LLM failed; flat-state fallback is SKIP",
                    },
                    "entry": None, "stop": None, "target": None,
                    "position_id": current_position.get("position_id") if has_pos else None,
                    "thesis_health": "valid" if has_pos else "not_applicable",
                    "reason": f"Agent error: {e}",
                    "invalidation": None,
                },
            }

        signal = agent_result.get("final_signal", {})
        if has_pos and signal.get("action") in ("HOLD", "EXIT") and not signal.get("position_id"):
            current_position = position.get_position() or {}
            signal["position_id"] = current_position.get("position_id") or "open_position"

        # ---- Update charts with signal levels ----
        entry = signal.get("entry")
        stop = signal.get("stop")
        target = signal.get("target")

        # Regenerate charts with signal levels
        final_charts = initial_charts
        raw_action = signal.get("action", "HOLD" if has_pos else "SKIP")
        if raw_action in ("BUY", "SELL") and entry and stop and target:
            try:
                final_charts = generate_all_charts(
                    df_5m, df_daily, df_weekly, T, entry, stop, target
                )
            except Exception:
                pass  # Keep initial charts

        # ---- Validate Signal ----
        validation_result = validate_signal(signal, T, session_end, has_open_position=has_pos)

        # ---- Auto-open position if valid BUY/SELL ----
        is_valid = validation_result.get("is_valid", False)
        final_action = validation_result.get("action", signal.get("action", "HOLD"))
        sizing = validation_result.get("sizing", {})

        if is_valid and final_action in ("BUY", "SELL") and not has_pos:
            position.open_position(
                entry_price=float(sizing["entry"]),
                direction=final_action,
                stop=float(sizing["stop"]),
                target=float(sizing["target"]),
                quantity=int(sizing["quantity"]),
                entry_time=str(T),
            )
            print(f" [OPEN {final_action} qty={sizing['quantity']}]", end="", flush=True)

        if validation_result.get("rejection_reason"):
            print(f" [{validation_result['rejection_reason']}]", end="", flush=True)
        else:
            print(f" [{final_action}]", end="", flush=True)

        # ---- Journal the decision ----
        record = journal.record(
            decision_time=T,
            market_state_package=package,
            agent_result=agent_result,
            validation_result=validation_result,
            chart_paths=final_charts,
        )

        # ---- Log to Langfuse (v3 API) ----
        if root_span:
            # Log each LLM call as a generation
            for i, raw_resp in enumerate(agent_result.get("raw_responses", [])):
                # Estimate token usage from response length (DeepSeek doesn't always return usage)
                approx_prompt_tokens = len(market_state_text) // 4
                approx_resp_tokens = len(str(raw_resp)) // 4
                tracer.add_generation(
                    parent_span=root_span,
                    name=f"llm_call_{i+1}",
                    model=config.MODEL_NAME,
                    input_data=market_state_text[:2000],  # Truncate for tracing
                    output_data=str(raw_resp)[:2000],
                    usage={
                        "input": approx_prompt_tokens,
                        "output": approx_resp_tokens,
                        "total": approx_prompt_tokens + approx_resp_tokens,
                    },
                    metadata={
                        "call_number": i + 1,
                        "total_calls": len(agent_result.get("raw_responses", [])),
                    },
                )

            # Log tool calls as child spans
            for tc in agent_result.get("tool_calls", []):
                tracer.add_span(
                    parent_span=root_span,
                    name=f"tool_{tc['tool']}",
                    input_data={"tool": tc["tool"], "arguments": tc.get("arguments")},
                    output_data=tc.get("result"),
                    metadata={"reason": tc.get("reason", "")},
                )

            # Log validation as a child span
            tracer.add_span(
                parent_span=root_span,
                name="validation",
                input_data={
                    "action": signal.get("action"),
                    "entry": signal.get("entry"),
                    "stop": signal.get("stop"),
                    "target": signal.get("target"),
                },
                output_data={
                    "is_valid": validation_result.get("is_valid"),
                    "action": validation_result.get("action"),
                    "rejection_reason": validation_result.get("rejection_reason"),
                },
            )

            # End root span with final output
            tracer.end_root_span(
                root_span,
                output_data={
                    "action": validation_result.get("action", signal.get("action")),
                    "confidence": signal.get("confidence"),
                    "is_valid": validation_result.get("is_valid"),
                    "tool_calls_made": len(agent_result.get("tool_calls", [])),
                },
            )

        # ---- Redis caching ----
        if redis_avail and redis_client:
            try:
                cache_key = f"decision:{config.SYMBOL}:{T.strftime('%Y%m%d_%H%M')}"
                redis_client.setex(cache_key, 3600, json.dumps({
                    "action": final_action,
                    "price": package.get("current_price"),
                    "confidence": signal.get("confidence"),
                    "has_position": has_pos,
                }, default=str))
            except Exception:
                pass

    # ---- REPLAY COMPLETE ----
    elapsed = time.time() - start_time
    print(f"\n\n{'─' * 60}")
    print(f"  REPLAY COMPLETE")
    print(f"{'─' * 60}")
    print(f"  Steps executed: {step_count}")
    print(f"  Elapsed: {elapsed:.1f}s ({elapsed/step_count:.1f}s per step)" if step_count > 0 else "")

    # ---- Close any remaining position ----
    pos_summary = position.summary()
    print(f"  Position state: {'OPEN' if pos_summary['has_position'] else 'CLOSED'}")
    if pos_summary['has_position']:
        square_off_df = df_5m
        if replay_date:
            square_off_df = df_5m[df_5m.index.date == replay_date]
        last_price = float(square_off_df["close"].iloc[-1])
        position.close_position(last_price, "replay_end_square_off")
        print(f"  Forced square-off at replay end @ {last_price:.2f}")
    print(f"  Signals evaluated (LLM calls): {pos_summary['signal_count']}")

    # ---- Flush Langfuse ----
    tracer.flush()

    # ---- Journal Summary ----
    summary = journal.summary()
    print(f"\n{'─' * 60}")
    print(f"  JOURNAL SUMMARY")
    print(f"{'─' * 60}")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    # Save journal summary
    summary_path = config.JOURNAL_DIR / "journal_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Journal: {journal.output_path}")
    print(f"  Summary: {summary_path}")

    # ---- Evaluate Outcomes ----
    print(f"\n{'─' * 60}")
    print(f"  EVALUATING OUTCOMES")
    print(f"{'─' * 60}")

    eval_results = evaluator.evaluate_all(journal.get_records())
    metrics = evaluator.compute_metrics()

    print(f"\n  Evaluation Metrics:")
    for k, v in metrics.items():
        print(f"    {k}: {v}")

    eval_path = evaluator.save_results()
    print(f"\n  Evaluation results: {eval_path}")

    # ---- Final Report ----
    print(f"\n{'=' * 60}")
    print(f"  POC COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Total decisions:   {summary['total_decisions']}")
    print(f"  Actionable trades: {summary['actionable_trades']}")
    print(f"  Rejected signals:  {summary['rejected_signals']}")
    print(f"  HOLD rate:         {summary['hold_rate']:.2%}")
    if "win_rate" in metrics:
        print(f"  Win rate:          {metrics['win_rate']:.2%}")
    if "avg_net_r_multiple" in metrics:
        print(f"  Avg Net R:         {metrics['avg_net_r_multiple']:.2f}R")
    if "total_net_pnl" in metrics:
        print(f"  Total Net P&L:     ₹{metrics['total_net_pnl']:,.2f}")
    if "total_net_profit" in metrics:
        print(f"  Net Profit:        ₹{metrics['total_net_profit']:,.2f}")
        print(f"  Net Loss:          ₹{metrics['total_net_loss']:,.2f}")
        print(f"  Profitable/Losing: {metrics['profitable_trades']} / {metrics['losing_trades']}")

    print(f"\n  Output files:")
    print(f"    Journal:    {journal.output_path}")
    print(f"    Evaluation: {config.JOURNAL_DIR / 'evaluation_results.json'}")
    print(f"    Charts:     {config.CHARTS_DIR}")

    return {
        "journal": journal,
        "evaluator": evaluator,
        "summary": summary,
        "metrics": metrics,
    }


def main():
    """Entry point."""
    # Check for --max-steps argument
    max_steps = None
    force_refresh = False
    skip_refresh = False
    replay_date = None
    for arg in sys.argv[1:]:
        if arg.startswith("--max-steps="):
            max_steps = int(arg.split("=")[1])
        elif arg.startswith("--date="):
            replay_date = datetime.strptime(arg.split("=", 1)[1], "%Y-%m-%d").date()
        elif arg == "--quick":
            max_steps = 5
        elif arg == "--refresh-data":
            force_refresh = True
        elif arg == "--skip-data":
            skip_refresh = True

    print("\n" + "=" * 70)
    print("  DART MARKET AGENT HARNESS - POC")
    print("  Walk-Forward Backtest with LLM Decision Agent")
    print("=" * 70)

    # Setup
    redis_avail, redis_client = setup()

    # Load data
    data = load_data(force_refresh=force_refresh, skip_refresh=skip_refresh)

    # Validate data
    decision_data = resample_to_timeframe(data["5m"], config.DECISION_INTERVAL)
    if len(decision_data) < 20:
        print(f"\n  ERROR: Insufficient {config.DECISION_INTERVAL} data ({len(decision_data)} candles). Need at least 20.")
        sys.exit(1)

    # Run replay
    results = run_replay(
        data,
        max_steps=max_steps,
        redis_avail=redis_avail,
        redis_client=redis_client,
        replay_date=replay_date,
    )

    print(f"\n{'=' * 70}")
    print("  DONE. Hypothesis ready for inspection.")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
