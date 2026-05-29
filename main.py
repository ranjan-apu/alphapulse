"""
Main orchestration script for the Market Agent Harness POC.
Runs the complete walk-forward replay loop with DART agent decisions.
"""
import sys
from datetime import datetime

from config import config
from data.collector import collect_all_data, load_cached_data, cache_is_fresh, resample_to_timeframe
from core.replay_runner import ReplayRunner
import logging

# Suppress OpenTelemetry batch export errors when Langfuse server is unavailable
logging.getLogger("opentelemetry").setLevel(logging.CRITICAL)


def setup():
    """Initialize directories, validate config, and check Postgres connectivity."""
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

    # Check Postgres (fail fast - required for historical runs)
    from db.connection import ensure_connection_or_exit
    ensure_connection_or_exit()
    print(f"  Postgres:   Connected")

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
    redis_available = False
    redis_client = None
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

    return redis_available, redis_client


def load_data(force_refresh: bool = False, skip_refresh: bool = False):
    """Load or collect data."""
    print(f"\n{'─' * 60}")
    print("  DATA COLLECTION")
    print(f"{'─' * 60}")

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


def main():
    """Entry point."""
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

    # Run replay via ReplayRunner
    runner = ReplayRunner(
        data_5m=data["5m"],
        data_daily=data["daily"],
        data_weekly=data["weekly"],
        redis_avail=redis_avail,
        redis_client=redis_client,
        replay_date=replay_date,
    )
    results = runner.run(max_steps=max_steps)

    print(f"\n{'=' * 70}")
    print("  DONE. Hypothesis ready for inspection.")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
