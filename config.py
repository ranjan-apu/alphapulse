"""
Configuration loader for the Market Agent Harness POC.
Reads from .env file and provides typed config.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
PROJECT_ROOT = Path(__file__).parent
load_dotenv(PROJECT_ROOT / ".env")


class Config:
    """Central configuration. All secrets come from .env."""

    # OpenAI-compatible LLM API.
    # Prefer the newer generic/OpenRouter variables when present, while keeping
    # the original DeepSeek names as a text-only fallback.
    DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY") or os.getenv("OR_APIKEY", "")
    OPENROUTER_BASE_URL: str = (
        os.getenv("OPENROUTER_BASE_URL")
        or os.getenv("OPENROUTER_BASEURL")
        or "https://openrouter.ai/api/v1"
    )
    OPENROUTER_MODEL_NAME: str = (
        os.getenv("OPENROUTER_MODEL_NAME")
        or os.getenv("GROK_MODEL_NAME")
        or ""
    )

    USE_OPENROUTER: bool = bool(OPENROUTER_API_KEY and OPENROUTER_MODEL_NAME)
    API_KEY: str = os.getenv("LLM_API_KEY") or (
        OPENROUTER_API_KEY if USE_OPENROUTER else DEEPSEEK_API_KEY
    )
    BASE_URL: str = os.getenv("LLM_BASE_URL") or (
        OPENROUTER_BASE_URL if USE_OPENROUTER else os.getenv("BASE_URL", "https://api.deepseek.com")
    )
    MODEL_NAME: str = os.getenv("LLM_MODEL_NAME") or (
        OPENROUTER_MODEL_NAME if USE_OPENROUTER else os.getenv("MODEL_NAME", "deepseek-v4-pro")
    )
    LLM_PROVIDER: str = "openrouter" if USE_OPENROUTER else "deepseek"

    VISION_ENABLED: bool = os.getenv(
        "LLM_VISION_ENABLED",
        "true" if USE_OPENROUTER else "false",
    ).strip().lower() in ("1", "true", "yes", "on")
    VISION_IMAGE_DETAIL: str = os.getenv("VISION_IMAGE_DETAIL", "auto")
    VISION_CHART_KEYS: list = [
        key.strip()
        for key in os.getenv(
            "VISION_CHART_KEYS",
            "context_dashboard,decision_zoom_chart",
        ).split(",")
        if key.strip()
    ]

    # Langfuse observability
    LANGFUSE_SECRET_KEY: str = os.getenv("LANGFUSE_SECRET_KEY", "")
    LANGFUSE_PUBLIC_KEY: str = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    LANGFUSE_BASE_URL: str = os.getenv("LANGFUSE_BASE_URL", "http://localhost:9876")

    # Data
    DATA_DIR: Path = PROJECT_ROOT / "data"
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    OUTPUT_DIR: Path = PROJECT_ROOT / "outputs"
    CHARTS_DIR: Path = OUTPUT_DIR / "charts"
    JOURNAL_DIR: Path = OUTPUT_DIR / "journal"

    # Instrument
    SYMBOL: str = os.getenv("SYMBOL", "^NSEI")
    INSTRUMENT_NAME: str = os.getenv("INSTRUMENT_NAME", "NIFTY 50")

    # Session times (Indian equity: 09:15 - 15:30 IST)
    SESSION_START_HOUR: int = 9
    SESSION_START_MINUTE: int = 15
    SESSION_END_HOUR: int = 15
    SESSION_END_MINUTE: int = 30
    # Start making decisions after this time (avoid first 15 min volatility)
    DECISION_START_HOUR: int = 9
    DECISION_START_MINUTE: int = 30

    # Data fetch windows (how much raw data to pull/cache at startup).
    # Yahoo Finance usually limits 5m candles to the recent intraday window,
    # but daily/weekly fetches can cover the full six-month harness cache.
    FETCH_MONTHS: int = int(os.getenv("FETCH_MONTHS", "6"))
    FETCH_5M_PERIOD: str = os.getenv("FETCH_5M_PERIOD", f"{FETCH_MONTHS}mo")
    FETCH_5M_DAYS: int = int(os.getenv("FETCH_5M_DAYS", "60"))
    FETCH_DAILY_MONTHS: int = int(os.getenv("FETCH_DAILY_MONTHS", str(FETCH_MONTHS)))
    FETCH_WEEKLY_MONTHS: int = int(os.getenv("FETCH_WEEKLY_MONTHS", str(FETCH_MONTHS)))
    CACHE_MAX_AGE_HOURS: int = int(os.getenv("CACHE_MAX_AGE_HOURS", "12"))

    # Data context windows fed to agent (subset of fetched data, filtered <= T)
    MICRO_DAYS: int = 3   # 3 trading sessions of intraday context
    REQUIRE_FULL_MICRO_CONTEXT: bool = os.getenv(
        "REQUIRE_FULL_MICRO_CONTEXT",
        "true",
    ).strip().lower() in ("1", "true", "yes", "on")
    MACRO_MONTHS: int = 1
    HTF_MONTHS: int = 3

    # Minimum candles needed for meaningful analysis (per timeframe)
    MIN_CANDLES_INTRADAY: int = 5
    MIN_CANDLES_5M: int = MIN_CANDLES_INTRADAY  # backwards-compatible alias
    MIN_CANDLES_DAILY: int = 5
    MIN_CANDLES_WEEKLY: int = 3

    # Replay. Raw Yahoo intraday data is cached at 5m, then resampled here.
    DECISION_INTERVAL: str = os.getenv("DECISION_INTERVAL", "15min")
    INTRADAY_TIMEFRAME_LABEL: str = os.getenv("INTRADAY_TIMEFRAME_LABEL", DECISION_INTERVAL)

    # Agent
    DECISION_MODE: str = os.getenv("DECISION_MODE", "exploratory").strip().lower()
    MAX_TOOL_CALLS_PER_DECISION: int = int(os.getenv("MAX_TOOL_CALLS_PER_DECISION", "8"))
    LLM_TEMPERATURE: float = 0.2
    LLM_MAX_TOKENS: int = 2000

    # Signal gating.
    MIN_MINUTES_BETWEEN_SIGNALS: int = int(os.getenv("MIN_MINUTES_BETWEEN_SIGNALS", "15"))

    # Trade validation
    CAPITAL_CAP: float = 30000.0  # ₹30,000 per trade
    ORDER_CHARGE_PER_LEG: float = 30.0  # ₹30 per leg
    TOTAL_ORDER_CHARGES: float = 60.0  # ₹60 round-trip
    MIN_REWARD_TO_RISK: float = 2.0  # 2:1 minimum
    CANDIDATE_GROSS_REWARD_TO_RISK: float = float(os.getenv("CANDIDATE_GROSS_REWARD_TO_RISK", "3.0"))

    # Evaluation
    EVAL_HORIZONS: list = [15, 30]  # T+15m, T+30m in minutes

    # Redis
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_DB: int = int(os.getenv("REDIS_DB", "0"))

    # Postgres & Replay Defaults
    STATE_BACKEND: str = os.getenv("STATE_BACKEND", "postgres")
    RUNTIME_SCHEMA: str = os.getenv("PG_SCHEMA", "historical")
    AGENT_WORKFLOW: str = os.getenv("AGENT_WORKFLOW", "direct")
    PRODUCT_TYPE: str = os.getenv("PRODUCT_TYPE", "CNC")
    INSTRUMENT_TYPE: str = os.getenv("INSTRUMENT_TYPE", "equity_cash")
    STARTING_CAPITAL: float = float(os.getenv("STARTING_CAPITAL", "100000.0"))
    MAX_CAPITAL_PER_TRADE: float = float(os.getenv("MAX_CAPITAL_PER_TRADE", "30000.0"))
    RISK_BUDGET_PCT: float = float(os.getenv("RISK_BUDGET_PCT", "0.01"))
    MAX_DAILY_LOSS: float = float(os.getenv("MAX_DAILY_LOSS", "3000.0"))
    MAX_TRADES_PER_DAY: int = int(os.getenv("MAX_TRADES_PER_DAY", "5"))
    JSONL_AUDIT_ENABLED: bool = os.getenv("JSONL_AUDIT_ENABLED", "true").strip().lower() in ("1", "true", "yes", "on")

    # Version tracking for reproducibility
    AGENT_PROMPT_VERSION: str = os.getenv("AGENT_PROMPT_VERSION", "pa-checklist-v2")
    AGENT_TOOLSET_VERSION: str = os.getenv("AGENT_TOOLSET_VERSION", "structure-vwap-profile-v1")


    @classmethod
    def ensure_dirs(cls):
        """Create all output directories."""
        cls.DATA_DIR.mkdir(exist_ok=True)
        cls.RAW_DATA_DIR.mkdir(exist_ok=True)
        cls.OUTPUT_DIR.mkdir(exist_ok=True)
        cls.CHARTS_DIR.mkdir(exist_ok=True)
        cls.JOURNAL_DIR.mkdir(exist_ok=True)

    @classmethod
    def validate(cls):
        """Ensure required config is present."""
        missing = []
        if not cls.API_KEY:
            missing.append("LLM_API_KEY/OPENROUTER_API_KEY/OR_APIKEY or DEEPSEEK_API_KEY")
        if not cls.BASE_URL:
            missing.append("LLM_BASE_URL/OPENROUTER_BASE_URL/OPENROUTER_BASEURL or BASE_URL")
        if not cls.MODEL_NAME:
            missing.append("LLM_MODEL_NAME/OPENROUTER_MODEL_NAME/GROK_MODEL_NAME or MODEL_NAME")
        if missing:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing)}. "
                f"Check your .env file."
            )


config = Config()
