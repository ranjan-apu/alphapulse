"""
Structured output schemas for the DART agent using Pydantic v2.

Defines strict schemas for:
- AnalysisPlan: the agent's planned analysis direction
- DartThesis: DART framework components
- PriceActionChecklist: systematic scoring of setup quality
- ToolRequest: agent requests a tool
- FinalSignal: the agent's final decision

Action semantics (CNC delivery):
- FLAT state: BUY / SKIP only (SELL reserved for MIS future phase)
- OPEN position: HOLD / EXIT
"""
from typing import Literal, Optional, List, Any
from pydantic import BaseModel, Field, model_validator


# ---- Analysis Plan ----

class AnalysisPlan(BaseModel):
    """The agent's planned analysis direction before tool execution."""
    direction_bias: Literal["bullish", "bearish", "neutral", "unclear"] = "neutral"
    setup_tags: List[str] = Field(default_factory=list, description="Proposed setup types")
    areas_of_interest: List[str] = Field(default_factory=list)
    planned_tools: List[str] = Field(default_factory=list)
    reason: str = ""

    # Derived at plan time from MarketStatePackage (deterministic, not LLM-decided)
    market_regime: Optional[str] = None
    session_type: Optional[str] = None
    gap_type: Optional[str] = None
    structure_state: Optional[str] = None
    vwap_relation: Optional[str] = None
    vwap_distance_atr: Optional[float] = None
    profile_location: Optional[str] = None
    price_location: Optional[str] = None
    time_bucket: Optional[str] = None
    volatility_bucket: Optional[str] = None


# ---- DART Thesis ----

class DartThesis(BaseModel):
    """DART framework components for a trade decision."""
    direction: str = Field(description="Higher-timeframe bias and immediate momentum")
    area: str = Field(description="The price zone where action matters")
    risk: str = Field(description="Invalidation level, stop distance, target distance")
    trigger: str = Field(description="The lower-timeframe confirmation for entry")

    def is_complete(self) -> bool:
        """Check if all DART components are filled with meaningful values."""
        for field_name in ("direction", "area", "risk", "trigger"):
            value = getattr(self, field_name)
            if not value or value.lower() in ("unclear", "", "none", "n/a"):
                return False
        return True


# ---- Price Action Checklist ----

class PriceActionChecklist(BaseModel):
    """Systematic scoring of setup quality across multiple dimensions."""
    market_regime: Literal["trend", "range", "volatile", "compression", "unclear"] = "unclear"
    session_type: Literal["trend_day", "range_day", "reversal_day", "inside_day", "opening_drive", "unclear"] = "unclear"
    structure_state: Literal["bullish_bos", "bearish_bos", "range_bound", "choch", "unclear"] = "unclear"

    # Quality scores (0-5)
    location_quality: int = Field(default=0, ge=0, le=5)
    trigger_quality: int = Field(default=0, ge=0, le=5)
    risk_quality: int = Field(default=0, ge=0, le=5)
    volume_confirmation: int = Field(default=0, ge=0, le=5)
    higher_tf_alignment: int = Field(default=0, ge=0, le=5)

    reason_to_wait: Optional[str] = None


# ---- Tool Request ----

class ToolRequest(BaseModel):
    """Agent requests a deterministic tool."""
    type: Literal["tool_request"] = "tool_request"
    tool: str = Field(description="Name of the tool to call")
    arguments: dict = Field(default_factory=dict, description="Tool parameters")
    reason: str = Field(description="Why this tool is needed for the analysis")


# ---- Final Signal ----

class FinalSignal(BaseModel):
    """
    The agent's final trading decision.

    Action semantics:
    - FLAT state (CNC delivery): BUY, SKIP
    - FLAT state (MIS future): BUY, SELL, SKIP
    - OPEN position state: HOLD, EXIT
    """
    type: Literal["final_signal"] = "final_signal"
    action: Literal["BUY", "SELL", "SKIP", "HOLD", "EXIT"]

    # Core components
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    dart: DartThesis = Field(default_factory=lambda: DartThesis(
        direction="", area="", risk="", trigger=""
    ))
    checklist: PriceActionChecklist = Field(default_factory=PriceActionChecklist)

    # BUY/SELL fields (required for entry actions)
    entry: Optional[float] = None
    stop: Optional[float] = None
    target: Optional[float] = None
    gross_reward_risk: Optional[float] = None
    net_reward_risk: Optional[float] = None
    expected_horizon_minutes: Optional[int] = None

    # HOLD/EXIT fields (required for position management)
    position_id: Optional[str] = None
    thesis_health: Optional[Literal["valid", "weakening", "invalidated", "not_applicable"]] = "not_applicable"
    exit_reason: Optional[str] = None
    suggested_exit_price: Optional[float] = None

    # Common
    invalidation: Optional[str] = None
    reason: str = ""

    @model_validator(mode="after")
    def validate_action_fields(self):
        """
        Validate required fields based on action type.
        See Section 6.2.1 of the plan for action-specific rules.
        """
        action = self.action

        if action == "BUY":
            if self.entry is None or self.stop is None or self.target is None:
                raise ValueError("BUY signal requires entry, stop, and target")
            if self.net_reward_risk is None:
                raise ValueError("BUY signal requires net_reward_risk")
            if self.expected_horizon_minutes is None:
                raise ValueError("BUY signal requires expected_horizon_minutes")
            if self.invalidation is None:
                raise ValueError("BUY signal requires invalidation")
            if not self.dart.is_complete():
                raise ValueError("BUY signal requires complete DART thesis (no empty/unclear fields)")

        elif action == "SELL":
            if self.entry is None or self.stop is None or self.target is None:
                raise ValueError("SELL signal requires entry, stop, and target")
            if self.net_reward_risk is None:
                raise ValueError("SELL signal requires net_reward_risk")
            if self.expected_horizon_minutes is None:
                raise ValueError("SELL signal requires expected_horizon_minutes")
            if self.invalidation is None:
                raise ValueError("SELL signal requires invalidation")
            if not self.dart.is_complete():
                raise ValueError("SELL signal requires complete DART thesis (no empty/unclear fields)")

        elif action == "SKIP":
            if not self.reason:
                raise ValueError("SKIP signal requires reason")
            if not self.checklist.reason_to_wait:
                raise ValueError("SKIP signal requires checklist.reason_to_wait explaining why no trade")

        elif action == "HOLD":
            if self.position_id is None:
                raise ValueError("HOLD signal requires position_id")
            if not self.reason:
                raise ValueError("HOLD signal requires reason")

        elif action == "EXIT":
            if self.position_id is None:
                raise ValueError("EXIT signal requires position_id")
            if self.exit_reason is None:
                raise ValueError("EXIT signal requires exit_reason")
            if self.suggested_exit_price is None:
                raise ValueError("EXIT signal requires suggested_exit_price")
            if not self.reason:
                raise ValueError("EXIT signal requires reason")

        return self

    def meets_scoring_thresholds(self) -> bool:
        """
        Check if the signal meets minimum scoring thresholds for a BUY/SELL.

        For BUY/SELL:
        - Direction score >= 3
        - Area (location) score >= 4
        - Trigger score >= 3
        - Risk score >= 4
        """
        if self.action in ("SKIP", "HOLD", "EXIT"):
            return True

        checklist = self.checklist
        return (
            checklist.location_quality >= 4
            and checklist.trigger_quality >= 3
            and checklist.risk_quality >= 4
        )


# ---- Schema Validation Wrapper ----

def validate_llm_output(raw_output: dict) -> tuple:
    """
    Validate raw LLM output against Pydantic schemas.

    Returns:
        (is_valid: bool, parsed_object: Any, errors: List[str])
    """
    msg_type = raw_output.get("type", "")

    try:
        if msg_type == "tool_request":
            parsed = ToolRequest.model_validate(raw_output)
            return True, parsed, []
        elif msg_type == "final_signal":
            parsed = FinalSignal.model_validate(raw_output)
            return True, parsed, []
        else:
            # Try final_signal without type field
            if "action" in raw_output:
                parsed = FinalSignal.model_validate(raw_output)
                return True, parsed, []
            return False, None, [f"Unknown message type: {msg_type}"]
    except Exception as e:
        return False, None, [str(e)]


def signal_to_dict(signal: FinalSignal) -> dict:
    """Convert FinalSignal to dict for JSON serialization and journal."""
    return signal.model_dump(exclude_none=False)
