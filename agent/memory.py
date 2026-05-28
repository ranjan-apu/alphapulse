"""
Layered memory system for the DART agent.

Four memory layers:
1. Working Memory: current decision only (active state, tool outputs, candidate ideas)
2. Session Memory: current trading session (levels, VWAP behavior, predictions, zones)
3. Episodic Memory: past signals and outcomes (with setup tags and context features)
4. Reflection Memory: learned rules and warnings from past outcomes

Implements hybrid retrieval:
- Structured filters (symbol, regime, direction, setup_tags)
- Weighted feature similarity
- Memory decay and staleness
"""
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

from agent.schema import AnalysisPlan


# ---- Working Memory ----

@dataclass
class WorkingMemory:
    """Stores active state for the current decision only."""
    active_market_state: Dict[str, Any] = field(default_factory=dict)
    tool_outputs: List[Dict] = field(default_factory=list)
    candidate_trades: List[Dict] = field(default_factory=list)
    analysis_plan: Optional[AnalysisPlan] = None

    def add_tool_output(self, tool_name: str, result: dict):
        self.tool_outputs.append({"tool": tool_name, "result": result})

    def clear_for_next_decision(self):
        self.tool_outputs = []
        self.candidate_trades = []


# ---- Session Memory ----

@dataclass
class SessionLevel:
    """A price level tracked during the session."""
    level_id: str
    price: float
    level_type: str  # support, resistance, vwap, poc, vah, val, swing_high, swing_low
    state: str = "ACTIVE"  # ACTIVE, TESTED, REJECTED, BROKEN, FLIPPED, INVALIDATED, EXPIRED
    strength: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now().astimezone())


@dataclass
class SessionMemory:
    """Tracks the current trading session's market map."""
    session_id: str = ""
    session_date: str = ""
    symbol: str = ""

    # Opening range
    opening_range_high: Optional[float] = None
    opening_range_low: Optional[float] = None

    # Session extremes
    session_high: Optional[float] = None
    session_low: Optional[float] = None

    # VWAP behavior
    session_vwap: Optional[float] = None
    vwap_slope: Optional[float] = None
    vwap_reclaim_events: List[Dict] = field(default_factory=list)
    vwap_rejection_events: List[Dict] = field(default_factory=list)

    # Volume profile
    current_poc: Optional[float] = None
    current_vah: Optional[float] = None
    current_val: Optional[float] = None

    # Levels
    active_levels: List[SessionLevel] = field(default_factory=list)
    rejected_zones: List[Dict] = field(default_factory=list)
    accepted_zones: List[Dict] = field(default_factory=list)

    # Predictions and bias
    prior_predictions: List[Dict] = field(default_factory=list)
    current_bias: str = "neutral"
    market_regime: str = "unclear"
    session_type: str = "unclear"
    gap_classification: str = "no_gap"

    # Failed setups
    failed_breakouts: List[Dict] = field(default_factory=list)

    def add_level(self, price: float, level_type: str) -> SessionLevel:
        level = SessionLevel(
            level_id=f"lvl_{uuid.uuid4().hex[:8]}",
            price=price,
            level_type=level_type,
        )
        self.active_levels.append(level)
        return level

    def update_level_state(self, level_id: str, new_state: str):
        for level in self.active_levels:
            if level.level_id == level_id:
                level.state = new_state
                return

    def remove_expired_levels(self):
        self.active_levels = [
            l for l in self.active_levels
            if l.state not in ("EXPIRED", "INVALIDATED")
        ]

    def summary_text(self) -> str:
        """Compact text summary for the agent prompt."""
        lines = ["Session Memory:"]

        if self.opening_range_high:
            lines.append(f"  Opening Range: {self.opening_range_low} - {self.opening_range_high}")
        if self.session_high:
            lines.append(f"  Session: H={self.session_high}, L={self.session_low}")
        if self.session_vwap:
            lines.append(f"  VWAP: {self.session_vwap:.2f} (slope: {self.vwap_slope})")
        if self.current_poc:
            lines.append(f"  Volume Profile: POC={self.current_poc}, VAH={self.current_vah}, VAL={self.current_val}")

        lines.append(f"  Regime: {self.market_regime}, Type: {self.session_type}")
        lines.append(f"  Gap: {self.gap_classification}")
        lines.append(f"  Bias: {self.current_bias}")

        if self.active_levels:
            lines.append(f"  Active Levels ({len(self.active_levels)}):")
            for level in self.active_levels[-8:]:
                lines.append(f"    {level.level_type} @ {level.price} [{level.state}]")

        if self.failed_breakouts:
            lines.append(f"  Failed Breakouts: {len(self.failed_breakouts)}")

        return "\n".join(lines)


# ---- Episodic Memory ----

@dataclass
class MemoryEpisode:
    """A stored trade episode with outcome and context tags."""
    episode_id: str
    symbol: str
    action: str
    direction: Optional[str]
    market_regime: str
    session_type: str
    gap_type: str
    structure_state: str
    vwap_relation: str
    vwap_distance_atr: Optional[float]
    profile_location: str
    price_location: str
    time_bucket: str
    volatility_bucket: str
    setup_tags: List[str]
    outcome_net_r: Optional[float]
    outcome_label: str  # 'win', 'loss', 'breakeven', 'ambiguous'
    mfe_pct: Optional[float]
    mae_pct: Optional[float]
    confidence: float = 0.0
    sample_quality: int = 1
    created_at: datetime = field(default_factory=lambda: datetime.now().astimezone())

    @classmethod
    def from_decision(
        cls,
        symbol: str,
        action: str,
        direction: Optional[str],
        market_regime: str,
        session_type: str,
        gap_type: str,
        structure_state: str,
        vwap_relation: str,
        vwap_distance_atr: Optional[float],
        profile_location: str,
        price_location: str,
        time_bucket: str,
        volatility_bucket: str,
        setup_tags: List[str],
        outcome_net_r: Optional[float],
        outcome_label: str,
        mfe_pct: Optional[float] = None,
        mae_pct: Optional[float] = None,
    ) -> "MemoryEpisode":
        return cls(
            episode_id=f"ep_{uuid.uuid4().hex[:12]}",
            symbol=symbol,
            action=action,
            direction=direction,
            market_regime=market_regime,
            session_type=session_type,
            gap_type=gap_type,
            structure_state=structure_state,
            vwap_relation=vwap_relation,
            vwap_distance_atr=vwap_distance_atr,
            profile_location=profile_location,
            price_location=price_location,
            time_bucket=time_bucket,
            volatility_bucket=volatility_bucket,
            setup_tags=setup_tags,
            outcome_net_r=outcome_net_r,
            outcome_label=outcome_label,
            mfe_pct=mfe_pct,
            mae_pct=mae_pct,
        )


# ---- Reflection Memory ----

@dataclass
class MemoryReflection:
    """A learned rule or warning from past outcomes."""
    reflection_id: str
    symbol: str
    lesson: str
    tags: List[str]
    source_episode_ids: List[str]
    direction: Optional[str]
    reflection_level: str  # HIGH, MEDIUM, LOW
    confidence: float = 0.0
    num_supporting_episodes: int = 1
    last_updated: datetime = field(default_factory=lambda: datetime.now().astimezone())

    @classmethod
    def create(
        cls,
        symbol: str,
        lesson: str,
        tags: List[str],
        source_episode_ids: List[str],
        direction: Optional[str] = None,
        reflection_level: str = "LOW",
        confidence: float = 0.0,
    ) -> "MemoryReflection":
        return cls(
            reflection_id=f"ref_{uuid.uuid4().hex[:12]}",
            symbol=symbol,
            lesson=lesson,
            tags=tags,
            source_episode_ids=source_episode_ids,  # Use provided IDs, not generated dates
            direction=direction,
            reflection_level=reflection_level,
            confidence=confidence,
            num_supporting_episodes=len(source_episode_ids),
        )


# ---- Memory Store (in-memory, backed by Postgres later) ----

class MemoryStore:
    """
    Stores and retrieves session, episodic, and reflection memories.

    In the Phase 3 harness, uses in-memory stores.
    Postgres-backed persistence is added when DB is available.
    """

    def __init__(self):
        self._episodes: List[MemoryEpisode] = []
        self._reflections: List[MemoryReflection] = []
        self._session_memory: Optional[SessionMemory] = None
        self._working_memory = WorkingMemory()

        # Config
        self._episodic_half_life_days: int = 30
        self._reflection_half_life_days: int = 60
        self._min_episodes_for_lesson: int = 5
        self._stale_after_days: int = 120
        self._regime_mismatch_penalty: float = 0.5

    @property
    def session(self) -> Optional[SessionMemory]:
        return self._session_memory

    @property
    def working(self) -> WorkingMemory:
        return self._working_memory

    def init_session(self, symbol: str, session_date: str):
        """Initialize session memory for a new trading day."""
        self._session_memory = SessionMemory(
            session_id=f"ses_{uuid.uuid4().hex[:12]}",
            session_date=session_date,
            symbol=symbol,
        )

    def add_episode(self, episode: MemoryEpisode):
        self._episodes.append(episode)

    def add_reflection(self, reflection: MemoryReflection):
        self._reflections.append(reflection)

    def retrieve_similar_setups(
        self,
        symbol: str,
        market_regime: str,
        session_type: str,
        gap_type: str,
        structure_state: str,
        vwap_relation: str,
        vwap_distance_atr: Optional[float],
        profile_location: str,
        price_location: str,
        time_bucket: str,
        volatility_bucket: str,
        setup_tags: List[str],
        direction: Optional[str] = None,
        top_k: int = 5,
    ) -> List[MemoryEpisode]:
        """
        Retrieve similar past episodes using weighted feature similarity.

        Retrieval score weights (from plan Section 9.4.2):
        - 0.20 regime_match
        - 0.15 session_type_match
        - 0.15 structure_match
        - 0.15 profile_vwap_similarity
        - 0.10 gap_type_match
        - 0.10 time_bucket_match
        - 0.10 tag_overlap
        - 0.05 (semantic - placeholder)
        """
        candidates = [e for e in self._episodes if e.symbol == symbol]
        if not candidates:
            return []

        scored = []
        for ep in candidates:
            # Apply recency decay
            age_days = (datetime.now().astimezone() - ep.created_at).days
            recency_weight = 2 ** (-age_days / self._episodic_half_life_days) if self._episodic_half_life_days > 0 else 1.0

            # Skip stale episodes
            if age_days > self._stale_after_days:
                continue

            score = 0.0

            # Regime match (0.20)
            score += 0.20 * (1.0 if ep.market_regime == market_regime else 0.0)

            # Session type match (0.15)
            score += 0.15 * (1.0 if ep.session_type == session_type else 0.0)

            # Structure match (0.15)
            score += 0.15 * (1.0 if ep.structure_state == structure_state else 0.0)

            # Profile/VWAP similarity (0.15) - simplified
            vwap_sim = 1.0 if ep.vwap_relation == vwap_relation else 0.5
            score += 0.15 * vwap_sim

            # Gap type match (0.10)
            score += 0.10 * (1.0 if ep.gap_type == gap_type else 0.0)

            # Time bucket match (0.10)
            score += 0.10 * (1.0 if ep.time_bucket == time_bucket else 0.0)

            # Tag overlap (0.10)
            if setup_tags and ep.setup_tags:
                overlap = len(set(setup_tags) & set(ep.setup_tags))
                tag_score = min(overlap / max(len(setup_tags), 1), 1.0)
                score += 0.10 * tag_score

            # Apply recency decay (but NOT regime mismatch penalty here - that's already
            # handled by scoring 0 for regime_match when regimes differ)
            score *= recency_weight

            scored.append((ep, score))

        # Sort by score descending and return top_k
        scored.sort(key=lambda x: x[1], reverse=True)
        return [ep for ep, _ in scored[:top_k]]

    def retrieve_relevant_reflections(
        self,
        symbol: str,
        setup_tags: List[str],
        market_regime: str,
        direction: Optional[str] = None,
        top_k: int = 5,
    ) -> List[MemoryReflection]:
        """
        Retrieve relevant reflections/lessons based on tags and context.
        """
        candidates = [r for r in self._reflections if r.symbol == symbol]
        if not candidates:
            return []

        scored = []
        for ref in candidates:
            # Apply recency decay
            age_days = (datetime.now().astimezone() - ref.last_updated).days
            recency_weight = 2 ** (-age_days / self._reflection_half_life_days) if self._reflection_half_life_days > 0 else 1.0

            score = 0.0

            # Tag overlap
            if setup_tags and ref.tags:
                overlap = len(set(setup_tags) & set(ref.tags))
                tag_score = min(overlap / max(len(setup_tags), 1), 1.0)
                score += 0.5 * tag_score

            # Confidence weighting
            score += 0.3 * ref.confidence

            # Supporting episodes
            score += 0.2 * min(ref.num_supporting_episodes / self._min_episodes_for_lesson, 1.0)

            score *= recency_weight

            scored.append((ref, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [ref for ref, _ in scored[:top_k]]

    def build_retrieval_query(
        self,
        symbol: str,
        analysis_plan: AnalysisPlan,
        # Fallback values only used if analysis_plan fields are None
        market_regime: str = "unclear",
        session_type: str = "unclear",
        gap_type: str = "no_gap",
        structure_state: str = "unclear",
        vwap_relation: str = "at_vwap",
        vwap_distance_atr: Optional[float] = None,
        profile_location: str = "no_data",
        price_location: str = "unknown",
        time_bucket: str = "unknown",
        volatility_bucket: str = "unknown",
    ) -> dict:
        """
        Build a retrieval query from analysis plan and market state.
        Called after analysis plan is produced and before tool execution.

        AnalysisPlan fields take priority; fallback params are used when plan fields are None.
        """
        return {
            "symbol": symbol,
            "market_regime": analysis_plan.market_regime or market_regime,
            "session_type": analysis_plan.session_type or session_type,
            "gap_type": analysis_plan.gap_type or gap_type,
            "structure_state": analysis_plan.structure_state or structure_state,
            "vwap_relation": analysis_plan.vwap_relation or vwap_relation,
            "vwap_distance_atr": analysis_plan.vwap_distance_atr or vwap_distance_atr,
            "profile_location": analysis_plan.profile_location or profile_location,
            "price_location": analysis_plan.price_location or price_location,
            "time_bucket": analysis_plan.time_bucket or time_bucket,
            "volatility_bucket": analysis_plan.volatility_bucket or volatility_bucket,
            "setup_tags": analysis_plan.setup_tags,
            "direction": analysis_plan.direction_bias
            if analysis_plan.direction_bias not in ("neutral", None)
            else None,
        }
