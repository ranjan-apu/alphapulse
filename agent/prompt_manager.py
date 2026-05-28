"""
LangChain-based prompt registry with versioning, A/B comparison, and experiment tracking.

Key features:
- Single unified ChatPromptTemplate with injectable sections
- Multiple named prompt variants (v1-dart, v2-price-action-checklist, v3-strict-minimal, etc.)
- Each variant stored with metadata: version, description, expected behavior, metrics
- PromptManager: register, retrieve, compare, A/B test prompts
- PromptComparisonResult: track which prompt variant produced which outcome
- Integration with CalibrationTracker for statistical significance

Usage:
    from agent.prompt_manager import PromptManager, PromptVariant

    pm = PromptManager()
    pm.register_variant("pa-checklist-v2", system_template, user_template, metadata)
    
    # Compare two prompts on the same market state
    comparison = pm.compare(prompt_a, prompt_b, market_state, llm)
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
import json
import uuid

# LangChain imports - gracefully degrade if not installed
try:
    from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
    from langchain.prompts.chat import SystemMessage, HumanMessage
    from langchain.schema import BaseMessage
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False
    # Stub classes so the module loads without langchain
    class ChatPromptTemplate:
        def __init__(self, messages): self.messages = messages
        def format_messages(self, **kwargs): return []
        def format(self, **kwargs): return ""


@dataclass
class PromptVariant:
    """A named, versioned prompt variant with metadata and performance tracking."""
    name: str                          # e.g., "dart-v1", "pa-checklist-v2", "strict-minimal-v1"
    version: str                       # e.g., "2.0.0"
    description: str                   # What this prompt is designed to test
    system_template: str               # The raw system prompt template
    user_template: str                 # The raw user prompt template (with {placeholders})
    tool_result_template: str = ""     # Template for tool result messages
    final_reminder: str = ""           # Template for final reminder when max tools reached
    
    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    expected_behavior: str = ""        # e.g., "Higher win rate, more selective entries"
    hypothesis: str = ""               # e.g., "Checklist scoring reduces false breakouts"
    tags: List[str] = field(default_factory=list)
    
    # Performance tracking (updated by CalibrationTracker)
    total_decisions: int = 0
    total_trades: int = 0
    win_rate: Optional[float] = None
    avg_net_r: Optional[float] = None
    avg_tool_calls: Optional[float] = None
    avg_confidence: Optional[float] = None
    skip_rate: Optional[float] = None
    
    # LangChain prompt object (lazy-built)
    _lc_prompt: Optional[Any] = field(default=None, repr=False)
    
    def get_langchain_prompt(self) -> "ChatPromptTemplate":
        """Build and cache the LangChain ChatPromptTemplate."""
        if self._lc_prompt is not None:
            return self._lc_prompt
        
        if not HAS_LANGCHAIN:
            self._lc_prompt = ChatPromptTemplate(messages=[])
            return self._lc_prompt
        
        messages = [
            SystemMessagePromptTemplate.from_template(self.system_template),
            HumanMessagePromptTemplate.from_template(self.user_template),
        ]
        self._lc_prompt = ChatPromptTemplate(messages=messages)
        return self._lc_prompt
    
    def format_prompt(self, **kwargs) -> str:
        """
        Format the full prompt (system + user) as a single string for OpenAI API.
        
        Args:
            **kwargs: All template variables (market_state_text, tool_descriptions, 
                      portfolio_summary, session_summary, memory_summary, etc.)
        
        Returns:
            Full formatted prompt string
        """
        system = self.system_template
        user = self.user_template
        
        # Apply formatting to user template
        for key, value in kwargs.items():
            placeholder = "{" + key + "}"
            if placeholder in user:
                user = user.replace(placeholder, str(value))
        
        return system + "\n\n" + user
    
    def format_messages(self, **kwargs) -> list:
        """
        Format as OpenAI-compatible message list [{"role": ..., "content": ...}].
        
        Returns:
            List of message dicts for OpenAI API
        """
        system = self.system_template
        
        user = self.user_template
        for key, value in kwargs.items():
            placeholder = "{" + key + "}"
            if placeholder in user:
                user = user.replace(placeholder, str(value))
        
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
    
    def to_dict(self) -> dict:
        """Serialize for storage."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "system_template": self.system_template,
            "user_template": self.user_template,
            "tool_result_template": self.tool_result_template,
            "final_reminder": self.final_reminder,
            "created_at": self.created_at,
            "expected_behavior": self.expected_behavior,
            "hypothesis": self.hypothesis,
            "tags": self.tags,
            "stats": {
                "total_decisions": self.total_decisions,
                "total_trades": self.total_trades,
                "win_rate": self.win_rate,
                "avg_net_r": self.avg_net_r,
                "avg_tool_calls": self.avg_tool_calls,
                "avg_confidence": self.avg_confidence,
                "skip_rate": self.skip_rate,
            },
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "PromptVariant":
        """Deserialize from storage."""
        stats = data.pop("stats", {})
        variant = cls(**data)
        for key, value in stats.items():
            if hasattr(variant, key):
                setattr(variant, key, value)
        return variant


@dataclass
class PromptComparisonResult:
    """Result of comparing two prompt variants on the same market state."""
    comparison_id: str
    decision_time: str
    market_state_hash: str
    
    prompt_a_name: str
    prompt_b_name: str
    
    # Raw outputs
    prompt_a_raw_signal: Optional[dict] = None
    prompt_b_raw_signal: Optional[dict] = None
    
    # Parsed decisions
    prompt_a_action: Optional[str] = None
    prompt_b_action: Optional[str] = None
    
    # Do they agree?
    actions_match: bool = False
    entry_match: bool = False  # Within 0.5% if both BUY
    
    # Tool usage
    prompt_a_tool_calls: int = 0
    prompt_b_tool_calls: int = 0
    
    # Confidence
    prompt_a_confidence: Optional[float] = None
    prompt_b_confidence: Optional[float] = None
    
    # Winner (which prompt made the better decision, assessed later by evaluator)
    winner: Optional[str] = None  # 'a', 'b', 'tie', 'pending'
    winner_reason: str = ""
    
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


class PromptManager:
    """
    Registry for prompt variants with comparison and A/B testing support.
    
    Usage:
        pm = PromptManager()
        
        # Register variants
        pm.register_variant(PromptVariant("dart-v1", "1.0", ...))
        pm.register_variant(PromptVariant("pa-checklist-v2", "2.0", ...))
        
        # Get a specific variant
        prompt = pm.get("pa-checklist-v2")
        messages = prompt.format_messages(market_state_text=..., tool_descriptions=...)
        
        # Compare two prompts
        comparison = pm.compare_decisions("dart-v1", "pa-checklist-v2", 
                                          market_state, llm_callable)
        
        # Get A/B test results
        results = pm.get_comparison_stats()
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        self._variants: Dict[str, PromptVariant] = {}
        self._comparisons: List[PromptComparisonResult] = []
        self._active_comparison: Optional[str] = None  # Name of active A/B test
        self._storage_path = storage_path
    
    # ---- Registration ----
    
    def register_variant(self, variant: PromptVariant) -> PromptVariant:
        """Register a prompt variant. Overwrites if same name exists."""
        self._variants[variant.name] = variant
        return variant
    
    def create_variant(
        self,
        name: str,
        version: str,
        description: str,
        system_template: str,
        user_template: str,
        tool_result_template: str = "",
        final_reminder: str = "",
        expected_behavior: str = "",
        hypothesis: str = "",
        tags: Optional[List[str]] = None,
    ) -> PromptVariant:
        """Convenience method to create and register a variant in one call."""
        variant = PromptVariant(
            name=name,
            version=version,
            description=description,
            system_template=system_template,
            user_template=user_template,
            tool_result_template=tool_result_template,
            final_reminder=final_reminder,
            expected_behavior=expected_behavior,
            hypothesis=hypothesis,
            tags=tags or [],
        )
        return self.register_variant(variant)
    
    # ---- Retrieval ----
    
    def get(self, name: str) -> Optional[PromptVariant]:
        """Get a prompt variant by name."""
        return self._variants.get(name)
    
    def list_variants(self) -> List[str]:
        """List all registered prompt variant names."""
        return list(self._variants.keys())
    
    def get_variant_metadata(self, name: str) -> Optional[dict]:
        """Get metadata for a variant."""
        variant = self.get(name)
        return variant.to_dict() if variant else None
    
    # ---- Formatting ----
    
    def format_prompt(self, name: str, **kwargs) -> str:
        """Format a prompt variant with template variables."""
        variant = self.get(name)
        if not variant:
            raise ValueError(f"Unknown prompt variant: {name}")
        return variant.format_prompt(**kwargs)
    
    def format_messages(self, name: str, **kwargs) -> list:
        """Format a prompt variant as OpenAI-compatible messages."""
        variant = self.get(name)
        if not variant:
            raise ValueError(f"Unknown prompt variant: {name}")
        return variant.format_messages(**kwargs)
    
    # ---- Comparison / A/B Testing ----
    
    def compare_decisions(
        self,
        name_a: str,
        name_b: str,
        market_state: dict,
        llm_callable: Callable[[list], dict],  # fn(messages) -> raw_signal
        tool_harness=None,
    ) -> PromptComparisonResult:
        """
        Compare two prompt variants by running both on the same market state.
        
        Args:
            name_a: First prompt variant name
            name_b: Second prompt variant name
            market_state: MarketStatePackage dict with all template variables
            llm_callable: Function that takes messages list and returns raw LLM output dict
            tool_harness: Optional ToolHarness for tool calls
        
        Returns:
            PromptComparisonResult with both decisions
        """
        variant_a = self.get(name_a)
        variant_b = self.get(name_b)
        
        if not variant_a or not variant_b:
            raise ValueError(f"Unknown prompt variant: {name_a if not variant_a else name_b}")
        
        import hashlib
        state_hash = hashlib.md5(
            json.dumps(market_state, sort_keys=True, default=str).encode()
        ).hexdigest()[:12]
        
        comparison = PromptComparisonResult(
            comparison_id=f"cmp_{uuid.uuid4().hex[:12]}",
            decision_time=str(market_state.get("decision_time", "")),
            market_state_hash=state_hash,
            prompt_a_name=name_a,
            prompt_b_name=name_b,
        )
        
        # Extract template variables from market state
        template_vars = self._extract_template_vars(market_state)
        
        # Run prompt A
        try:
            messages_a = variant_a.format_messages(**template_vars)
            result_a = llm_callable(messages_a)
            comparison.prompt_a_raw_signal = result_a
            comparison.prompt_a_action = result_a.get("action", "UNKNOWN") if result_a else None
            comparison.prompt_a_confidence = result_a.get("confidence") if result_a else None
            comparison.prompt_a_tool_calls = len(result_a.get("tool_calls", [])) if result_a else 0
        except Exception as e:
            comparison.prompt_a_raw_signal = {"error": str(e)}
            comparison.prompt_a_action = "ERROR"
        
        # Run prompt B
        try:
            messages_b = variant_b.format_messages(**template_vars)
            result_b = llm_callable(messages_b)
            comparison.prompt_b_raw_signal = result_b
            comparison.prompt_b_action = result_b.get("action", "UNKNOWN") if result_b else None
            comparison.prompt_b_confidence = result_b.get("confidence") if result_b else None
            comparison.prompt_b_tool_calls = len(result_b.get("tool_calls", [])) if result_b else 0
        except Exception as e:
            comparison.prompt_b_raw_signal = {"error": str(e)}
            comparison.prompt_b_action = "ERROR"
        
        # Compare
        comparison.actions_match = (
            comparison.prompt_a_action == comparison.prompt_b_action
        )
        
        # Check entry proximity if both are BUY
        if comparison.prompt_a_action == "BUY" and comparison.prompt_b_action == "BUY":
            entry_a = (comparison.prompt_a_raw_signal or {}).get("entry")
            entry_b = (comparison.prompt_b_raw_signal or {}).get("entry")
            if entry_a and entry_b and entry_a > 0:
                comparison.entry_match = abs(entry_a - entry_b) / entry_a < 0.005
        
        # Determine winner (initially pending, updated by evaluator later)
        comparison.winner = "pending"
        
        self._comparisons.append(comparison)
        variant_a.total_decisions += 1
        variant_b.total_decisions += 1
        
        return comparison
    
    def record_outcome(
        self,
        comparison_id: str,
        prompt_label: str,  # 'a' or 'b'
        outcome_net_r: float,
        is_win: bool,
    ):
        """
        Record which prompt variant won a comparison based on actual outcome.
        Should be called after the evaluator computes outcomes.
        """
        for comp in self._comparisons:
            if comp.comparison_id == comparison_id:
                # Update variant stats
                variant_name = comp.prompt_a_name if prompt_label == 'a' else comp.prompt_b_name
                variant = self.get(variant_name)
                if variant:
                    variant.total_trades += 1
                    # Running average update
                    if variant.avg_net_r is None:
                        variant.avg_net_r = outcome_net_r
                    else:
                        n = variant.total_trades
                        variant.avg_net_r = (variant.avg_net_r * (n - 1) + outcome_net_r) / n
                    
                    wins = int(variant.win_rate * (n - 1)) if variant.win_rate else 0
                    if is_win:
                        wins += 1
                    variant.win_rate = wins / n
                
                break
    
    def get_comparison_stats(self) -> dict:
        """
        Get aggregate comparison statistics across all comparisons.
        
        Returns dict with per-variant stats and head-to-head records.
        """
        stats = {
            "total_comparisons": len(self._comparisons),
            "variants": {},
            "head_to_head": {},
        }
        
        # Per variant stats
        for name, variant in self._variants.items():
            stats["variants"][name] = {
                "version": variant.version,
                "total_decisions": variant.total_decisions,
                "total_trades": variant.total_trades,
                "win_rate": variant.win_rate,
                "avg_net_r": variant.avg_net_r,
                "skip_rate": variant.skip_rate,
                "avg_confidence": variant.avg_confidence,
            }
        
        # Head-to-head records
        for comp in self._comparisons:
            pair_key = f"{comp.prompt_a_name}_vs_{comp.prompt_b_name}"
            if pair_key not in stats["head_to_head"]:
                stats["head_to_head"][pair_key] = {
                    "total": 0, "actions_match": 0, 
                    "a_wins": 0, "b_wins": 0, "ties": 0,
                }
            h2h = stats["head_to_head"][pair_key]
            h2h["total"] += 1
            if comp.actions_match:
                h2h["actions_match"] += 1
            if comp.winner == 'a':
                h2h["a_wins"] += 1
            elif comp.winner == 'b':
                h2h["b_wins"] += 1
            elif comp.winner == 'tie':
                h2h["ties"] += 1
        
        return stats
    
    def generate_comparison_report(self) -> str:
        """Generate a human-readable comparison report."""
        stats = self.get_comparison_stats()
        lines = []
        lines.append("=" * 70)
        lines.append("PROMPT COMPARISON REPORT")
        lines.append("=" * 70)
        lines.append(f"Total comparisons: {stats['total_comparisons']}")
        lines.append("")
        
        lines.append("Variant Performance:")
        lines.append("-" * 50)
        for name, vs in stats["variants"].items():
            lines.append(f"  {name} (v{vs['version']})")
            lines.append(f"    Decisions: {vs['total_decisions']}")
            lines.append(f"    Trades: {vs['total_trades']}")
            if vs['win_rate'] is not None:
                lines.append(f"    Win Rate: {vs['win_rate']:.1%}")
            if vs['avg_net_r'] is not None:
                lines.append(f"    Avg Net R: {vs['avg_net_r']:+.2f}")
            lines.append("")
        
        if stats["head_to_head"]:
            lines.append("Head-to-Head:")
            lines.append("-" * 50)
            for pair_key, h2h in stats["head_to_head"].items():
                lines.append(f"  {pair_key}")
                lines.append(f"    Comparisons: {h2h['total']}")
                lines.append(f"    Actions Agree: {h2h['actions_match']}/{h2h['total']}")
                lines.append(f"    A wins: {h2h['a_wins']}, B wins: {h2h['b_wins']}, Ties: {h2h['ties']}")
                lines.append("")
        
        return "\n".join(lines)
    
    # ---- Persistence ----
    
    def save(self, path: Optional[str] = None):
        """Save all variants and comparisons to JSON."""
        path = path or self._storage_path
        if not path:
            raise ValueError("No storage path specified")
        
        data = {
            "variants": {name: v.to_dict() for name, v in self._variants.items()},
            "comparisons": [
                {
                    "comparison_id": c.comparison_id,
                    "decision_time": c.decision_time,
                    "prompt_a_name": c.prompt_a_name,
                    "prompt_b_name": c.prompt_b_name,
                    "prompt_a_action": c.prompt_a_action,
                    "prompt_b_action": c.prompt_b_action,
                    "actions_match": c.actions_match,
                    "entry_match": c.entry_match,
                    "prompt_a_tool_calls": c.prompt_a_tool_calls,
                    "prompt_b_tool_calls": c.prompt_b_tool_calls,
                    "winner": c.winner,
                    "winner_reason": c.winner_reason,
                    "created_at": c.created_at,
                }
                for c in self._comparisons
            ],
        }
        
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
    
    def load(self, path: Optional[str] = None):
        """Load variants and comparisons from JSON."""
        path = path or self._storage_path
        if not path:
            raise ValueError("No storage path specified")
        
        try:
            with open(path) as f:
                data = json.load(f)
            
            for name, vdata in data.get("variants", {}).items():
                self._variants[name] = PromptVariant.from_dict(vdata)
            
            for cdata in data.get("comparisons", []):
                comp = PromptComparisonResult(**cdata)
                self._comparisons.append(comp)
        except FileNotFoundError:
            pass
    
    def _extract_template_vars(self, market_state: dict) -> dict:
        """Extract template variables from a MarketStatePackage."""
        return {
            "market_state_text": market_state.get("_formatted_text", ""),
            "tool_descriptions": market_state.get("_tool_descriptions", ""),
            "portfolio_summary": market_state.get("_portfolio_summary", ""),
            "session_summary": market_state.get("_session_summary", ""),
            "memory_summary": market_state.get("_memory_summary", ""),
        }


# ============================================================
# Pre-built Prompt Variants
# ============================================================

# These are the same templates used in prompts.py, packaged as PromptVariant objects.

def build_default_variants() -> Dict[str, PromptVariant]:
    """
    Build and return the default set of prompt variants.
    
    Call this once at startup, then register with PromptManager.
    """
    from agent.prompts import (
        BASE_SYSTEM_PROMPT, STRICT_MODE_PROMPT, EXPLORATORY_MODE_PROMPT,
        TOOL_RESULT_PROMPT, FINAL_REMINDER,
    )
    
    variants = {}
    
    # ---- Variant 1: DART Basic (original from Phase 1) ----
    dart_system = """You are a disciplined trading analyst operating under the DART decision framework.

## DART Framework
- **D**irection: Higher-timeframe bias and immediate momentum.
- **A**rea: The price zone where action matters.
- **R**isk: Invalidation level, stop distance, target distance.
- **T**rigger: The lower-timeframe confirmation.

## Rules
1. If no testable setup exists, output HOLD.
2. Do NOT invent levels. HOLD is a legitimate decision.
3. Every BUY or SELL signal MUST include entry, stop, target, and reward-to-risk.
4. A trade is only valid if net_reward_to_risk >= 2.0 after charges.
5. Base decisions on price action and levels, not just indicators.

Output: JSON with type='final_signal', action, entry, stop, target, rewardRisk, reason."""

    dart_user = """{tool_descriptions}

---

{market_state_text}

---

Decision mode: Look for testable BUY candidates. Output HOLD if uncertain."""

    variants["dart-v1"] = PromptVariant(
        name="dart-v1",
        version="1.0.0",
        description="Original DART framework prompt. Simple, direct, no checklist.",
        system_template=dart_system,
        user_template=dart_user,
        tool_result_template="Tool result: {result}\nRemaining calls: {remaining}.",
        final_reminder="Max tool calls reached. Output final signal now.",
        expected_behavior="Higher trade frequency, lower precision. Baseline for comparison.",
        hypothesis="Simple DART without structured scoring will produce more trades but lower win rate.",
        tags=["baseline", "dart", "v1"],
    )
    
    # ---- Variant 2: Price-Action Checklist v2 (current) ----
    pa_system = BASE_SYSTEM_PROMPT + "\n" + EXPLORATORY_MODE_PROMPT
    pa_user = """{tool_descriptions}

---

{portfolio_summary}

---

{session_summary}

---

{memory_summary}

---

{market_state_text}

---

Decision mode is EXPLORATORY: look for a testable BUY candidate, use tools when appropriate, and let the validator reject weak trades.

Follow the Price-Action Workflow (A through H) and output your tool request or final signal."""

    variants["pa-checklist-v2"] = PromptVariant(
        name="pa-checklist-v2",
        version="2.0.0",
        description="Full price-action workflow with 8-step analysis, checklist scoring, state-aware actions, portfolio/memory context.",
        system_template=pa_system,
        user_template=pa_user,
        tool_result_template=TOOL_RESULT_PROMPT,
        final_reminder=FINAL_REMINDER,
        expected_behavior="Higher selectivity, better R:R on accepted trades. More SKIPs. Better position management.",
        hypothesis="Structured checklist + portfolio awareness reduces false breakouts and improves risk management.",
        tags=["price-action", "checklist", "structured", "v2"],
    )
    
    # ---- Variant 3: Strict Minimal (concise, rules-first) ----
    strict_system = """You are a trading risk manager. Your ONLY job is to find high-probability BUY setups.

RULES (violate any = SKIP):
1. Price must be at a clear support/resistance/VWAP/value edge — not range middle
2. Must have a 15m candle close confirming direction with volume expansion
3. Stop must be at a structural level, max 1.5 ATR away
4. Net R:R >= 2.0 after charges (call calculate_trade_math)
5. Session must allow new entries

Action vocabulary: BUY (enter), SKIP (no trade), HOLD (keep position), EXIT (close early)

Output: JSON with type='final_signal', action, entry, stop, target, net_reward_risk, reason, invalidation."""

    strict_user = """{tool_descriptions}

---

PORTFOLIO: {portfolio_summary}

SESSION: {session_summary}

---

{market_state_text}

---

STRICT MODE: Output BUY only if ALL 5 rules pass. Otherwise SKIP."""

    variants["strict-minimal-v3"] = PromptVariant(
        name="strict-minimal-v3",
        version="3.0.0",
        description="Minimalist rules-first prompt. Concise, no workflow fluff. Tests whether less is more.",
        system_template=strict_system,
        user_template=strict_user,
        tool_result_template="Result: {result} | Calls left: {remaining}",
        final_reminder="Max calls. Output final signal.",
        expected_behavior="Highest selectivity. Fewest trades. Highest win rate but potentially missed opportunities.",
        hypothesis="Concise rules outperform verbose workflows by reducing LLM 'creativity'.",
        tags=["minimal", "strict", "rules-first", "v3"],
    )
    
    return variants
