"""
Agent planner: adds explicit planning step before tool execution.

Implements the ReAct-inspired workflow from Section 2.2 and 6.2:
1. Read compact context
2. Produce analysis plan (regime, levels, volume/auction needed, trade location)
3. Retrieve relevant memories
4. Execute planned tools
5. Synthesize results
6. Emit final signal

Also provides calibration hints when statistics are reliable (Section 6.9).
"""
from typing import Dict, List, Optional, Any

from agent.schema import AnalysisPlan
from agent.memory import MemoryStore


class AgentPlanner:
    """
    Manages the analysis planning step for the DART agent.

    The planner:
    1. Builds the retrieval query from market state + initial plan
    2. Retrieves similar past episodes and relevant reflections
    3. Provides calibration hints (if samples are sufficient)
    4. Formats memory context for the prompt
    """

    def __init__(self, memory_store: Optional[MemoryStore] = None):
        self.memory_store = memory_store or MemoryStore()
        self.calibration_stats: Dict[str, dict] = {}

    def plan_analysis(
        self,
        market_state: dict,
        portfolio_state: dict,
        session_phase: dict,
        has_position: bool,
    ) -> AnalysisPlan:
        """
        Create an initial analysis plan from market state.

        The structured fields (market_regime, etc.) are filled from
        MarketStatePackage deterministically, not by the LLM.

        Args:
            market_state: MarketStatePackage dict
            portfolio_state: Portfolio state dict
            session_phase: Session controller phase info
            has_position: Whether there's an open position

        Returns:
            AnalysisPlan with deterministically-filled context fields
        """
        # Extract deterministic fields from market state
        indicators = market_state.get("indicators", {})
        vwap_context = market_state.get("vwap_context", {})
        gap_context = market_state.get("gap_context", {})
        volume_profile = market_state.get("volume_profile", {})
        market_structure = market_state.get("market_structure", {})

        plan = AnalysisPlan(
            market_regime=market_state.get("regime", "unclear"),
            session_type=market_state.get("session_type", "unclear"),
            gap_type=gap_context.get("gap_type", "no_gap"),
            structure_state=market_structure.get("state", "unclear"),
            vwap_relation=vwap_context.get("relation", "at_vwap"),
            vwap_distance_atr=vwap_context.get("distance_atr"),
            profile_location=volume_profile.get("price_location", "no_data"),
            price_location=market_state.get("price_location", "unknown"),
            time_bucket=self._classify_time_bucket(market_state.get("decision_time", "")),
            volatility_bucket=self._classify_volatility(indicators),
        )

        # Set initial direction bias based on structure and VWAP
        if plan.structure_state in ("bullish_bos",) and plan.vwap_relation == "above_vwap":
            plan.direction_bias = "bullish"
        elif plan.structure_state in ("bearish_bos",) and plan.vwap_relation == "below_vwap":
            plan.direction_bias = "bearish"
        elif plan.structure_state == "range_bound":
            plan.direction_bias = "neutral"

        # Suggest setup tags based on context
        plan.setup_tags = self._suggest_setup_tags(plan, market_state, has_position)

        # Plan tools based on context
        plan.planned_tools = self._suggest_tools(plan, has_position)

        plan.reason = f"Regime: {plan.market_regime}, Structure: {plan.structure_state}, VWAP: {plan.vwap_relation}"
        return plan

    def retrieve_context_memories(
        self,
        symbol: str,
        plan: AnalysisPlan,
    ) -> Dict[str, Any]:
        """
        Retrieve relevant memories for the current decision.

        Called after analysis plan is produced, before tool execution.
        The retrieval query is built deterministically (Section 9.4.3).

        Returns dict with:
        - similar_setups: list of similar past episodes
        - relevant_reflections: list of relevant lessons
        - calibration_hints: current calibration stats
        """
        result = {
            "similar_setups": [],
            "relevant_reflections": [],
            "calibration_hints": None,
        }

        if not self.memory_store:
            return result

        # Build retrieval query from plan fields
        query = self.memory_store.build_retrieval_query(
            symbol=symbol,
            analysis_plan=plan,
            market_regime=plan.market_regime or "unclear",
            session_type=plan.session_type or "unclear",
            gap_type=plan.gap_type or "no_gap",
            structure_state=plan.structure_state or "unclear",
            vwap_relation=plan.vwap_relation or "at_vwap",
            vwap_distance_atr=plan.vwap_distance_atr,
            profile_location=plan.profile_location or "no_data",
            price_location=plan.price_location or "unknown",
            time_bucket=plan.time_bucket or "unknown",
            volatility_bucket=plan.volatility_bucket or "unknown",
        )

        # Retrieve similar setups
        result["similar_setups"] = self.memory_store.retrieve_similar_setups(
            **query, top_k=5
        )

        # Retrieve relevant reflections
        result["relevant_reflections"] = self.memory_store.retrieve_relevant_reflections(
            symbol=symbol,
            setup_tags=plan.setup_tags,
            market_regime=plan.market_regime or "unclear",
            direction=plan.direction_bias if plan.direction_bias != "neutral" else None,
            top_k=5,
        )

        # Get calibration hints
        result["calibration_hints"] = self._get_calibration_hints(plan)

        return result

    def format_memory_context(self, memory_result: Dict[str, Any]) -> str:
        """Format retrieved memories for the agent's prompt."""
        lines = []

        # Similar setups
        similar = memory_result.get("similar_setups", [])
        if similar:
            lines.append("Similar Past Setups:")
            for ep in similar[:3]:
                outcome_str = f"Net R: {ep.outcome_net_r:.2f}R" if ep.outcome_net_r else "N/A"
                lines.append(
                    f"  [{ep.action}] {ep.market_regime}/{ep.session_type}/{ep.structure_state} | "
                    f"Tags: {', '.join(ep.setup_tags[:3])} | "
                    f"Outcome: {ep.outcome_label} ({outcome_str})"
                )

        # Reflections
        reflections = memory_result.get("relevant_reflections", [])
        if reflections:
            lines.append("\nRelevant Lessons:")
            for ref in reflections[:3]:
                lines.append(f"  [{ref.reflection_level}] {ref.lesson}")
                if ref.tags:
                    lines.append(f"    Tags: {', '.join(ref.tags[:5])}")

        # Calibration hints
        hints = memory_result.get("calibration_hints")
        if hints:
            lines.append(f"\nCalibration: {hints}")

        return "\n".join(lines)

    def _suggest_setup_tags(
        self,
        plan: AnalysisPlan,
        market_state: dict,
        has_position: bool,
    ) -> List[str]:
        """Suggest setup tags based on market context."""
        tags = []

        gap_type = plan.gap_type or ""
        structure = plan.structure_state or ""
        vwap_rel = plan.vwap_relation or ""

        # Gap-based tags
        if "gap_and_go" in gap_type:
            tags.append("gap_and_go")
        elif "gap_fade" in gap_type:
            tags.append("gap_fade")
        elif "gap_fill" in gap_type:
            tags.append("gap_fill")

        # Structure-based tags
        if "bullish_bos" in structure:
            tags.append("breakout")
        elif "bearish_bos" in structure:
            tags.append("breakout")
        elif "choch" in structure:
            tags.append("reversal_setup")
        elif "range_bound" in structure:
            tags.append("range_trade")

        # VWAP-based tags
        if vwap_rel == "above_vwap":
            if market_state.get("vwap_context", {}).get("vwap_reclaim"):
                tags.append("vwap_reclaim")
        elif vwap_rel == "below_vwap":
            if market_state.get("vwap_context", {}).get("vwap_rejection"):
                tags.append("vwap_rejection")

        # Session-based tags
        session_type = plan.session_type or ""
        if "opening_drive" in session_type:
            tags.append("opening_drive")
        elif "trend_day" in session_type:
            tags.append("trend_day_trade")

        # Volume profile tags
        profile_loc = plan.profile_location or ""
        if "above_vah" in profile_loc:
            tags.append("above_value")
        elif "below_val" in profile_loc:
            tags.append("below_value")
        elif "inside_value" in profile_loc:
            tags.append("inside_value")

        return tags[:5]  # Max 5 tags

    def _suggest_tools(self, plan: AnalysisPlan, has_position: bool) -> List[str]:
        """Suggest tools based on the current context."""
        tools = []

        if has_position:
            tools.append("get_open_position")
            tools.append("get_portfolio_state")
        else:
            tools.append("get_portfolio_state")

        structure = plan.structure_state or ""
        vwap_rel = plan.vwap_relation or ""

        if "bos" in structure or "choch" in structure or structure == "range_bound":
            tools.append("detect_market_structure")

        if vwap_rel in ("above_vwap", "below_vwap", "at_vwap"):
            tools.append("compute_session_vwap")

        profile_loc = plan.profile_location or ""
        if profile_loc != "no_data":
            tools.append("compute_volume_profile")

        return tools[:6]

    def _classify_time_bucket(self, decision_time_str: str) -> str:
        """Classify time bucket from decision time string."""
        if not decision_time_str:
            return "unknown"

        try:
            from datetime import datetime as dt
            ts = dt.fromisoformat(decision_time_str.replace("+05:30", ""))
            return classify_time_bucket(ts)
        except Exception:
            return "unknown"

    def _classify_volatility(self, indicators: dict) -> str:
        """Classify volatility from indicators."""
        atr = indicators.get("atr_14")
        rsi = indicators.get("rsi_14")

        if atr and rsi:
            # Higher RSI + higher ATR = more volatile
            if rsi > 70 or rsi < 30:
                return "high"
            elif 40 <= rsi <= 60:
                return "medium"
        return "medium"

    def _get_calibration_hints(self, plan: AnalysisPlan) -> Optional[str]:
        """
        Generate calibration hints from stored stats.

        Only shows hints when statistics are reliable
        (Section 6.9: update_after_each_session, min_trades_per_bucket: 20)
        """
        # Check if we have sufficient data for this setup type
        setup_key = f"setup_{plan.setup_tags[0]}" if plan.setup_tags else None
        regime_key = f"regime_{plan.market_regime}"

        hints = []

        for key in [setup_key, regime_key]:
            if key and key in self.calibration_stats:
                stats = self.calibration_stats[key]
                if stats.get("total_trades", 0) >= 20:
                    win_rate = stats.get("win_rate", 0)
                    avg_r = stats.get("avg_net_r", 0)
                    hints.append(
                        f"{key}: {stats['total_trades']} trades, "
                        f"{win_rate:.0%} win rate, {avg_r:.2f} avg net R"
                    )

        if hints:
            return " | ".join(hints)

        # If no calibration data, check if sample is too small to show
        for key in [setup_key, regime_key]:
            if key and key in self.calibration_stats:
                stats = self.calibration_stats[key]
                if stats.get("total_trades", 0) < 20 and stats.get("total_trades", 0) >= 5:
                    return f"[low-confidence, {stats['total_trades']} samples]"

        return None

    def update_calibration(
        self,
        setup_tags: List[str],
        regime: str,
        outcome_net_r: float,
        is_win: bool,
    ):
        """Update calibration statistics after each trade outcome."""
        for tag in setup_tags:
            key = f"setup_{tag}"
            self._update_stats(key, outcome_net_r, is_win)

        if regime:
            key = f"regime_{regime}"
            self._update_stats(key, outcome_net_r, is_win)

    def _update_stats(self, key: str, net_r: float, is_win: bool):
        """Update running statistics for a bucket."""
        if key not in self.calibration_stats:
            self.calibration_stats[key] = {
                "total_trades": 0,
                "wins": 0,
                "sum_net_r": 0.0,
                "win_rate": 0.0,
                "avg_net_r": 0.0,
            }

        stats = self.calibration_stats[key]
        stats["total_trades"] += 1
        if is_win:
            stats["wins"] += 1
        stats["sum_net_r"] += net_r
        stats["win_rate"] = stats["wins"] / stats["total_trades"]
        stats["avg_net_r"] = stats["sum_net_r"] / stats["total_trades"]


# Import at bottom to avoid circular import
from core.regime import classify_time_bucket
