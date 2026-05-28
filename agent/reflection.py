"""
Reflection writer: generates lessons from evaluated outcomes.
After evaluator scores a decision, the reflection writer:
1. Analyzes what was correct/wrong
2. Checks if the level was respected
3. Assesses if the trigger was late/early
4. Determines if similar future setups should be traded or avoided
5. Tags for memory retrieval

Input: original signal + market state snapshot + future outcome
Output: MemoryReflection with lesson, tags, confidence level
"""
from typing import Dict, List, Optional
from datetime import datetime
from agent.memory import MemoryReflection

# Known setup tags for categorization
KNOWN_SETUP_TAGS = {
    "breakout", "failed_breakout", "vwap_reclaim", "vwap_rejection",
    "range_middle_trade", "near_prior_high", "near_prior_low",
    "value_area_rejection", "poc_magnet", "lvn_breakout", "hvn_chop",
    "low_volume_breakout", "opening_drive", "late_day_trade",
    "gap_up_above_value", "gap_down_below_value",
    "gap_fill", "gap_and_go", "same_level_retest", "post_stop_reentry",
}


class ReflectionWriter:
    """
    Generates structured reflections from evaluated trade outcomes.

    Quality gates (Section 9.4.5):
    - HIGH: target/stop/order/path clearly resolved
    - MEDIUM: useful but some ambiguity
    - LOW: noisy; store only as episode, do not promote to lesson
    - SKIP: do not write reflection
    """

    def __init__(self):
        self._pending_reflections: List[MemoryReflection] = []

    def write_reflection(
        self,
        signal: dict,
        market_state: dict,
        outcome: dict,
        setup_tags: Optional[List[str]] = None,
    ) -> Optional[MemoryReflection]:
        """
        Generate a reflection from an evaluated trade.

        Args:
            signal: The original agent signal (action, dart, entry, stop, target, etc.)
            market_state: Market context at decision time
            outcome: Evaluation outcome from FeedbackEvaluator
            setup_tags: Tags categorizing the setup type

        Returns:
            MemoryReflection if quality gate passes, None otherwise
        """
        action = signal.get("action", "UNKNOWN")
        outcome_label = outcome.get("outcome", "unknown")

        # Determine reflection confidence level
        confidence_level = self._determine_confidence_level(outcome)

        if confidence_level == "SKIP":
            return None

        # Generate lesson text
        lesson = self._generate_lesson(signal, outcome, action)

        # Extract tags
        tags = setup_tags or []
        # Auto-tag based on outcome
        if outcome_label == "stop_hit":
            tags.append("stop_loss_taken")
        elif outcome_label == "target_hit":
            tags.append("target_reached")
        elif "break" in lesson.lower() or "rejection" in lesson.lower():
            tags.append("structure_event")

        # Deduplicate tags
        tags = list(set(tags))

        # Create reflection with proper episode ID
        import uuid
        episode_id = f"ep_{uuid.uuid4().hex[:12]}"
        
        reflection = MemoryReflection.create(
            symbol=signal.get("symbol", ""),
            lesson=lesson,
            tags=tags,
            source_episode_ids=[episode_id],  # Use UUID, not date format
            direction=action if action in ("BUY", "SELL") else None,
            reflection_level=confidence_level,
            confidence=self._compute_confidence(outcome),
        )

        self._pending_reflections.append(reflection)
        return reflection

    def _determine_confidence_level(self, outcome: dict) -> str:
        """
        Determine reflection confidence level based on outcome clarity.

        See Section 9.4.5:
        HIGH = target/stop/order/path clearly resolved
        MEDIUM = useful but some ambiguity
        LOW = noisy; store only as episode
        SKIP = do not write reflection
        """
        outcome_label = outcome.get("outcome", "unknown")

        # Clearly resolved outcomes
        if outcome_label in ("target_hit", "stop_hit"):
            # Check if path was clean (no ambiguity)
            stop_touched = outcome.get("stop_touched", False)
            target_touched = outcome.get("target_touched", False)

            # Both touched = ambiguous
            if stop_touched and target_touched:
                return "MEDIUM"

            return "HIGH"

        # Somewhat resolved
        if outcome_label in ("square_off_at_close", "stop_first", "target_first"):
            return "MEDIUM"

        # Unclear
        if outcome_label in ("both_touched", "simultaneous_touch", "no_future_data"):
            return "LOW"

        # Not a trade
        if outcome_label == "no_trade":
            hold_quality = outcome.get("hold_quality", "")
            if hold_quality in ("good_hold_avoided_chop", "missed_opportunity"):
                return "MEDIUM"
            return "LOW"

        return "LOW"

    def _compute_confidence(self, outcome: dict) -> float:
        """Compute reflection confidence based on outcome."""
        outcome_label = outcome.get("outcome", "unknown")

        if outcome_label in ("target_hit", "stop_hit"):
            return 0.9
        elif outcome_label in ("square_off_at_close",):
            return 0.6
        elif outcome_label == "no_trade":
            hold_quality = outcome.get("hold_quality", "")
            if hold_quality == "good_hold_avoided_chop":
                return 0.6
            elif hold_quality == "missed_opportunity":
                return 0.5
        return 0.4

    def _generate_lesson(
        self,
        signal: dict,
        outcome: dict,
        action: str,
    ) -> str:
        """Generate a natural-language lesson from the trade outcome."""
        outcome_label = outcome.get("outcome", "unknown")
        dart = signal.get("dart", {})
        entry = signal.get("entry")
        stop = signal.get("stop")
        target = signal.get("target")
        net_r = outcome.get("net_r_multiple", 0)

        if action in ("BUY", "SELL"):
            direction_text = "long" if action == "BUY" else "short"

            if outcome_label == "target_hit":
                return (
                    f"{action} signal at {entry} with stop {stop} and target {target} "
                    f"hit target. Net R: {net_r}R. "
                    f"Setup area: {dart.get('area', 'unknown')}. "
                    f"Trigger: {dart.get('trigger', 'unknown')}. "
                    f"Thesis was valid. Consider this setup type for future trades."
                )

            elif outcome_label == "stop_hit":
                return (
                    f"{action} signal at {entry} hit stop at {stop}. Net R: {net_r}R. "
                    f"Setup area: {dart.get('area', 'unknown')}. "
                    f"Invalidation: {signal.get('invalidation', 'not specified')}. "
                    f"Review whether stop placement was too tight or thesis was premature."
                )

            elif outcome_label == "square_off_at_close":
                sq_price = outcome.get("square_off_price", "?")
                return (
                    f"{action} signal at {entry} neither hit stop nor target. "
                    f"Closed at session end {sq_price}. Net R: {net_r}R. "
                    f"Review whether target was realistic or time horizon was too long."
                )

            elif outcome_label in ("stop_first", "target_first"):
                return (
                    f"{action} signal touched both stop and target. "
                    f"Outcome: {outcome_label}. "
                    f"Stop at {stop}, target at {target}. "
                    f"Setup had conflicting signals - review confluence next time."
                )

            else:
                return (
                    f"{action} signal at {entry}. Outcome: {outcome_label}. "
                    f"Net R: {net_r}R. Setup needs more data to evaluate."
                )

        elif action == "HOLD":
            hold_quality = outcome.get("hold_quality", "")
            if hold_quality == "good_hold_avoided_chop":
                return "HOLD was correct: market was choppy with no clear direction."
            elif hold_quality == "missed_opportunity":
                mfe = outcome.get("max_favorable_excursion_pct", 0)
                return f"HOLD may have missed a {mfe}% move. Review why the setup was not identified."
            return "HOLD outcome unclear."

        elif action == "SKIP":
            # SKIP quality evaluation (Section 9.3.1)
            skip_quality = outcome.get("skip_quality", "neutral")
            if skip_quality == "good_skip_chop":
                return "SKIP was correct: market was noisy with no valid setup."
            elif skip_quality == "missed_long_opportunity":
                return "SKIP may have missed a viable long setup. Review filtering criteria."
            elif skip_quality == "missed_short_opportunity":
                return "SKIP may have missed a viable short setup."
            return "SKIP outcome neutral."

        return f"Trade outcome: {outcome_label}. Net R: {net_r}R."

    def get_pending_reflections(self) -> List[MemoryReflection]:
        """Get pending reflections for memory storage."""
        return self._pending_reflections[:]

    def clear_pending(self):
        self._pending_reflections.clear()
