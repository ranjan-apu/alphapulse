"""
DART Decision Agent: orchestrates LLM calls with tool use.
The agent receives a MarketStatePackage, can request tools,
and outputs a structured BUY/SELL/HOLD signal.
"""
import base64
import json
import mimetypes
import re
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

from openai import OpenAI

from config import config
from core.tools import ToolHarness
from agent.prompts import (
    build_system_prompt,
    build_user_prompt,
    TOOL_RESULT_PROMPT,
    FINAL_REMINDER,
)
from agent.schema import AnalysisPlan


def _extract_json(text: str) -> Optional[Dict]:
    """Extract JSON object from LLM response, handling markdown code fences."""
    text = text.strip()

    # Try to find JSON in markdown code fences
    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
    if json_match:
        text = json_match.group(1).strip()

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find first { ... } block
    brace_match = re.search(r'\{[\s\S]*\}', text)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    return None


def _image_part_from_path(path: str) -> Optional[Dict[str, Any]]:
    """Build an OpenAI-compatible image_url content part from a local chart path."""
    if not path:
        return None

    chart_path = Path(path)
    if not chart_path.exists() or not chart_path.is_file():
        return None

    mime_type = mimetypes.guess_type(str(chart_path))[0] or "image/png"
    data = base64.b64encode(chart_path.read_bytes()).decode("ascii")
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:{mime_type};base64,{data}",
            "detail": config.VISION_IMAGE_DETAIL,
        },
    }


class DartAgent:
    """
    The DART decision agent. Makes one decision per 5m step.
    Can use tools (up to max_tool_calls) before final signal.
    """

    def __init__(self, langfuse_tracer=None):
        self.client = OpenAI(
            api_key=config.API_KEY,
            base_url=config.BASE_URL,
        )
        self.model = config.MODEL_NAME
        self.max_tool_calls = config.MAX_TOOL_CALLS_PER_DECISION
        self.langfuse = langfuse_tracer
        # Stateful conversation and timestamp tracking
        self.conversation_history = []
        self.last_weekly_time = None
        self.last_daily_time = None
        self.last_intraday_time = None
        self.last_session_date = None

    def reset_session(self) -> None:
        """Clear conversation history and tracking timestamps."""
        self.conversation_history = []
        self.last_weekly_time = None
        self.last_daily_time = None
        self.last_intraday_time = None

    def _get_new_candles(self, package: Dict[str, Any]) -> tuple:
        """
        Compare package candle timestamps against tracking variables.
        Returns (new_weekly, new_daily, new_intraday).
        """
        new_weekly = []
        new_daily = []
        new_intraday = []

        # Weekly
        for w in package.get("weekly_summaries", []):
            w_time = w["week"]
            if self.last_weekly_time is None or w_time > self.last_weekly_time:
                new_weekly.append(w)

        # Daily
        for d in package.get("daily_summaries", []):
            d_time = d["date"]
            if self.last_daily_time is None or d_time > self.last_daily_time:
                new_daily.append(d)

        # Intraday
        for c in package.get("recent_intraday_candles", []):
            c_time = c["time"]
            if self.last_intraday_time is None or c_time > self.last_intraday_time:
                new_intraday.append(c)

        return new_weekly, new_daily, new_intraday

    def _build_step_user_prompt(
        self,
        transient_market_state_text: str,
        portfolio_summary: str = "",
        session_summary: str = "",
        memory_summary: str = "",
    ) -> str:
        """Build the compact step prompt for subsequent steps of a session."""
        mode_line = (
            "Decision mode is STRICT: choose BUY only when the validated setup is fully complete. Otherwise SKIP (if flat) or HOLD (if in position)."
            if config.DECISION_MODE == "strict"
            else "Decision mode is EXPLORATORY: look for a testable BUY candidate, use tools when appropriate, and let the validator reject weak trades."
        )

        sections = [
            f"### STEP DECISION POINT at {config.SYMBOL}",
        ]

        if portfolio_summary:
            sections.append(f"---\nPORTFOLIO & POSITION STATE:\n{portfolio_summary}")

        if session_summary:
            sections.append(f"---\nSESSION STATE:\n{session_summary}")

        if memory_summary:
            sections.append(f"---\nRELEVANT MEMORIES:\n{memory_summary}")

        sections.append(f"---\n{transient_market_state_text}")
        sections.append(f"---\n{mode_line}")
        sections.append("Follow the Price-Action Workflow. Note: Your FIRST response MUST be a structured 'analysis_plan' JSON block, not a tool request or final signal. Define your HTF bias, gap classification, questions to resolve, and tools needed first.")

        return "\n\n".join(sections)

    def decide(
        self,
        market_state_package: Dict[str, Any],
        market_state_text: str,
        harness: ToolHarness,
    ) -> Dict[str, Any]:
        """
        Run the decision loop: send market state, process tool requests,
        return final signal.

        Returns dict with:
          - raw_signals: list of LLM responses
          - tool_calls: list of tool requests and results
          - final_signal: parsed final decision
        """
        run_id = harness.run_id
        T = harness.decision_time

        # 1. Reset check at day/session boundary
        if T:
            if self.last_session_date is None or T.date() != self.last_session_date:
                self.reset_session()
                self.last_session_date = T.date()

        portfolio_summary = ""
        session_summary = ""
        memory_summary = ""

        has_open_position = False
        if run_id and T:
            # Let's import ReplayStateService and UnitOfWork dynamically to avoid circular imports
            from db.services import ReplayStateService
            from db.unit_of_work import UnitOfWork
            
            # Fetch latest portfolio snapshot
            port_snap = ReplayStateService.get_latest_portfolio_snapshot(run_id)
            active_pos = ReplayStateService.get_active_position(run_id, config.SYMBOL)
            has_open_position = bool(active_pos)
            
            if port_snap:
                portfolio_summary = (
                    f"- Cash available: ₹{port_snap['cash_available']:,.2f}\n"
                    f"- Capital deployed: ₹{port_snap['capital_deployed']:,.2f}\n"
                )
                if active_pos:
                    portfolio_summary += (
                        f"- Open position: {active_pos['direction']} {active_pos['symbol']}, "
                        f"qty={active_pos['quantity']}, entry=₹{active_pos['executed_entry']:.2f}, "
                        f"stop=₹{active_pos['stop']:.2f}, target=₹{active_pos['target']:.2f}\n"
                        f"- Unrealized P&L: ₹{active_pos.get('unrealized_pnl', 0.0):,.2f}\n"
                    )
                else:
                    portfolio_summary += "- Open position: NONE\n"
                portfolio_summary += (
                    f"- Realized P&L today: ₹{port_snap['realized_pnl']:,.2f}\n"
                    f"- Charges paid today: ₹{port_snap['charges_paid']:,.2f}\n"
                    f"- Trades today: {port_snap['trades_taken_today']} / {port_snap['max_trades_per_day']}\n"
                    f"- Daily loss used: ₹{port_snap['daily_loss_used']:,.2f} / ₹{port_snap['max_daily_loss']:,.2f}"
                )
            
            # Fetch session map
            session_date = T.date()
            session_id = f"sess_{run_id}_{session_date.strftime('%Y%m%d')}"
            with UnitOfWork() as uow:
                sess_map = uow.sessions.get_session_map(session_id)
                if sess_map:
                    opening_range_high = sess_map.get('opening_range_high')
                    opening_range_low = sess_map.get('opening_range_low')
                    session_high = sess_map.get('session_high')
                    session_low = sess_map.get('session_low')
                    session_vwap = sess_map.get('session_vwap')
                    vwap_slope = sess_map.get('vwap_slope')
                    
                    or_high_str = f"₹{opening_range_high:.2f}" if opening_range_high else "None"
                    or_low_str = f"₹{opening_range_low:.2f}" if opening_range_low else "None"
                    high_str = f"₹{session_high:.2f}" if session_high else "None"
                    low_str = f"₹{session_low:.2f}" if session_low else "None"
                    vwap_str = f"₹{session_vwap:.2f}" if session_vwap else "None"
                    slope_str = f"{vwap_slope:.6f}" if vwap_slope else "0.0"

                    session_summary = (
                        f"- Phase: {ReplayStateService.get_session_phase(T).value}\n"
                        f"- Opening range: High={or_high_str}, Low={or_low_str}\n"
                        f"- Session High/Low: High={high_str}, Low={low_str}\n"
                        f"- VWAP: {vwap_str}, Slope={slope_str}\n"
                        f"- Gap classification: {sess_map.get('gap_classification')}\n"
                        f"- Market regime: {sess_map.get('market_regime')}\n"
                        f"- Current bias: {sess_map.get('current_bias')}"
                    )
                
                # Initial generic memory fetch (used before analysis plan is available)
                generic_episodes = uow.memory.get_episodes(config.SYMBOL, limit=3)
                generic_reflections = uow.memory.get_reflections(config.SYMBOL, limit=3)
                
                mem_lines = []
                if generic_episodes:
                    mem_lines.append("Recent Episodes:")
                    for ep in generic_episodes:
                        mem_lines.append(f"  - Setup: {ep.get('setup_tags')}, Action: {ep.get('action')}, Outcome: {ep.get('outcome_label')} ({ep.get('outcome_net_r')}R)")
                if generic_reflections:
                    mem_lines.append("Reflections/Lessons:")
                    for refl in generic_reflections:
                        mem_lines.append(f"  - Lesson: {refl.get('lesson')} (confidence: {refl.get('confidence')})")
                memory_summary = "\n".join(mem_lines)

        # 2. Build conversation history state (Case A or Case B)
        is_first_prompt = len(self.conversation_history) == 0

        if is_first_prompt:
            # 2a. Prepend System Prompt
            self.conversation_history.append({"role": "system", "content": build_system_prompt()})
            
            # 2b. Format all 13 weekly, 22 daily, and 75 intraday candles from the current package
            from core.context import format_market_state_for_prompt
            market_state_text_full = format_market_state_for_prompt(market_state_package, include_candles=True, full_history=True)
            
            tool_descriptions = ToolHarness.get_tool_descriptions()
            initial_user_text = build_user_prompt(
                market_state_text_full,
                tool_descriptions,
                portfolio_summary=portfolio_summary,
                session_summary=session_summary,
                memory_summary=memory_summary,
            )
            
            initial_user_content = self._build_initial_user_content(initial_user_text, market_state_package)
            self.conversation_history.append({"role": "user", "content": initial_user_content})
            
            # Initialize tracking timestamps
            weekly = market_state_package.get("weekly_summaries", [])
            if weekly:
                self.last_weekly_time = weekly[-1]["week"]
            daily = market_state_package.get("daily_summaries", [])
            if daily:
                self.last_daily_time = daily[-1]["date"]
            intraday = market_state_package.get("recent_intraday_candles", [])
            if intraday:
                self.last_intraday_time = intraday[-1]["time"]
        else:
            # 2c. Subsequent step - check for new candles
            new_weekly, new_daily, new_intraday = self._get_new_candles(market_state_package)
            if new_weekly or new_daily or new_intraday:
                from core.context import format_incremental_candles
                incremental_text = format_incremental_candles(new_weekly, new_daily, new_intraday)
                self.conversation_history.append({"role": "user", "content": incremental_text})
                
                # Update tracking timestamps
                if new_weekly:
                    self.last_weekly_time = new_weekly[-1]["week"]
                if new_daily:
                    self.last_daily_time = new_daily[-1]["date"]
                if new_intraday:
                    self.last_intraday_time = new_intraday[-1]["time"]

        # 3. Build the step-specific prompt for this decision point (without candle tables)
        from core.context import format_market_state_for_prompt
        transient_market_state_text = format_market_state_for_prompt(market_state_package, include_candles=False)
        
        step_user_text = self._build_step_user_prompt(
            transient_market_state_text,
            portfolio_summary=portfolio_summary,
            session_summary=session_summary,
            memory_summary=memory_summary,
        )
        step_user_content = self._build_initial_user_content(step_user_text, market_state_package)
        
        # 4. Construct messages history (temporarily appending step prompt to a copy)
        messages = list(self.conversation_history)
        messages.append({"role": "user", "content": step_user_content})

        raw_responses = []
        tool_log = []
        final_signal = None

        # Enforce planned agent workflow: round 0 is AnalysisPlan
        for round_num in range(self.max_tool_calls + 2):  # +2 to account for AnalysisPlan round
            response = self._call_llm(messages)
            raw_responses.append(response)

            if response is None:
                final_signal = self._fallback_signal("LLM returned empty response", has_open_position)
                break

            parsed = _extract_json(response)
            if parsed is None:
                final_signal = self._fallback_signal(f"Could not parse LLM response as JSON: {response[:200]}", has_open_position)
                break

            msg_type = parsed.get("type", "")

            # Enforce analysis plan in round 0
            if round_num == 0:
                if msg_type != "analysis_plan":
                    messages.append({
                        "role": "assistant",
                        "content": response
                    })
                    messages.append({
                        "role": "user",
                        "content": "Please start by outputting a structured 'analysis_plan' first, following the Price-Action Workflow."
                    })
                    continue
                else:
                    try:
                        validated_plan = AnalysisPlan.model_validate(parsed)
                    except Exception as exc:
                        messages.append({
                            "role": "assistant",
                            "content": response
                        })
                        messages.append({
                            "role": "user",
                            "content": f"Your analysis_plan failed schema validation: {exc}. Output a valid analysis_plan JSON object before requesting tools."
                        })
                        continue

                    # Build focused memory context from the analysis plan fields
                    focused_memory = self._build_focused_memory_context(validated_plan, run_id, config.SYMBOL)

                    memory_injection = (
                        "Plan accepted.\n\n"
                        "Retrieved relevant past episodes and lessons matching today's context:\n"
                        f"{focused_memory}\n\n"
                        "Now execute your required tools in sequence, and synthesize your final signal when ready."
                    )
                    messages.append({
                        "role": "assistant",
                        "content": response
                    })
                    messages.append({
                        "role": "user",
                        "content": memory_injection
                    })
                    continue

            if msg_type == "final_signal":
                final_signal = parsed
                break

            elif msg_type == "tool_request":
                if round_num >= self.max_tool_calls + 1:
                    # Force final decision
                    messages.append({"role": "user", "content": FINAL_REMINDER})
                    continue

                tool_name = parsed.get("tool", "")
                tool_args = parsed.get("arguments", {})
                tool_reason = parsed.get("reason", "")

                # Execute tool with latency tracking
                t_tool_start = time.time()
                tool_result = harness.execute(tool_name, tool_args)
                t_tool_elapsed = (time.time() - t_tool_start) * 1000
                tool_log.append({
                    "round": round_num,
                    "tool": tool_name,
                    "arguments": tool_args,
                    "reason": tool_reason,
                    "result": tool_result,
                    "latency_ms": round(t_tool_elapsed, 2),
                })

                # Format tool result for LLM
                result_str = json.dumps(tool_result, indent=2)
                remaining = self.max_tool_calls - harness.call_count
                messages.append({
                    "role": "assistant",
                    "content": json.dumps(parsed),
                })
                messages.append({
                    "role": "user",
                    "content": self._build_tool_result_content(
                        TOOL_RESULT_PROMPT.format(result=result_str, remaining=remaining),
                        tool_result,
                    ),
                })

            else:
                # Unknown type - treat as final attempt
                final_signal = parsed
                if "action" in parsed:
                    break
                messages.append({
                    "role": "user",
                    "content": "Please output your final signal with type='final_signal' and action field."
                })
                continue

        if final_signal is None:
            final_signal = self._fallback_signal("Max rounds reached without final signal", has_open_position)

        # Ensure action is present
        if "action" not in final_signal or final_signal["action"] not in ("BUY", "SELL", "HOLD", "SKIP", "EXIT"):
            final_signal["action"] = "HOLD" if has_open_position else "SKIP"

        return {
            "raw_responses": raw_responses,
            "tool_calls": tool_log,
            "final_signal": final_signal,
        }

    def _build_initial_user_content(
        self,
        text: str,
        market_state_package: Dict[str, Any],
    ) -> Union[str, List[Dict[str, Any]]]:
        """Attach chart images when the configured endpoint supports vision."""
        if not config.VISION_ENABLED:
            return text

        parts: List[Dict[str, Any]] = [{"type": "text", "text": text}]
        chart_paths = market_state_package.get("chart_paths") or {}

        attached = []
        for key in config.VISION_CHART_KEYS:
            image_part = _image_part_from_path(chart_paths.get(key))
            if image_part:
                parts.append({"type": "text", "text": f"Chart image: {key}"})
                parts.append(image_part)
                attached.append(key)

        if not attached:
            parts.append({
                "type": "text",
                "text": "No chart images were attachable; use the text summaries and chart paths only.",
            })

        return parts

    def _build_tool_result_content(
        self,
        text: str,
        tool_result: Dict[str, Any],
    ) -> Union[str, List[Dict[str, Any]]]:
        """Attach chart images produced by charting tools during a tool round."""
        if not config.VISION_ENABLED:
            return text

        parts: List[Dict[str, Any]] = [{"type": "text", "text": text}]
        chart_path = tool_result.get("chart_path")
        image_part = _image_part_from_path(chart_path)
        if image_part:
            parts.append({"type": "text", "text": f"Tool chart image: {tool_result.get('chart_type', 'chart')}"})
            parts.append(image_part)
        return parts

    def _call_llm(self, messages: list) -> Optional[str]:
        """Call the configured OpenAI-compatible LLM API with the current messages."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=config.LLM_TEMPERATURE,
                max_tokens=config.LLM_MAX_TOKENS,
            )
            content = response.choices[0].message.content
            return content
        except Exception as e:
            print(f"  [LLM Error] {e}")
            return None

    def _build_focused_memory_context(
        self,
        analysis_plan: AnalysisPlan,
        run_id: Optional[str],
        symbol: str,
        max_episodes: int = 5,
        max_reflections: int = 5,
    ) -> str:
        """
        Build a focused memory context from analysis plan fields.

        After the analysis plan is validated, this builds the same structured
        retrieval query used by MemoryStore and applies it to persisted
        episodes/reflections, rather than returning generic recent episodes.
        """
        if not run_id or not symbol:
            return "No run context available for memory retrieval."

        from agent.memory import MemoryStore
        from db.unit_of_work import UnitOfWork

        query = MemoryStore().build_retrieval_query(
            symbol=symbol,
            analysis_plan=analysis_plan,
            market_regime=analysis_plan.market_regime or "unclear",
            session_type=analysis_plan.session_type or "unclear",
            gap_type=analysis_plan.gap_type or "no_gap",
            structure_state=analysis_plan.structure_state or "unclear",
            vwap_relation=analysis_plan.vwap_relation or "at_vwap",
            vwap_distance_atr=analysis_plan.vwap_distance_atr,
            profile_location=analysis_plan.profile_location or "no_data",
            price_location=analysis_plan.price_location or "unknown",
            time_bucket=analysis_plan.time_bucket or "unknown",
            volatility_bucket=analysis_plan.volatility_bucket or "unknown",
        )

        plan_regime = (query.get("market_regime") or "").lower()
        plan_session = (query.get("session_type") or "").lower()
        plan_structure = (query.get("structure_state") or "").lower()
        plan_vwap = (query.get("vwap_relation") or "").lower()
        plan_gap = (query.get("gap_type") or "").lower()
        plan_profile = (query.get("profile_location") or "").lower()
        plan_price = (query.get("price_location") or "").lower()
        plan_time = (query.get("time_bucket") or "").lower()
        plan_volatility = (query.get("volatility_bucket") or "").lower()
        plan_setup_tags = set(query.get("setup_tags") or [])
        plan_direction = (query.get("direction") or "").lower()
        direction_aliases = {
            "bullish": {"bullish", "buy", "long"},
            "bearish": {"bearish", "sell", "short"},
        }.get(plan_direction, {plan_direction} if plan_direction else set())

        with UnitOfWork() as uow:
            all_episodes = uow.memory.get_episodes(symbol, limit=50)
            all_reflections = uow.memory.get_reflections(symbol, limit=30)

        if not all_episodes and not all_reflections:
            return "No past episodes or lessons available for this symbol."

        # Score each episode using weighted feature similarity
        # Weights match MemoryStore.retrieve_similar_setups logic in memory.py
        scored_episodes = []
        for ep in all_episodes:
            score = 0.0
            matches = []

            # Regime match (0.20)
            if plan_regime and str(ep.get("market_regime") or "").lower() == plan_regime:
                score += 0.20
                matches.append("regime")

            # Session type match (0.15)
            if plan_session and str(ep.get("session_type") or "").lower() == plan_session:
                score += 0.15
                matches.append("session")

            # Structure state match (0.15)
            if plan_structure and str(ep.get("structure_state") or "").lower() == plan_structure:
                score += 0.15
                matches.append("structure")

            # VWAP relation match (0.10)
            if plan_vwap and str(ep.get("vwap_relation") or "").lower() == plan_vwap:
                score += 0.10
                matches.append("vwap")

            # Gap type match (0.10)
            if plan_gap and str(ep.get("gap_type") or "").lower() == plan_gap:
                score += 0.10
                matches.append("gap")

            # Profile location match (0.05)
            if plan_profile and str(ep.get("profile_location") or "").lower() == plan_profile:
                score += 0.05
                matches.append("profile")

            # Price location match (0.05)
            if plan_price and str(ep.get("price_location") or "").lower() == plan_price:
                score += 0.05
                matches.append("price")

            # Time bucket match (0.05)
            if plan_time and str(ep.get("time_bucket") or "").lower() == plan_time:
                score += 0.05
                matches.append("time")

            # Volatility bucket match (0.05)
            if plan_volatility and str(ep.get("volatility_bucket") or "").lower() == plan_volatility:
                score += 0.05
                matches.append("volatility")

            # Setup tag overlap (0.10)
            ep_tags = set(ep.get("setup_tags") or [])
            if plan_setup_tags and ep_tags:
                overlap = len(plan_setup_tags & ep_tags)
                tag_frac = overlap / max(len(plan_setup_tags), 1)
                score += 0.10 * min(tag_frac, 1.0)
                if overlap > 0:
                    matches.append(f"{overlap} tag(s)")

            if score > 0:
                scored_episodes.append((ep, score, matches))

        # Sort by score descending, take top N
        scored_episodes.sort(key=lambda x: x[1], reverse=True)
        top_episodes = scored_episodes[:max_episodes]

        # Score reflections by tag overlap and direction
        scored_reflections = []
        for ref in all_reflections:
            ref_score = 0.0
            ref_tags = set(ref.get("tags") or [])
            ref_direction = (ref.get("direction") or "").lower()

            if plan_setup_tags and ref_tags:
                overlap = len(plan_setup_tags & ref_tags)
                ref_score += 0.5 * min(overlap / max(len(plan_setup_tags), 1), 1.0)

            if direction_aliases and ref_direction in direction_aliases:
                ref_score += 0.3

            ref_score += 0.2 * ref.get("confidence", 0.0)

            if ref_score > 0:
                scored_reflections.append((ref, ref_score))

        scored_reflections.sort(key=lambda x: x[1], reverse=True)
        top_reflections = scored_reflections[:max_reflections]

        # Build formatted output
        lines = []
        if top_episodes:
            lines.append(f"Retrieved {len(top_episodes)} relevant past episodes (filtered by analysis plan context):")
            for ep, score, matches in top_episodes:
                outcome = ep.get("outcome_label", "unknown")
                net_r = ep.get("outcome_net_r", "?")
                action = ep.get("action", "?")
                tags = ep.get("setup_tags", [])
                lines.append(
                    f"  [{score:.2f}] {action} | Setup: {tags} | "
                    f"Outcome: {outcome} ({net_r}R) | "
                    f"Matched on: {', '.join(matches)}"
                )
        else:
            lines.append("No past episodes matched the current analysis plan context.")

        if top_reflections:
            lines.append(f"\nRelevant lessons ({len(top_reflections)}):")
            for ref, score in top_reflections:
                lesson = ref.get("lesson", "")[:120]
                confidence = ref.get("confidence", 0.0)
                tags = ref.get("tags", [])
                lines.append(f"  [{score:.2f}] \"{lesson}\" (confidence: {confidence}, tags: {tags})")

        return "\n".join(lines)

    def _fallback_signal(self, reason: str, has_open_position: bool) -> Dict:
        """Generate a state-aware fallback signal."""
        action = "HOLD" if has_open_position else "SKIP"
        return {
            "type": "final_signal",
            "action": action,
            "confidence": 0.0,
            "dart": {
                "direction": "unclear",
                "area": "unclear",
                "risk": "unclear",
                "trigger": "unclear",
            },
            "entry": None,
            "stop": None,
            "target": None,
            "rewardRisk": None,
            "net_reward_risk_after_charges": None,
            "quantity": None,
            "deployed_capital": None,
            "reason": f"Fallback {action}: {reason}",
            "invalidation": None,
        }
