"""
DART Decision Agent: orchestrates LLM calls with tool use.
The agent receives a MarketStatePackage, can request tools,
and outputs a structured BUY/SELL/HOLD signal.
"""
import base64
import json
import mimetypes
import re
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
                
                # Fetch memory
                episodes = uow.memory.get_episodes(config.SYMBOL, limit=3)
                reflections = uow.memory.get_reflections(config.SYMBOL, limit=3)
                
                mem_lines = []
                if episodes:
                    mem_lines.append("Recent Episodes:")
                    for ep in episodes:
                        mem_lines.append(f"  - Setup: {ep.get('setup_tags')}, Action: {ep.get('action')}, Outcome: {ep.get('outcome_label')} ({ep.get('outcome_net_r')}R)")
                if reflections:
                    mem_lines.append("Reflections/Lessons:")
                    for refl in reflections:
                        mem_lines.append(f"  - Lesson: {refl.get('lesson')} (confidence: {refl.get('confidence')})")
                memory_summary = "\n".join(mem_lines)

        tool_descriptions = ToolHarness.get_tool_descriptions()
        initial_user_text = build_user_prompt(
            market_state_text,
            tool_descriptions,
            portfolio_summary=portfolio_summary,
            session_summary=session_summary,
            memory_summary=memory_summary,
        )
        messages = [
            {"role": "system", "content": build_system_prompt()},
            {
                "role": "user",
                "content": self._build_initial_user_content(
                    initial_user_text,
                    market_state_package,
                ),
            },
        ]

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
                        AnalysisPlan.model_validate(parsed)
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
                    messages.append({
                        "role": "assistant",
                        "content": response
                    })
                    messages.append({
                        "role": "user",
                        "content": "Plan accepted. Now retrieve memories/lessons if needed, execute your required tools in sequence, and synthesize your final signal when ready."
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

                # Execute tool
                tool_result = harness.execute(tool_name, tool_args)
                tool_log.append({
                    "round": round_num,
                    "tool": tool_name,
                    "arguments": tool_args,
                    "reason": tool_reason,
                    "result": tool_result,
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
