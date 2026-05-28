#!/usr/bin/env python3
"""
Prompt Comparison Harness: A/B test different prompt variants on the same market data.

Runs the same market replay through multiple prompt variants and compares:
- Action agreement (do they agree on BUY vs SKIP?)
- Entry/stop/target proximity (when both say BUY, how close are the levels?)
- Tool call efficiency (which prompt uses fewer tool calls?)
- Confidence calibration (which prompt's confidence better predicts outcomes?)
- Outcome quality (which prompt leads to better actual trades?)

Usage:
    python compare_prompts.py --symbol RELIANCE --date 2026-05-28 --prompts dart-v1,pa-checklist-v2,strict-minimal-v3
    
    python compare_prompts.py --quick  # Quick 5-step test
    
    python compare_prompts.py --full-backtest  # Full backtest with all prompts
"""
import sys
import json
import time
import argparse
from datetime import datetime, date
from typing import Dict, List, Optional
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import config
from agent.prompt_manager import (
    PromptManager, PromptVariant, PromptComparisonResult, build_default_variants
)


class PromptComparisonHarness:
    """
    Runs the same market data through multiple prompt variants and compares results.
    
    Integrates with the existing replay loop but routes each decision through
    multiple prompts for head-to-head comparison.
    """
    
    def __init__(
        self,
        prompt_names: List[str],
        llm_client,
        model_name: str,
        max_tool_calls: int = 3,
    ):
        """
        Args:
            prompt_names: List of prompt variant names to compare
            llm_client: OpenAI-compatible client
            model_name: Model name to use for all comparisons
            max_tool_calls: Max tool calls per prompt per decision
        """
        self.prompt_manager = PromptManager()
        
        # Register default variants
        defaults = build_default_variants()
        for name, variant in defaults.items():
            self.prompt_manager.register_variant(variant)
        
        # Filter to requested prompts
        self.active_prompts = []
        for name in prompt_names:
            variant = self.prompt_manager.get(name)
            if variant:
                self.active_prompts.append(variant)
            else:
                print(f"  Warning: Unknown prompt variant '{name}', skipping")
        
        if len(self.active_prompts) < 2:
            raise ValueError(f"Need at least 2 valid prompt variants. Found: {[p.name for p in self.active_prompts]}")
        
        self.llm = llm_client
        self.model = model_name
        self.max_tool_calls = max_tool_calls
        self.comparisons: List[PromptComparisonResult] = []
        
        print(f"\n  Comparing {len(self.active_prompts)} prompts:")
        for p in self.active_prompts:
            print(f"    - {p.name} v{p.version}: {p.description[:80]}...")
    
    def run_single_decision(
        self,
        market_state_package: dict,
        market_state_text: str,
        tool_descriptions: str,
        portfolio_summary: str,
        session_summary: str,
        memory_summary: str,
        tool_harness,
    ) -> Dict[str, dict]:
        """
        Run all active prompts on the same market state and return their decisions.
        
        Returns:
            Dict mapping prompt_name -> decision dict
        """
        template_vars = {
            "market_state_text": market_state_text,
            "tool_descriptions": tool_descriptions,
            "portfolio_summary": portfolio_summary,
            "session_summary": session_summary,
            "memory_summary": memory_summary,
        }
        
        results = {}
        
        for variant in self.active_prompts:
            try:
                decision = self._run_prompt_with_tools(
                    variant, template_vars, tool_harness, market_state_package
                )
                results[variant.name] = decision
            except Exception as e:
                results[variant.name] = {
                    "action": "ERROR",
                    "reason": str(e),
                    "confidence": 0.0,
                }
        
        # Record pairwise comparisons
        names = [p.name for p in self.active_prompts]
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                comp = PromptComparisonResult(
                    comparison_id=f"cmp_{datetime.now().strftime('%Y%m%d%H%M%S')}_{i}_{j}",
                    decision_time=str(market_state_package.get("decision_time", "")),
                    market_state_hash="",
                    prompt_a_name=names[i],
                    prompt_b_name=names[j],
                    prompt_a_raw_signal=results.get(names[i]),
                    prompt_b_raw_signal=results.get(names[j]),
                    prompt_a_action=results.get(names[i], {}).get("action"),
                    prompt_b_action=results.get(names[j], {}).get("action"),
                    actions_match=(
                        results.get(names[i], {}).get("action")
                        == results.get(names[j], {}).get("action")
                    ),
                )
                self.comparisons.append(comp)
                self.prompt_manager._comparisons.append(comp)
        
        return results
    
    def _run_prompt_with_tools(
        self,
        variant: PromptVariant,
        template_vars: dict,
        tool_harness,
        market_state_package: dict,
    ) -> dict:
        """
        Run a single prompt variant through the tool-calling loop.
        
        Returns the final parsed decision.
        """
        import json as _json
        import re
        
        # Format initial messages
        messages = variant.format_messages(**template_vars)
        
        tool_calls_made = 0
        
        for round_num in range(self.max_tool_calls + 1):
            # Call LLM
            response = self._call_llm(messages)
            
            if response is None:
                return {"action": "SKIP", "reason": "LLM returned empty", "confidence": 0.0}
            
            # Parse JSON
            parsed = self._extract_json(response)
            if parsed is None:
                return {"action": "SKIP", "reason": f"Could not parse: {response[:100]}", "confidence": 0.0}
            
            msg_type = parsed.get("type", "")
            
            if msg_type == "final_signal":
                return parsed
            
            elif msg_type == "tool_request":
                if round_num >= self.max_tool_calls:
                    # Add final reminder and retry
                    if variant.final_reminder:
                        messages.append({"role": "user", "content": variant.final_reminder})
                    continue
                
                tool_name = parsed.get("tool", "")
                tool_args = parsed.get("arguments", {})
                
                # Execute tool
                tool_result = tool_harness.execute(tool_name, tool_args)
                tool_calls_made += 1
                
                # Format result
                result_str = _json.dumps(tool_result, indent=2)
                remaining = self.max_tool_calls - tool_calls_made
                
                if variant.tool_result_template:
                    reminder = variant.tool_result_template.format(
                        result=result_str, remaining=remaining
                    )
                else:
                    reminder = f"Tool result:\n{result_str}\n\nRemaining calls: {remaining}."
                
                messages.append({"role": "assistant", "content": _json.dumps(parsed)})
                messages.append({"role": "user", "content": reminder})
            
            else:
                # Unknown type, try to extract action
                if "action" in parsed:
                    return parsed
                messages.append({
                    "role": "user",
                    "content": "Output your final signal with type='final_signal' and action field."
                })
        
        return {"action": "SKIP", "reason": "Max rounds", "confidence": 0.0}
    
    def _call_llm(self, messages: list) -> Optional[str]:
        """Call the LLM and return response text."""
        try:
            response = self.llm.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.2,
                max_tokens=2000,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"  [LLM Error] {e}")
            return None
    
    @staticmethod
    def _extract_json(text: str) -> Optional[dict]:
        """Extract JSON from LLM response."""
        import re, json as _json
        text = text.strip()
        
        # Try markdown code fences
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
        if json_match:
            text = json_match.group(1).strip()
        
        try:
            return _json.loads(text)
        except _json.JSONDecodeError:
            pass
        
        brace_match = re.search(r'\{[\s\S]*\}', text)
        if brace_match:
            try:
                return _json.loads(brace_match.group(0))
            except _json.JSONDecodeError:
                pass
        
        return None
    
    def print_comparison_summary(self):
        """Print a human-readable comparison summary."""
        print("\n" + "=" * 70)
        print("PROMPT COMPARISON SUMMARY")
        print("=" * 70)
        
        if not self.comparisons:
            print("  No comparisons recorded.")
            return
        
        # Per-prompt stats
        stats: Dict[str, dict] = {}
        for comp in self.comparisons:
            for label, name in [("a", comp.prompt_a_name), ("b", comp.prompt_b_name)]:
                if name not in stats:
                    stats[name] = {
                        "BUY": 0, "SELL": 0, "SKIP": 0, "HOLD": 0,
                        "EXIT": 0, "ERROR": 0, "total": 0,
                        "confidence_sum": 0.0,
                    }
                action = comp.prompt_a_action if label == "a" else comp.prompt_b_action
                action = action or "UNKNOWN"
                stats[name][action] = stats[name].get(action, 0) + 1
                stats[name]["total"] += 1
        
        for name, s in stats.items():
            total = s["total"]
            print(f"\n  {name}:")
            print(f"    Total decisions: {total}")
            for action in ["BUY", "SELL", "SKIP", "HOLD", "EXIT"]:
                if s.get(action, 0) > 0:
                    print(f"    {action}: {s[action]} ({s[action]/total*100:.0f}%)")
        
        # Agreement stats
        agreements = sum(1 for c in self.comparisons if c.actions_match)
        print(f"\n  Action Agreement: {agreements}/{len(self.comparisons)} ({agreements/len(self.comparisons)*100:.0f}%)")
        
        # Entry agreement for BUY-BUY pairs
        both_buy = [c for c in self.comparisons if c.prompt_a_action == "BUY" and c.prompt_b_action == "BUY"]
        entry_agree = sum(1 for c in both_buy if c.entry_match)
        if both_buy:
            print(f"  Entry Agreement (both BUY): {entry_agree}/{len(both_buy)} ({entry_agree/len(both_buy)*100:.0f}%)")
    
    def save_results(self, output_path: str):
        """Save comparison results to JSON."""
        results = {
            "prompts_compared": [p.name for p in self.active_prompts],
            "total_comparisons": len(self.comparisons),
            "comparisons": [
                {
                    "comparison_id": c.comparison_id,
                    "decision_time": c.decision_time,
                    "prompt_a": c.prompt_a_name,
                    "prompt_b": c.prompt_b_name,
                    "action_a": c.prompt_a_action,
                    "action_b": c.prompt_b_action,
                    "actions_match": c.actions_match,
                    "entry_match": c.entry_match,
                    "tool_calls_a": c.prompt_a_tool_calls,
                    "tool_calls_b": c.prompt_b_tool_calls,
                    "winner": c.winner,
                }
                for c in self.comparisons
            ],
        }
        
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n  Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare prompt variants on market data")
    parser.add_argument("--symbol", default="RELIANCE", help="Stock symbol")
    parser.add_argument("--date", type=str, help="Replay date (YYYY-MM-DD)")
    parser.add_argument("--prompts", default="dart-v1,pa-checklist-v2,strict-minimal-v3",
                       help="Comma-separated prompt variant names")
    parser.add_argument("--max-steps", type=int, default=10, help="Max decision steps")
    parser.add_argument("--quick", action="store_true", help="Quick 3-step test")
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()
    
    if args.quick:
        args.max_steps = 3
    
    prompt_names = [p.strip() for p in args.prompts.split(",")]
    
    print("\n" + "=" * 70)
    print("PROMPT COMPARISON HARNESS")
    print("=" * 70)
    print(f"  Symbol: {args.symbol}")
    print(f"  Date: {args.date or 'earliest available'}")
    print(f"  Max steps: {args.max_steps}")
    print(f"  Prompts: {prompt_names}")
    
    # Initialize LLM client
    from openai import OpenAI
    client = OpenAI(api_key=config.API_KEY, base_url=config.BASE_URL)
    
    # Create harness
    harness = PromptComparisonHarness(
        prompt_names=prompt_names,
        llm_client=client,
        model_name=config.MODEL_NAME,
        max_tool_calls=config.MAX_TOOL_CALLS_PER_DECISION,
    )
    
    # Load data
    from data.collector import load_cached_data, collect_all_data, resample_to_timeframe
    data = load_cached_data()
    has_cache = all(k in data for k in ["5m", "daily", "weekly"])
    
    if not has_cache:
        print("  No cache found, fetching data...")
        data = collect_all_data()
    
    df_5m = resample_to_timeframe(data["5m"], config.DECISION_INTERVAL)
    df_daily = data["daily"]
    df_weekly = data["weekly"]
    
    # Set up replay
    from core.clock import WalkForwardClock
    from core.context import build_market_state_package, format_market_state_for_prompt
    from core.tools import ToolHarness
    from core.portfolio_state import PortfolioStateManager
    from core.session_controller import MarketSessionController
    from core.cooldown import CooldownController
    from core.gap_context import classify_gap, get_gap_context_dict
    
    clock = WalkForwardClock(df_5m)
    portfolio = PortfolioStateManager()
    session_ctrl = MarketSessionController()
    cooldown = CooldownController()
    
    total_steps = clock.total_steps()
    if args.date:
        replay_date = date.fromisoformat(args.date)
        total_steps = sum(
            1 for point in clock.iterate()
            if point["decision_time"].date() == replay_date
        )
    
    print(f"\n  Total decision points: {total_steps}")
    print(f"  Running {args.max_steps} steps...\n")
    
    step_count = 0
    start_time = time.time()
    
    for decision_point in clock.iterate():
        T = decision_point["decision_time"]
        session_start = decision_point["session_start"]
        session_end = decision_point["session_end"]
        
        if args.date and T.date() != replay_date:
            continue
        
        if args.max_steps and step_count >= args.max_steps:
            break
        
        step_count += 1
        
        # Build market state
        try:
            package = build_market_state_package(T, df_5m, df_daily, df_weekly, {})
        except Exception as e:
            print(f"  [{step_count}] Error building state: {e}")
            continue
        
        market_state_text = format_market_state_for_prompt(package)
        
        # Build context variables
        from core.tools import ToolHarness
        tool_harness = ToolHarness(df_5m, df_daily, df_weekly, T)
        tool_descriptions = ToolHarness.get_tool_descriptions()
        
        # Portfolio summary
        portfolio_state = portfolio.get_portfolio_state()
        portfolio_summary = portfolio_state.summary_text()
        
        # Session summary
        session_phase = session_ctrl.get_session_summary(T)
        session_summary = f"Phase: {session_phase['phase']}, Can open: {session_phase['can_open_new']}"
        
        # Memory summary (placeholder)
        memory_summary = ""
        
        # Gap context
        daily = df_daily[df_daily.index <= T]
        if len(daily) >= 2:
            prior_day = daily.iloc[-2]
            gap = classify_gap(
                prior_close=float(prior_day["close"]),
                today_open=float(package.get("current_price", 0)),  # Approximate
                prior_high=float(prior_day["high"]),
                prior_low=float(prior_day["low"]),
            )
            package["gap_context"] = get_gap_context_dict(gap)
        
        # Run comparison
        results = harness.run_single_decision(
            market_state_package=package,
            market_state_text=market_state_text,
            tool_descriptions=tool_descriptions,
            portfolio_summary=portfolio_summary,
            session_summary=session_summary,
            memory_summary=memory_summary,
            tool_harness=tool_harness,
        )
        
        # Print per-step summary
        actions = " | ".join(
            f"{name}: {r.get('action', '?')}"
            for name, r in results.items()
        )
        agreements = len(set(r.get("action") for r in results.values())) == 1
        agree_mark = "✓" if agreements else "✗"
        print(f"  [{step_count}] {T.strftime('%H:%M')} {agree_mark} {actions}")
    
    elapsed = time.time() - start_time
    print(f"\n  Completed {step_count} steps in {elapsed:.1f}s")
    
    # Print summary
    harness.print_comparison_summary()
    
    # Save results
    output_path = args.output or str(
        config.OUTPUT_DIR / f"prompt_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    harness.save_results(output_path)


if __name__ == "__main__":
    main()
