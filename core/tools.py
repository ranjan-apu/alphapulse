"""
DART Tool Harness: deterministic tools the LLM agent can request.
All tools filter data to decisionTime T (no future leakage).

Whitelist:
- get_candles(timeframe, lookback)
- resample_candles(targetTimeframe)
- compute_indicators(timeframe)
- detect_swings(timeframe)
- find_levels(timeframe)
- summarize_price_action(timeframe)
- estimate_risk(direction, entryIdea)
- calculate_trade_math(entry_price, stop_price, target_price, direction)
- plot_market_view(timeframe)
- plot_volume_view(timeframe)
- plot_context_dashboard()
- get_historical_data(timeframe, startDate, endDate)
"""
from datetime import datetime, timedelta
from typing import Dict, Any

import pandas as pd

from config import config
from core.summarizer import (
    compute_all_indicators,
    detect_swings as _detect_swings,
    find_levels as _find_levels,
    detect_pattern,
    summarize_trend,
    estimate_risk as _estimate_risk,
)
from core.charts import (
    plot_micro_5m_chart,
    plot_decision_zoom_chart,
    plot_volume_chart,
    plot_daily_context_chart,
    plot_weekly_context_chart,
    plot_indicator_panel,
    plot_context_dashboard,
)


class ToolHarness:
    """
    Mediates tool calls from the LLM agent. Ensures:
    - Only approved tools are executable
    - All data filtered to decisionTime T
    - Max tool calls per decision enforced
    - Tool results returned as structured JSON
    """

    # Tool whitelist with parameter specs
    TOOL_SPECS = {
        "get_candles": {
            "params": ["timeframe", "lookback"],
            "description": "Get recent candles for a timeframe ending at decision time T.",
        },
        "resample_candles": {
            "params": ["targetTimeframe"],
            "description": "Resample cached intraday source data into higher timeframe candles.",
        },
        "compute_indicators": {
            "params": ["timeframe"],
            "description": "Compute RSI, ATR, MA slope, momentum, and volume change.",
        },
        "detect_swings": {
            "params": ["timeframe"],
            "description": "Detect recent swing highs and lows.",
        },
        "find_levels": {
            "params": ["timeframe"],
            "description": "Find nearby support and resistance levels.",
        },
        "summarize_price_action": {
            "params": ["timeframe"],
            "description": "Summarize trend, pattern, and price location.",
        },
        "estimate_risk": {
            "params": ["direction", "entryIdea"],
            "description": "Estimate stop, target, and reward-to-risk for a candidate entry.",
        },
        "calculate_trade_math": {
            "params": ["entry_price", "stop_price", "target_price", "direction", "capital_cap", "order_charge"],
            "description": "Calculate quantity, deployed capital, risk, profit, charges, and net R:R.",
        },
        "plot_market_view": {
            "params": ["timeframe"],
            "description": "Generate a price chart for a timeframe.",
        },
        "plot_volume_view": {
            "params": ["timeframe"],
            "description": "Generate volume bar chart.",
        },
        "plot_context_dashboard": {
            "params": [],
            "description": "Generate combined context dashboard chart.",
        },
        "get_historical_data": {
            "params": ["timeframe", "startDate", "endDate", "startDaysAgo", "endDaysAgo", "maxCandles"],
            "description": "Pull additional historical candles for a specific period, always capped at decision time T. Prefer ISO startDate/endDate; days-ago arguments are also accepted.",
        },
    }

    def __init__(
        self,
        df_5m: pd.DataFrame,
        df_daily: pd.DataFrame,
        df_weekly: pd.DataFrame,
        decision_time: datetime,
    ):
        self.df_5m = df_5m
        self.df_daily = df_daily
        self.df_weekly = df_weekly
        self.decision_time = decision_time
        self.call_count = 0
        self.max_calls = config.MAX_TOOL_CALLS_PER_DECISION

    def _get_data_for_timeframe(self, timeframe: str) -> pd.DataFrame:
        """Get the appropriate DataFrame for a timeframe, filtered to <= T."""
        tf_lower = timeframe.lower()

        decision_tf = config.DECISION_INTERVAL.lower()
        if tf_lower in ("intraday", "micro", decision_tf, decision_tf.replace("min", "m")):
            return self.df_5m[self.df_5m.index <= self.decision_time]
        elif tf_lower in ("5m", "5min", "5t"):
            return self.df_5m[self.df_5m.index <= self.decision_time]
        elif tf_lower in ("15min", "15t", "15m"):
            from data.collector import resample_to_timeframe
            return resample_to_timeframe(
                self.df_5m[self.df_5m.index <= self.decision_time], "15min"
            )
        elif tf_lower in ("30min", "30t", "30m"):
            from data.collector import resample_to_timeframe
            return resample_to_timeframe(
                self.df_5m[self.df_5m.index <= self.decision_time], "30min"
            )
        elif tf_lower in ("1h", "60min", "hourly"):
            from data.collector import resample_to_timeframe
            return resample_to_timeframe(
                self.df_5m[self.df_5m.index <= self.decision_time], "1h"
            )
        elif tf_lower in ("daily", "1d", "day", "d", "macro"):
            from core.context_window import get_completed_daily_context
            completed, _, _ = get_completed_daily_context(self.df_daily, self.decision_time, months=config.MACRO_MONTHS)
            return completed
        elif tf_lower in ("weekly", "1w", "week", "w", "htf"):
            from core.context_window import get_completed_weekly_context
            completed, _, _ = get_completed_weekly_context(self.df_weekly, self.decision_time, months=config.HTF_MONTHS)
            return completed
        else:
            # Default: 5m
            return self.df_5m[self.df_5m.index <= self.decision_time]

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a tool. Validates tool name and filters data.
        Returns tool result as a dict.
        """
        if self.call_count >= self.max_calls:
            return {
                "error": f"Maximum tool call limit ({self.max_calls}) reached. Make your final decision now.",
                "max_calls_reached": True,
            }

        if tool_name not in self.TOOL_SPECS:
            return {
                "error": f"Unknown tool '{tool_name}'. Available tools: {list(self.TOOL_SPECS.keys())}"
            }

        self.call_count += 1

        try:
            result = self._dispatch(tool_name, arguments)
            result["_tool_call_number"] = self.call_count
            result["_remaining_calls"] = self.max_calls - self.call_count
            return result
        except Exception as e:
            return {
                "error": f"Tool '{tool_name}' failed: {str(e)}",
                "_tool_call_number": self.call_count,
                "_remaining_calls": self.max_calls - self.call_count,
            }

    def _dispatch(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Route tool call to handler."""
        handlers = {
            "get_candles": self._get_candles,
            "resample_candles": self._resample_candles,
            "compute_indicators": self._compute_indicators,
            "detect_swings": self._detect_swings_tool,
            "find_levels": self._find_levels_tool,
            "summarize_price_action": self._summarize_price_action,
            "estimate_risk": self._estimate_risk_tool,
            "calculate_trade_math": self._calculate_trade_math,
            "plot_market_view": self._plot_market_view,
            "plot_volume_view": self._plot_volume_view,
            "plot_context_dashboard": self._plot_context_dashboard_tool,
            "get_historical_data": self._get_historical_data,
        }
        handler = handlers[tool_name]
        return handler(args)

    # ---- Tool implementations ----

    def _get_candles(self, args: Dict) -> Dict:
        timeframe = args.get("timeframe", config.DECISION_INTERVAL)
        lookback = int(args.get("lookback", 20))

        df = self._get_data_for_timeframe(timeframe)
        df = df.iloc[-lookback:]

        candles = []
        for idx, row in df.iterrows():
            candles.append({
                "time": str(idx),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": int(row["volume"]),
            })

        return {
            "timeframe": timeframe,
            "lookback": lookback,
            "candle_count": len(candles),
            "candles": candles,
        }

    def _resample_candles(self, args: Dict) -> Dict:
        target_tf = args.get("targetTimeframe", "15min")
        from data.collector import resample_to_timeframe

        df_5m_filtered = self.df_5m[self.df_5m.index <= self.decision_time]
        resampled = resample_to_timeframe(df_5m_filtered, target_tf)

        candles = []
        for idx, row in resampled.iterrows():
            candles.append({
                "time": str(idx),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": int(row["volume"]),
            })

        return {
            "target_timeframe": target_tf,
            "candle_count": len(candles),
            "candles": candles[-10:],  # Last 10 for context
        }

    def _compute_indicators(self, args: Dict) -> Dict:
        timeframe = args.get("timeframe", config.DECISION_INTERVAL)
        df = self._get_data_for_timeframe(timeframe)
        indicators = compute_all_indicators(df)
        return {"timeframe": timeframe, "indicators": indicators}

    def _detect_swings_tool(self, args: Dict) -> Dict:
        timeframe = args.get("timeframe", config.DECISION_INTERVAL)
        df = self._get_data_for_timeframe(timeframe)
        swings = _detect_swings(df)
        return {"timeframe": timeframe, "swings": swings}

    def _find_levels_tool(self, args: Dict) -> Dict:
        timeframe = args.get("timeframe", config.DECISION_INTERVAL)
        df = self._get_data_for_timeframe(timeframe)
        current_price = float(df["close"].iloc[-1])
        levels = _find_levels(df, current_price)
        return {"timeframe": timeframe, "levels": levels}

    def _summarize_price_action(self, args: Dict) -> Dict:
        timeframe = args.get("timeframe", config.DECISION_INTERVAL)
        df = self._get_data_for_timeframe(timeframe)
        trend = summarize_trend(df, timeframe)
        pattern = detect_pattern(df)
        from core.summarizer import price_location as _price_location
        location = _price_location(df)

        return {
            "timeframe": timeframe,
            "trend": trend,
            "pattern": pattern,
            "price_location": location,
        }

    def _estimate_risk_tool(self, args: Dict) -> Dict:
        direction = args.get("direction", "BUY")
        entry_price = float(args.get("entryIdea", 0))
        df = self._get_data_for_timeframe(config.DECISION_INTERVAL)

        if entry_price == 0:
            entry_price = float(df["close"].iloc[-1])

        risk = _estimate_risk(direction, entry_price, df)
        return risk

    def _calculate_trade_math(self, args: Dict) -> Dict:
        """Deterministic trade math calculator using plan-compliant sizing."""
        entry_price = float(args["entry_price"])
        stop_price = float(args["stop_price"])
        target_price = float(args["target_price"])
        direction = str(args.get("direction", "BUY")).upper()
        capital_cap = float(args.get("capital_cap", config.CAPITAL_CAP))
        starting_capital = float(args.get("starting_capital", 100000.0))
        risk_budget_pct = float(args.get("risk_budget_pct", 0.01))

        # Validate
        if entry_price <= 0 or stop_price <= 0 or target_price <= 0:
            return {"error": "Prices must be positive."}
        if direction not in ("BUY", "SELL"):
            return {"error": "Direction must be BUY or SELL."}

        from core.charges import EquityCashCharges, compute_charges
        from core.position_sizing import PositionSizingConfig, compute_position_size

        sizing_config = PositionSizingConfig(
            starting_capital=starting_capital,
            risk_budget_pct=risk_budget_pct,
            max_capital_per_trade=capital_cap,
            min_net_reward_risk=config.MIN_REWARD_TO_RISK,
        )
        preliminary = compute_position_size(
            entry_price, stop_price, target_price, direction, sizing_config, total_charges=0
        )
        quantity_for_charge = max(preliminary.quantity, 1)
        charge_result = compute_charges(
            EquityCashCharges(), direction, quantity_for_charge, entry_price, target_price
        )
        sizing = compute_position_size(
            entry_price,
            stop_price,
            target_price,
            direction,
            sizing_config,
            total_charges=charge_result.total_charges,
        )

        result = {
            "entry_price": float(round(entry_price, 2)),
            "stop_price": float(round(stop_price, 2)),
            "target_price": float(round(target_price, 2)),
            "direction": direction,
            "capital_cap": float(capital_cap),
            "risk_budget": sizing_config.risk_budget,
            "quantity": sizing.quantity,
            "actual_deployed_capital": sizing.deployed_capital,
            "total_order_charges": charge_result.total_charges,
            "charge_breakdown": charge_result.breakdown,
            "gross_risk": sizing.gross_risk,
            "gross_target_profit": sizing.gross_reward,
            "net_target_profit": sizing.net_reward,
            "net_reward_to_risk": sizing.net_reward_risk,
            "meets_2_to_1_threshold": sizing.net_reward_risk >= config.MIN_REWARD_TO_RISK,
            "risk_per_share": sizing.risk_per_share,
            "reward_per_share": sizing.reward_per_share,
            "gross_rr": sizing.gross_reward_risk,
            "risk_budget_used_pct": sizing.risk_budget_used,
            "capital_ceiling_hit": sizing.capital_ceiling_hit,
            "risk_budget_hit": sizing.risk_budget_hit,
            "warnings": sizing.warnings,
            "actionable": sizing.actionable,
        }

        if sizing.errors:
            result["errors"] = sizing.errors

        return result

    def _plot_market_view(self, args: Dict) -> Dict:
        timeframe = args.get("timeframe", config.DECISION_INTERVAL)
        tf_map = {
            "5m": self.df_5m,
            "15m": self.df_5m,
            "15min": self.df_5m,
            "intraday": self.df_5m,
            "daily": self.df_daily,
            "weekly": self.df_weekly,
        }
        df = tf_map.get(timeframe.lower(), self.df_5m)

        if timeframe.lower() == "daily":
            path = plot_daily_context_chart(df, self.decision_time)
        elif timeframe.lower() == "weekly":
            path = plot_weekly_context_chart(df, self.decision_time)
        else:
            path = plot_micro_5m_chart(df, self.decision_time)

        return {"chart_type": f"market_view_{timeframe}", "chart_path": str(path) if path else None}

    def _plot_volume_view(self, args: Dict) -> Dict:
        timeframe = args.get("timeframe", config.DECISION_INTERVAL)
        df = self._get_data_for_timeframe(timeframe)
        path = plot_volume_chart(df, self.decision_time)
        return {"chart_type": "volume_view", "chart_path": str(path) if path else None}

    def _plot_context_dashboard_tool(self, args: Dict) -> Dict:
        path = plot_context_dashboard(
            self.df_5m, self.df_daily, self.df_weekly, self.decision_time
        )
        return {"chart_type": "context_dashboard", "chart_path": str(path) if path else None}

    def _get_historical_data(self, args: Dict) -> Dict:
        """
        Pull additional historical candles for any timeframe.
        Agent can request data from further back than the default context windows.
        All data is still filtered to <= decision_time T.
        """
        timeframe = args.get("timeframe", "daily")
        max_candles = max(1, min(int(args.get("maxCandles", 60)), 200))
        start_date, end_date = self._parse_historical_range(args)

        # Get the right data source
        tf_lower = timeframe.lower()
        if tf_lower in ("5m", "5min", "5t", "intraday", "micro", config.DECISION_INTERVAL.lower()):
            df_source = self.df_5m
        elif tf_lower in ("15min", "15t", "15m", "30min", "30t", "30m", "1h", "60min", "4h"):
            from data.collector import resample_to_timeframe
            df_source = resample_to_timeframe(self.df_5m[self.df_5m.index <= self.decision_time], timeframe)
        elif tf_lower in ("daily", "1d", "day", "d", "macro"):
            df_source = self.df_daily
        elif tf_lower in ("weekly", "1w", "week", "w", "htf"):
            df_source = self.df_weekly
        else:
            df_source = self.df_daily

        # Filter to [start_date, end_date] AND <= decision_time
        from data.collector import get_historical_slice
        sliced = get_historical_slice(df_source, start_date, end_date)

        # Also ensure nothing after decision_time
        sliced = sliced[sliced.index <= self.decision_time]
        if len(sliced) > max_candles:
            sliced = sliced.iloc[-max_candles:]

        candles = []
        for idx, row in sliced.iterrows():
            candles.append({
                "time": str(idx),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": int(row["volume"]),
            })

        # Compute basic summary
        summary = {}
        if len(sliced) >= 3:
            close_vals = sliced["close"]
            summary["first_close"] = float(close_vals.iloc[0])
            summary["last_close"] = float(close_vals.iloc[-1])
            summary["change_pct"] = float(round((close_vals.iloc[-1] - close_vals.iloc[0]) / close_vals.iloc[0] * 100, 2))
            summary["max_high"] = float(sliced["high"].max())
            summary["min_low"] = float(sliced["low"].min())
            summary["avg_volume"] = float(int(sliced["volume"].mean()))

        trend_label = summarize_trend(sliced, timeframe) if len(sliced) >= 3 else "insufficient data"

        return {
            "timeframe": timeframe,
            "date_range": {
                "requested_start": str(start_date),
                "requested_end": str(end_date),
                "returned_start": str(sliced.index[0]) if len(sliced) else None,
                "returned_end": str(sliced.index[-1]) if len(sliced) else None,
            },
            "candle_count": len(candles),
            "max_candles": max_candles,
            "trend": trend_label,
            "summary": summary,
            "candles": candles,
            "_note": "All data filtered to <= decision time T. No future leakage.",
        }

    def _parse_historical_range(self, args: Dict[str, Any]) -> tuple:
        """Parse explicit ISO dates or days-ago arguments for historical tools."""
        start_date = args.get("startDate") or args.get("start_date")
        end_date = args.get("endDate") or args.get("end_date")

        if start_date:
            start_ts = pd.Timestamp(start_date)
        else:
            start_days_ago = int(args.get("startDaysAgo", 30))
            start_ts = pd.Timestamp(self.decision_time - timedelta(days=start_days_ago))

        if end_date:
            end_ts = pd.Timestamp(end_date)
        else:
            end_days_ago = int(args.get("endDaysAgo", 0))
            end_ts = pd.Timestamp(self.decision_time - timedelta(days=end_days_ago))

        decision_ts = pd.Timestamp(self.decision_time)
        if start_ts.tzinfo is None:
            start_ts = start_ts.tz_localize(decision_ts.tz or "Asia/Kolkata")
        else:
            start_ts = start_ts.tz_convert(decision_ts.tz or "Asia/Kolkata")
        if end_ts.tzinfo is None:
            end_ts = end_ts.tz_localize(decision_ts.tz or "Asia/Kolkata")
        else:
            end_ts = end_ts.tz_convert(decision_ts.tz or "Asia/Kolkata")

        end_ts = min(end_ts, decision_ts)
        if start_ts > end_ts:
            start_ts, end_ts = end_ts, start_ts
            end_ts = min(end_ts, decision_ts)

        return start_ts, end_ts

    @classmethod
    def get_tool_descriptions(cls) -> str:
        """Get human-readable tool descriptions for the LLM system prompt."""
        lines = ["Available tools (request using JSON):"]
        for name, spec in cls.TOOL_SPECS.items():
            lines.append(f"  - {name}: {spec['description']}")
            lines.append(f"    Parameters: {spec['params']}")
        lines.append(f"\nMax {config.MAX_TOOL_CALLS_PER_DECISION} tool calls per decision.")
        lines.append(f'Request format: {{"type": "tool_request", "tool": "<name>", "arguments": {{...}}, "reason": "..."}}')
        return "\n".join(lines)
