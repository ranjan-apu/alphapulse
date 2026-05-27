"""
Visual context generation: creates charts for human inspection
and (when supported) model vision input.

Text-only models receive chart summaries and file paths. Vision-capable
OpenAI-compatible models can receive selected chart images from DartAgent.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import FancyBboxPatch
import mplfinance as mpf
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from config import config
from core.summarizer import compute_rsi, compute_atr, find_levels


def _ensure_dir() -> Path:
    config.CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    return config.CHARTS_DIR


def _safe_timestamp(ts) -> str:
    """Convert timestamp to safe filename string."""
    if isinstance(ts, str):
        ts = pd.Timestamp(ts)
    return str(ts).replace(":", "-").replace(" ", "_").replace("+", "").replace("/", "-")


def plot_micro_5m_chart(
    df_5m: pd.DataFrame,
    T: datetime,
    save: bool = True,
) -> Optional[str]:
    """
    Generate micro 5-minute chart: last 3 trading days with current session emphasized.
    Saves to charts directory.
    """
    if len(df_5m) < 10:
        return None

    _ensure_dir()

    fig, ax = plt.subplots(figsize=(14, 6))

    available = df_5m[df_5m.index <= T].copy()
    session_dates = pd.Index(available.index.date).unique()
    selected_dates = set(session_dates[-config.MICRO_DAYS:])
    chart_df = available[[idx.date() in selected_dates for idx in available.index]].copy()

    # Plot the bounded micro context on a numeric axis to avoid date-locator
    # blowups on sparse/resampled intraday data.
    session_start = T.replace(hour=config.SESSION_START_HOUR, minute=config.SESSION_START_MINUTE, second=0)
    prior = chart_df[chart_df.index < session_start]
    current_session = chart_df[(chart_df.index >= session_start) & (chart_df.index <= T)]
    x_all = list(range(len(chart_df)))
    x_lookup = {idx: i for i, idx in enumerate(chart_df.index)}

    # Plot prior sessions in gray
    if len(prior) > 0:
        ax.plot([x_lookup[idx] for idx in prior.index], prior["close"], color="gray", alpha=0.4, linewidth=0.5)

    # Plot current session candles
    if len(current_session) > 0:
        colors = ["green" if c >= o else "red" for o, c in zip(current_session["open"], current_session["close"])]
        current_x = [x_lookup[idx] for idx in current_session.index]
        ax.scatter(current_x, current_session["close"], c=colors, s=10, alpha=0.9)
        ax.plot(current_x, current_session["close"], color="blue", linewidth=0.8, alpha=0.6)

    # Mark decision point
    if T in chart_df.index:
        candle = df_5m.loc[T]
        color = "green" if candle["close"] >= candle["open"] else "red"
        ax.axvline(x=x_lookup[T], color=color, linestyle="--", linewidth=1.5, alpha=0.7, label=f"Decision T={T}")

    # Format
    ax.set_title(f"{config.INSTRUMENT_NAME} - {config.INTRADAY_TIMEFRAME_LABEL} Micro Context (Decision: {T})", fontsize=11)
    ax.set_ylabel("Price (₹)")
    tick_positions = list(range(0, len(chart_df), max(1, len(chart_df)//8)))
    tick_labels = [chart_df.index[i].strftime("%m/%d %H:%M") for i in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)

    plt.tight_layout()
    fname = f"micro_5m_{_safe_timestamp(T)}.png"
    path = config.CHARTS_DIR / fname
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)

    return str(path)


def plot_decision_zoom_chart(
    df_5m: pd.DataFrame,
    T: datetime,
    entry: Optional[float] = None,
    stop: Optional[float] = None,
    target: Optional[float] = None,
    save: bool = True,
) -> Optional[str]:
    """
    Generate zoomed-in chart showing recent candles around decision time T
    with support, resistance, entry, stop, and target levels.
    """
    if len(df_5m) < 5:
        return None

    _ensure_dir()

    # Take last ~50 candles ending at T
    zoom = df_5m[df_5m.index <= T].iloc[-50:].copy()
    if len(zoom) < 5:
        return None

    fig, ax = plt.subplots(figsize=(12, 5))

    # Candlestick data
    for i, (idx, row) in enumerate(zoom.iterrows()):
        color = "green" if row["close"] >= row["open"] else "red"
        body_bottom = min(row["open"], row["close"])
        body_height = abs(row["close"] - row["open"])
        ax.bar(i, body_height, bottom=body_bottom, color=color, width=0.6, alpha=0.8)
        ax.plot([i, i], [row["low"], row["high"]], color="black", linewidth=0.8)

    # Find current price and levels
    current = float(df_5m.loc[T]["close"]) if T in df_5m.index else float(zoom["close"].iloc[-1])
    levels = find_levels(zoom, current)

    # Plot support/resistance
    current_idx = len(zoom) - 1
    if levels.get("nearest_support"):
        ax.axhline(y=levels["nearest_support"], color="green", linestyle="--", alpha=0.5,
                    label=f"Support: {levels['nearest_support']}")
    if levels.get("nearest_resistance"):
        ax.axhline(y=levels["nearest_resistance"], color="red", linestyle="--", alpha=0.5,
                    label=f"Resistance: {levels['nearest_resistance']}")

    # Entry/stop/target
    if entry:
        ax.axhline(y=entry, color="blue", linestyle="-", alpha=0.7, label=f"Entry: {entry}")
    if stop:
        ax.axhline(y=stop, color="red", linestyle=":", alpha=0.7, label=f"Stop: {stop}")
    if target:
        ax.axhline(y=target, color="green", linestyle=":", alpha=0.7, label=f"Target: {target}")

    # Mark T
    ax.axvline(x=current_idx, color="orange", linestyle="--", linewidth=2, alpha=0.8, label=f"T={T}")

    # Labels
    tick_positions = list(range(0, len(zoom), max(1, len(zoom)//8)))
    tick_labels = [zoom.index[i].strftime("%H:%M") for i in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, fontsize=7)

    ax.set_title(f"{config.INSTRUMENT_NAME} - Decision Zoom (T={T})", fontsize=11)
    ax.set_ylabel("Price (₹)")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper left", fontsize=7)

    plt.tight_layout()
    fname = f"decision_zoom_{_safe_timestamp(T)}.png"
    path = config.CHARTS_DIR / fname
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)

    return str(path)


def plot_volume_chart(
    df: pd.DataFrame,
    T: datetime,
    save: bool = True,
) -> Optional[str]:
    """Generate volume bars chart with average volume line."""
    if len(df) < 10:
        return None

    _ensure_dir()

    fig, ax = plt.subplots(figsize=(12, 4))

    vol_data = df[df.index <= T].iloc[-50:].copy()
    avg_vol = vol_data["volume"].mean()

    colors = ["green" if vol_data["close"].iloc[i] >= vol_data["open"].iloc[i] else "red"
              for i in range(len(vol_data))]
    ax.bar(range(len(vol_data)), vol_data["volume"], color=colors, alpha=0.6, width=0.8)
    ax.axhline(y=avg_vol, color="orange", linestyle="--", linewidth=1,
               label=f"Avg Vol: {avg_vol:,.0f}")

    # Highlight unusual volume
    for i, vol in enumerate(vol_data["volume"]):
        if vol > avg_vol * 1.5:
            ax.bar(i, vol, color=colors[i], alpha=1.0, width=0.8, edgecolor="black", linewidth=0.5)

    tick_positions = list(range(0, len(vol_data), max(1, len(vol_data)//8)))
    tick_labels = [vol_data.index[i].strftime("%H:%M") for i in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, fontsize=7)

    ax.set_title(f"{config.INSTRUMENT_NAME} - Volume ({T})", fontsize=11)
    ax.set_ylabel("Volume")
    ax.grid(True, alpha=0.2, axis="y")
    ax.legend(fontsize=8)

    plt.tight_layout()
    fname = f"volume_{_safe_timestamp(T)}.png"
    path = config.CHARTS_DIR / fname
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)

    return str(path)


def plot_daily_context_chart(
    df_daily: pd.DataFrame,
    T: datetime,
    save: bool = True,
) -> Optional[str]:
    """Generate daily context chart with major zones."""
    if len(df_daily) < 5:
        return None

    _ensure_dir()

    fig, ax = plt.subplots(figsize=(12, 5))

    daily = df_daily[df_daily.index <= T].copy()
    if len(daily) < 3:
        return None

    for i, (idx, row) in enumerate(daily.iterrows()):
        color = "green" if row["close"] >= row["open"] else "red"
        body_bottom = min(row["open"], row["close"])
        body_height = abs(row["close"] - row["open"])
        ax.bar(i, body_height, bottom=body_bottom, color=color, width=0.6, alpha=0.8)
        ax.plot([i, i], [row["low"], row["high"]], color="black", linewidth=0.8)

    # Mark current day
    ax.axvline(x=len(daily)-1, color="orange", linestyle="--", alpha=0.7)

    tick_positions = list(range(0, len(daily), max(1, len(daily)//10)))
    tick_labels = [daily.index[i].strftime("%m/%d") for i in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, fontsize=8)

    ax.set_title(f"{config.INSTRUMENT_NAME} - Daily Context ({T.date()})", fontsize=11)
    ax.set_ylabel("Price (₹)")
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    fname = f"daily_context_{_safe_timestamp(T)}.png"
    path = config.CHARTS_DIR / fname
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)

    return str(path)


def plot_weekly_context_chart(
    df_weekly: pd.DataFrame,
    T: datetime,
    save: bool = True,
) -> Optional[str]:
    """Generate weekly context chart with broad trend/regime."""
    if len(df_weekly) < 3:
        return None

    _ensure_dir()

    fig, ax = plt.subplots(figsize=(12, 5))

    weekly = df_weekly[df_weekly.index <= T].copy()
    if len(weekly) < 2:
        return None

    for i, (idx, row) in enumerate(weekly.iterrows()):
        color = "green" if row["close"] >= row["open"] else "red"
        body_bottom = min(row["open"], row["close"])
        body_height = abs(row["close"] - row["open"])
        ax.bar(i, body_height, bottom=body_bottom, color=color, width=0.6, alpha=0.8)
        ax.plot([i, i], [row["low"], row["high"]], color="black", linewidth=0.8)

    ax.axvline(x=len(weekly)-1, color="orange", linestyle="--", alpha=0.7)

    tick_positions = list(range(0, len(weekly), max(1, len(weekly)//8)))
    tick_labels = [weekly.index[i].strftime("%m/%d") for i in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, fontsize=8)

    ax.set_title(f"{config.INSTRUMENT_NAME} - Weekly Context (3 months)", fontsize=11)
    ax.set_ylabel("Price (₹)")
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    fname = f"weekly_context_{_safe_timestamp(T)}.png"
    path = config.CHARTS_DIR / fname
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)

    return str(path)


def plot_indicator_panel(
    df: pd.DataFrame,
    T: datetime,
    save: bool = True,
) -> Optional[str]:
    """Generate compact indicator panel: RSI, MA slope, ATR, volume."""
    if len(df) < 30:
        return None

    _ensure_dir()

    data = df[df.index <= T].copy()
    if len(data) < 14:
        return None
    plot_data = data.iloc[-50:].copy()
    x = list(range(len(plot_data)))

    fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)

    # Panel 1: Price with MAs
    ax = axes[0]
    ax.plot(x, plot_data["close"], color="blue", linewidth=0.8, label="Close")
    ma20 = data["close"].rolling(20).mean()
    ma50 = data["close"].rolling(50).mean()
    if len(ma20.dropna()) > 0:
        ax.plot(x, ma20.iloc[-len(plot_data):], color="orange", linewidth=0.8, label="MA20")
    if len(ma50.dropna()) > 0:
        ax.plot(x, ma50.iloc[-len(plot_data):], color="purple", linewidth=0.8, label="MA50")
    ax.axvline(x=len(plot_data)-1, color="red", linestyle="--", alpha=0.5)
    ax.set_ylabel("Price")
    ax.legend(fontsize=6, loc="upper left")
    ax.grid(True, alpha=0.2)

    # Panel 2: RSI
    ax = axes[1]
    rsi = compute_rsi(data["close"])
    ax.plot(x, rsi.iloc[-len(plot_data):], color="blue", linewidth=0.8)
    ax.axhline(y=70, color="red", linestyle="--", alpha=0.4)
    ax.axhline(y=30, color="green", linestyle="--", alpha=0.4)
    ax.fill_between(x, 30, 70, alpha=0.05, color="gray")
    ax.set_ylabel("RSI(14)")
    ax.grid(True, alpha=0.2)

    # Panel 3: ATR / Volatility
    ax = axes[2]
    atr = compute_atr(data)
    ax.plot(x, atr.iloc[-len(plot_data):], color="darkred", linewidth=0.8)
    ax.set_ylabel("ATR(14)")
    ax.grid(True, alpha=0.2)

    # Panel 4: Volume
    ax = axes[3]
    vol = plot_data["volume"]
    colors = ["green" if row["close"] >= row["open"] else "red"
              for _, row in plot_data.iterrows()]
    ax.bar(x, vol, color=colors, alpha=0.6, width=0.8)
    avg_v = vol.mean()
    ax.axhline(y=avg_v, color="orange", linestyle="--", alpha=0.5)
    ax.set_ylabel("Volume")
    ax.grid(True, alpha=0.2, axis="y")

    # Format x-axis
    tick_pos = list(range(0, len(plot_data), max(1, len(plot_data)//8)))
    tick_lbl = [plot_data.index[i].strftime("%H:%M") for i in tick_pos]
    for ax in axes:
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_lbl, rotation=45, fontsize=7)

    fig.suptitle(f"{config.INSTRUMENT_NAME} - Indicator Panel (Decision: {T})", fontsize=11)
    plt.tight_layout()

    fname = f"indicator_panel_{_safe_timestamp(T)}.png"
    path = config.CHARTS_DIR / fname
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)

    return str(path)


def plot_context_dashboard(
    df_5m: pd.DataFrame,
    df_daily: pd.DataFrame,
    df_weekly: pd.DataFrame,
    T: datetime,
    entry: Optional[float] = None,
    stop: Optional[float] = None,
    target: Optional[float] = None,
    save: bool = True,
) -> Optional[str]:
    """
    Generate combined context dashboard for human review.
    Shows all chart types in a single view.
    """
    _ensure_dir()

    fig = plt.figure(figsize=(16, 12))

    # 2x3 grid layout
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)

    # 1. Micro 5m (spans top row, 2 cols)
    ax1 = fig.add_subplot(gs[0, :2])
    micro = df_5m[df_5m.index <= T].iloc[-100:].copy()
    if len(micro) > 0:
        colors = ["green" if micro["close"].iloc[i] >= micro["open"].iloc[i] else "red"
                  for i in range(len(micro))]
        x = list(range(len(micro)))
        ax1.scatter(x, micro["close"], c=colors, s=8, alpha=0.8)
        ax1.plot(x, micro["close"], color="blue", linewidth=0.5, alpha=0.4)
        ax1.axvline(x=len(micro)-1, color="orange", linestyle="--", linewidth=2)
        ax1.set_title(f"Micro {config.INTRADAY_TIMEFRAME_LABEL} Context", fontsize=10)
        ax1.set_ylabel("Price (₹)")
        ax1.grid(True, alpha=0.2)
        tick_pos = list(range(0, len(micro), max(1, len(micro)//6)))
        tick_lbl = [micro.index[i].strftime("%H:%M") for i in tick_pos]
        ax1.set_xticks(tick_pos)
        ax1.set_xticklabels(tick_lbl, rotation=45, fontsize=6)

    # 2. Decision Zoom (spans mid row, 2 cols)
    ax2 = fig.add_subplot(gs[1, :2])
    zoom = df_5m[df_5m.index <= T].iloc[-30:].copy()
    if len(zoom) > 0:
        for i, (idx, row) in enumerate(zoom.iterrows()):
            color = "green" if row["close"] >= row["open"] else "red"
            body_bottom = min(row["open"], row["close"])
            body_height = abs(row["close"] - row["open"])
            ax2.bar(i, body_height, bottom=body_bottom, color=color, width=0.6, alpha=0.8)
            ax2.plot([i, i], [row["low"], row["high"]], color="black", linewidth=0.6)

        if entry:
            ax2.axhline(y=entry, color="blue", linestyle="-", alpha=0.6, label=f"Entry: {entry}")
        if stop:
            ax2.axhline(y=stop, color="red", linestyle=":", alpha=0.6, label=f"Stop: {stop}")
        if target:
            ax2.axhline(y=target, color="green", linestyle=":", alpha=0.6, label=f"Target: {target}")
        ax2.axvline(x=len(zoom)-1, color="orange", linestyle="--", linewidth=2)
        ax2.set_title("Decision Zoom", fontsize=10)
        ax2.set_ylabel("Price (₹)")
        handles, labels = ax2.get_legend_handles_labels()
        if handles:
            ax2.legend(fontsize=6)
        ax2.grid(True, alpha=0.2)
        tick_pos = list(range(0, len(zoom), max(1, len(zoom)//6)))
        tick_lbl = [zoom.index[i].strftime("%H:%M") for i in tick_pos]
        ax2.set_xticks(tick_pos)
        ax2.set_xticklabels(tick_lbl, rotation=45, fontsize=6)

    # 3. Volume
    ax3 = fig.add_subplot(gs[2, :2])
    vol = df_5m[df_5m.index <= T].iloc[-50:]
    if len(vol) > 0:
        colors_vol = ["green" if vol["close"].iloc[i] >= vol["open"].iloc[i] else "red"
                      for i in range(len(vol))]
        ax3.bar(range(len(vol)), vol["volume"], color=colors_vol, alpha=0.5, width=0.8)
        ax3.axhline(y=vol["volume"].mean(), color="orange", linestyle="--", alpha=0.5)
        ax3.set_title("Volume", fontsize=10)
        ax3.set_ylabel("Volume")
        ax3.grid(True, alpha=0.2, axis="y")
        tick_pos = list(range(0, len(vol), max(1, len(vol)//6)))
        tick_lbl = [vol.index[i].strftime("%H:%M") for i in tick_pos]
        ax3.set_xticks(tick_pos)
        ax3.set_xticklabels(tick_lbl, rotation=45, fontsize=6)

    # 4. Daily context (right column, top)
    ax4 = fig.add_subplot(gs[0, 2])
    daily = df_daily[df_daily.index <= T].copy()
    if len(daily) > 0:
        for i, (idx, row) in enumerate(daily.iterrows()):
            color = "green" if row["close"] >= row["open"] else "red"
            body_bottom = min(row["open"], row["close"])
            body_height = abs(row["close"] - row["open"])
            ax4.bar(i, body_height, bottom=body_bottom, color=color, width=0.6, alpha=0.8)
            ax4.plot([i, i], [row["low"], row["high"]], color="black", linewidth=0.6)
        ax4.set_title("Daily Context", fontsize=10)
        ax4.tick_params(axis="x", rotation=45, labelsize=6)
        if len(daily) > 1:
            tick_pos = list(range(0, len(daily), max(1, len(daily)//6)))
            tick_lbl = [daily.index[i].strftime("%m/%d") for i in tick_pos]
            ax4.set_xticks(tick_pos)
            ax4.set_xticklabels(tick_lbl, rotation=45, fontsize=6)
        ax4.grid(True, alpha=0.2)

    # 5. Weekly context (right column, middle)
    ax5 = fig.add_subplot(gs[1, 2])
    weekly = df_weekly[df_weekly.index <= T].copy()
    if len(weekly) > 0:
        for i, (idx, row) in enumerate(weekly.iterrows()):
            color = "green" if row["close"] >= row["open"] else "red"
            body_bottom = min(row["open"], row["close"])
            body_height = abs(row["close"] - row["open"])
            ax5.bar(i, body_height, bottom=body_bottom, color=color, width=0.6, alpha=0.8)
            ax5.plot([i, i], [row["low"], row["high"]], color="black", linewidth=0.6)
        ax5.set_title("Weekly Context", fontsize=10)
        ax5.tick_params(axis="x", rotation=45, labelsize=6)
        if len(weekly) > 1:
            tick_pos = list(range(0, len(weekly), max(1, len(weekly)//4)))
            tick_lbl = [weekly.index[i].strftime("%m/%d") for i in tick_pos]
            ax5.set_xticks(tick_pos)
            ax5.set_xticklabels(tick_lbl, rotation=45, fontsize=6)
        ax5.grid(True, alpha=0.2)

    # 6. Metadata panel (right column, bottom)
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.axis("off")
    metadata_lines = [
        f"Instrument: {config.INSTRUMENT_NAME}",
        f"Symbol: {config.SYMBOL}",
        f"Decision Time: {T}",
    ]
    if entry:
        metadata_lines.append(f"Entry: {entry}")
    if stop:
        metadata_lines.append(f"Stop: {stop}")
    if target:
        metadata_lines.append(f"Target: {target}")
    if entry and stop and target:
        risk = abs(entry - stop)
        reward = abs(target - entry)
        rr = reward / risk if risk > 0 else 0
        metadata_lines.append(f"R:R = {rr:.2f}")

    for i, line in enumerate(metadata_lines):
        ax6.text(0.1, 0.9 - i * 0.1, line, fontsize=9, family="monospace",
                 transform=ax6.transAxes)

    fig.suptitle(f"{config.INSTRUMENT_NAME} - Context Dashboard", fontsize=13, fontweight="bold")
    fname = f"dashboard_{_safe_timestamp(T)}.png"
    path = config.CHARTS_DIR / fname
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    return str(path)


def generate_all_charts(
    df_5m: pd.DataFrame,
    df_daily: pd.DataFrame,
    df_weekly: pd.DataFrame,
    T: datetime,
    entry: Optional[float] = None,
    stop: Optional[float] = None,
    target: Optional[float] = None,
) -> Dict[str, str]:
    """
    Generate all chart types for a decision point.
    Returns dict of chart_name -> file_path.
    """
    chart_paths = {}

    p = plot_micro_5m_chart(df_5m, T)
    if p:
        chart_paths["micro_5m_chart"] = p

    p = plot_decision_zoom_chart(df_5m, T, entry, stop, target)
    if p:
        chart_paths["decision_zoom_chart"] = p

    p = plot_volume_chart(df_5m, T)
    if p:
        chart_paths["volume_chart"] = p

    p = plot_daily_context_chart(df_daily, T)
    if p:
        chart_paths["daily_context_chart"] = p

    p = plot_weekly_context_chart(df_weekly, T)
    if p:
        chart_paths["weekly_context_chart"] = p

    p = plot_indicator_panel(df_5m, T)
    if p:
        chart_paths["indicator_panel"] = p

    p = plot_context_dashboard(df_5m, df_daily, df_weekly, T, entry, stop, target)
    if p:
        chart_paths["context_dashboard"] = p

    return chart_paths
