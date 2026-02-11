#!/usr/bin/env python3
"""Send LC-Reversal backtest summary to Discord."""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.discord_prod import DiscordProductionNotifier


def fmt_pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to lc_reversal_backtest_*.json")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Backtest summary not found: {args.input}")

    with open(args.input, "r") as f:
        payload = json.load(f)

    metrics = payload.get("metrics", {})
    params = payload.get("params", {})

    message = "\n".join([
        "## LC-Reversal Weekly Backtest",
        "",
        f"- as_of_date: `{payload.get('as_of_date', 'N/A')}`",
        f"- lookback_days: `{params.get('days', 'N/A')}`",
        f"- universe_size: `{payload.get('universe_size', 'N/A')}`",
        f"- tx_cost_bps: `{params.get('tx_cost_bps', 'N/A')}`",
        f"- impact_slippage_mult: `{params.get('impact_slippage_mult', 'N/A')}`",
        "",
        "### Metrics",
        f"- CAGR: `{fmt_pct(float(metrics.get('cagr', 0.0) or 0.0))}`",
        f"- Sharpe: `{float(metrics.get('sharpe', 0.0) or 0.0):.3f}`",
        f"- Max Drawdown: `{fmt_pct(float(metrics.get('max_drawdown', 0.0) or 0.0))}`",
        f"- Turnover: `{float(metrics.get('turnover', 0.0) or 0.0):.3f}`",
        f"- Hit Rate: `{fmt_pct(float(metrics.get('hit_rate', 0.0) or 0.0))}`",
        f"- Total Return: `{fmt_pct(float(metrics.get('total_return', 0.0) or 0.0))}`",
    ])

    notifier = DiscordProductionNotifier()
    ok = notifier.send_markdown(message)
    if not ok:
        raise RuntimeError("Failed to send LC-Reversal backtest Discord notification")


if __name__ == "__main__":
    main()
