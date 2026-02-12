#!/usr/bin/env python3
"""Lightweight LC-Reversal backtest using daily bars with no lookahead."""

import os
import sys
import argparse
import json
from datetime import timedelta

from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config
from src.trading_calendar import TradingCalendar
from src.universe import Universe
from src.data_service import DataService
from src.strategies.lc_reversal import LCReversalStrategy
from src import alpaca_fix
import alpaca_trade_api as tradeapi


def _write_run_log(payload: dict, as_of_date: str) -> str:
    logs_dir = os.path.join("reports", "run_logs")
    os.makedirs(logs_dir, exist_ok=True)
    stamp = as_of_date.replace("-", "")
    filename = f"backtest_lc_reversal_{stamp}.json"
    path = os.path.join(logs_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    latest_path = os.path.join(logs_dir, "latest_backtest_lc_reversal.json")
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    logger.info(f"Saved LC-Reversal run log: {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=365, help="Backtest window length in calendar days")
    parser.add_argument("--tx-cost-bps", type=float, default=20.0)
    parser.add_argument("--impact-slippage-mult", type=float, default=0.0)
    parser.add_argument("--initial-capital", type=float, default=100000.0)
    parser.add_argument("--output", type=str, default="reports/lc_reversal_backtest_latest.json")
    args = parser.parse_args()

    config = load_config()
    api = tradeapi.REST(config["ALPACA_API_KEY"], config["ALPACA_SECRET_KEY"], config["ALPACA_BASE_URL"])

    calendar = TradingCalendar(api)
    as_of_date = calendar.get_last_completed_trading_day()
    start_date = as_of_date - timedelta(days=max(args.days, 90))

    universe_builder = Universe(api, config)
    symbols, _ = universe_builder.build_universe()

    logger.info(f"Backtest universe size: {len(symbols)}")
    data_service = DataService(config)
    df = data_service.get_historical_data(
        symbols,
        start_date.strftime("%Y-%m-%d"),
        as_of_date.strftime("%Y-%m-%d"),
        ensure_spy=True,
    )
    if df.empty:
        raise RuntimeError("No historical data for LC-Reversal backtest")

    strategy = LCReversalStrategy(config)
    df_features = strategy.compute_features(df)
    results = strategy.run_backtest(
        df_features=df_features,
        start_date=start_date,
        end_date=as_of_date,
        initial_capital=args.initial_capital,
        tx_cost_bps=args.tx_cost_bps,
        impact_slippage_mult=args.impact_slippage_mult,
    )

    if not results:
        raise RuntimeError("LC-Reversal backtest produced no results")

    logger.info("=" * 60)
    logger.info("LC-REVERSAL BACKTEST RESULTS")
    logger.info("=" * 60)
    logger.info(f"CAGR: {results.get('cagr', 0.0) * 100:.2f}%")
    logger.info(f"Sharpe: {results.get('sharpe', 0.0):.3f}")
    logger.info(f"Max Drawdown: {results.get('max_drawdown', 0.0) * 100:.2f}%")
    logger.info(f"Turnover: {results.get('turnover', 0.0):.3f}")
    logger.info(f"Hit Rate: {results.get('hit_rate', 0.0) * 100:.2f}%")
    logger.info(f"Total Return: {results.get('total_return', 0.0) * 100:.2f}%")

    output_payload = {
        "strategy": "lc_reversal",
        "as_of_date": as_of_date.isoformat(),
        "start_date": start_date.isoformat(),
        "universe_size": len(symbols),
        "params": {
            "days": args.days,
            "tx_cost_bps": args.tx_cost_bps,
            "impact_slippage_mult": args.impact_slippage_mult,
            "initial_capital": args.initial_capital,
        },
        "metrics": {
            "cagr": results.get("cagr", 0.0),
            "sharpe": results.get("sharpe", 0.0),
            "max_drawdown": results.get("max_drawdown", 0.0),
            "turnover": results.get("turnover", 0.0),
            "hit_rate": results.get("hit_rate", 0.0),
            "total_return": results.get("total_return", 0.0),
        },
    }

    output_path = args.output
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_payload, f, indent=2)
    logger.info(f"Saved LC-Reversal backtest summary: {output_path}")
    _write_run_log(output_payload, as_of_date=as_of_date.isoformat())


if __name__ == "__main__":
    main()
