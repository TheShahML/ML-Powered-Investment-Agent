#!/usr/bin/env python3
"""Safe monthly rebalance execution with all safety controls."""
import os
import sys
import argparse
from datetime import date
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config
from src.trading_calendar import TradingCalendar
from src.state_manager import StateManager
from src.file_storage import FileStorage
from src.execution_safe import (
    check_market_open,
    check_kill_switch,
    compute_target_weights_inverse_vol,
    execute_orders_safe,
    get_current_positions,
    get_account_equity
)
from src.discord_prod import DiscordProductionNotifier
import alpaca_trade_api as tradeapi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("MONTHLY REBALANCE CHECK")
    logger.info(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE'}")
    logger.info("=" * 60)

    # Load config and API
    config = load_config()
    api = tradeapi.REST(
        config['ALPACA_API_KEY'],
        config['ALPACA_SECRET_KEY'],
        config['ALPACA_BASE_URL']
    )

    # Get as-of date
    calendar = TradingCalendar(api)
    as_of_date = calendar.get_last_completed_trading_day()
    logger.info(f"As-of date: {as_of_date}")

    # Load state
    state_manager = StateManager(state_file_path="state-repo/latest_state.json")
    state = state_manager.load_state()

    # Check if rebalance due
    if not state_manager.check_rebalance_due(threshold=20):
        days_until = state.get('rebalance', {}).get('days_until_rebalance', 20)
        logger.info(f"Rebalance NOT due ({days_until} days remaining)")
        sys.exit(0)

    logger.info("Rebalance IS DUE - proceeding with checks")

    # Check idempotency
    if state_manager.check_already_rebalanced(as_of_date):
        logger.warning("Already rebalanced today - exiting")
        sys.exit(0)

    # Check market open
    if not check_market_open(api):
        logger.error("Market is CLOSED - cannot rebalance")
        discord = DiscordProductionNotifier()
        discord.send_monthly_rebalance_execution(
            broker_mode=os.environ.get('BROKER_MODE', 'paper'),
            as_of_date=as_of_date.isoformat(),
            model_promoted=None,
            equity_allocation=0,
            btc_allocation=0,
            portfolio_value=0,
            orders=[],
            kill_switch=False,
            market_closed=True,
            dry_run=args.dry_run
        )
        sys.exit(1)

    # Check kill switch
    kill_switch_enabled, kill_reason = check_kill_switch()
    if kill_switch_enabled:
        logger.error(f"KILL SWITCH ENABLED: {kill_reason}")
        discord = DiscordProductionNotifier()
        discord.send_monthly_rebalance_execution(
            broker_mode=os.environ.get('BROKER_MODE', 'paper'),
            as_of_date=as_of_date.isoformat(),
            model_promoted=None,
            equity_allocation=0,
            btc_allocation=0,
            portfolio_value=0,
            orders=[],
            kill_switch=True,
            market_closed=False,
            dry_run=args.dry_run
        )
        sys.exit(1)

    # Get account info
    portfolio_value = get_account_equity(api)
    current_positions = get_current_positions(api)

    logger.info(f"Portfolio value: ${portfolio_value:,.0f}")
    logger.info(f"Current positions: {len(current_positions)}")

    # Load latest signals
    storage = FileStorage()
    signals = storage.get_latest_signals()

    if signals is None or signals.empty:
        logger.error("No signals available!")
        sys.exit(1)

    # Compute target weights (stub - would load data for inverse-vol)
    # For now, use simple equal weight
    top_n = 25  # Increased from 20 to 25 for better diversification
    top_stocks = signals.head(top_n).index.tolist()
    target_weights = {s: 1.0 / top_n for s in top_stocks}

    logger.info(f"Target: {len(target_weights)} positions")

    # Execute orders
    exec_config = {
        'max_orders_per_rebalance': 100,
        'max_daily_notional': 1_000_000,
        'min_trade_notional': 10,
        'turnover_buffer_pct': 1.0
    }

    orders = execute_orders_safe(
        api,
        target_weights,
        current_positions,
        portfolio_value,
        exec_config,
        dry_run=args.dry_run
    )

    logger.info(f"Executed {len(orders)} orders")

    # Promote candidate if approved
    model_promoted = None
    candidate = state.get('models', {}).get('candidate_model', {})
    if candidate.get('approved_for_next_rebalance'):
        logger.info(f"Promoting candidate: {candidate.get('version')}")
        state_manager.promote_candidate_to_active()
        model_promoted = candidate.get('version')

    # Update rebalance schedule
    if not args.dry_run:
        state_manager.update_rebalance_schedule(as_of_date, rebalance_freq=20)

    # Send Discord notification
    discord = DiscordProductionNotifier()
    discord.send_monthly_rebalance_execution(
        broker_mode=os.environ.get('BROKER_MODE', 'paper'),
        as_of_date=as_of_date.isoformat(),
        model_promoted=model_promoted,
        equity_allocation=1.0,  # Stub
        btc_allocation=0.0,
        portfolio_value=portfolio_value,
        orders=orders,
        kill_switch=False,
        market_closed=False,
        dry_run=args.dry_run
    )

    logger.info("=" * 60)
    logger.info("REBALANCE COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
