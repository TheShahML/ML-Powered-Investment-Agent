#!/usr/bin/env python3
"""
Send weekly portfolio update to Discord.
Shows current holdings, performance, and account status.
"""

import os
import sys
from datetime import date
from loguru import logger
import alpaca_trade_api as tradeapi

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.file_storage import FileStorage
from src.config import load_config
from src.discord_notifier import DiscordNotifier


def get_alpaca_api(config):
    """Initialize Alpaca API client."""
    return tradeapi.REST(
        config['ALPACA_API_KEY'],
        config['ALPACA_SECRET_KEY'],
        config['ALPACA_BASE_URL']
    )


def get_account_info(api) -> dict:
    """Get account equity and cash."""
    account = api.get_account()
    return {
        'equity': float(account.equity),
        'cash': float(account.cash),
        'buying_power': float(account.buying_power)
    }


def get_positions_with_details(api) -> list:
    """Get current positions with market values and P&L."""
    positions = api.list_positions()
    return [
        {
            'symbol': p.symbol,
            'qty': float(p.qty),
            'market_value': float(p.market_value),
            'cost_basis': float(p.cost_basis),
            'unrealized_pl': float(p.unrealized_pl),
            'unrealized_plpc': float(p.unrealized_plpc),
            'current_price': float(p.current_price)
        }
        for p in positions
    ]


def main():
    logger.info("=" * 60)
    logger.info("WEEKLY PORTFOLIO UPDATE")
    logger.info(f"Date: {date.today()}")
    logger.info("=" * 60)

    # Load config and initialize
    config = load_config()
    storage = FileStorage()
    api = get_alpaca_api(config)
    discord = DiscordNotifier()

    # Get current account info
    account_info = get_account_info(api)
    positions = get_positions_with_details(api)

    logger.info(f"Account Equity: ${account_info['equity']:,.2f}")
    logger.info(f"Cash: ${account_info['cash']:,.2f}")
    logger.info(f"Positions: {len(positions)}")

    # Save current portfolio value
    storage.save_portfolio_value(account_info['equity'], account_info['cash'])

    # Get performance metrics
    performance = storage.get_performance_metrics()

    # Get rebalance state
    state = storage.get_rebalance_state()
    days_until_rebalance = 20 - state.get('days_since_rebalance', 0)

    logger.info(f"\nPerformance:")
    if performance:
        logger.info(f"  Total Return: {performance.get('total_return_pct', 0):.2f}%")
        logger.info(f"  Max Drawdown: {performance.get('max_drawdown_pct', 0):.2f}%")
    logger.info(f"  Days until next rebalance: {days_until_rebalance}")

    # Send Discord notification
    discord.send_portfolio_update(
        account_info=account_info,
        positions=positions,
        performance=performance
    )

    # Also send a quick status about upcoming rebalance
    if days_until_rebalance <= 3:
        discord.send_alert(
            title="Rebalance Coming Soon",
            message=f"Portfolio will rebalance in **{days_until_rebalance}** trading days.",
            level="info"
        )

    logger.info("\n" + "=" * 60)
    logger.info("Portfolio update sent to Discord!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
