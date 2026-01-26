#!/usr/bin/env python3
"""
Send weekly portfolio update to Discord.
Shows current holdings, performance, and account status with benchmark comparison.
"""

import os
import sys
from datetime import date, timedelta
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


def get_positions_with_details(api) -> tuple:
    """Get current positions with market values and P&L.

    Returns:
        tuple: (equity_positions, crypto_positions)
    """
    positions = api.list_positions()

    equity_positions = []
    crypto_positions = []

    for p in positions:
        position_dict = {
            'symbol': p.symbol,
            'qty': float(p.qty),
            'market_value': float(p.market_value),
            'cost_basis': float(p.cost_basis),
            'unrealized_pl': float(p.unrealized_pl),
            'unrealized_plpc': float(p.unrealized_plpc),
            'current_price': float(p.current_price)
        }

        # Separate crypto (BTC, ETH, etc.) from equities
        if p.symbol in ['BTCUSD', 'ETHUSD', 'LTCUSD']:
            crypto_positions.append(position_dict)
        else:
            equity_positions.append(position_dict)

    return equity_positions, crypto_positions


def get_benchmark_prices(api) -> dict:
    """Get current prices for benchmark ETFs."""
    benchmarks = {'spy': 0, 'qqq': 0, 'vti': 0}
    for symbol in ['SPY', 'QQQ', 'VTI']:
        try:
            quote = api.get_latest_trade(symbol, feed='iex')  # Use IEX feed (free tier)
            benchmarks[symbol.lower()] = float(quote.price)
        except Exception as e:
            logger.warning(f"Could not fetch {symbol} price: {e}")
    return benchmarks


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
    equity_positions, crypto_positions = get_positions_with_details(api)

    logger.info(f"Account Equity: ${account_info['equity']:,.2f}")
    logger.info(f"Cash: ${account_info['cash']:,.2f}")
    logger.info(f"Equity Positions: {len(equity_positions)}")
    logger.info(f"Crypto Positions: {len(crypto_positions)}")

    # Calculate position breakdown
    equity_value = sum(p['market_value'] for p in equity_positions)
    crypto_value = sum(p['market_value'] for p in crypto_positions)
    total_positions_value = equity_value + crypto_value

    if total_positions_value > 0:
        logger.info(f"  Equities: ${equity_value:,.2f} ({equity_value/total_positions_value*100:.1f}%)")
        logger.info(f"  Crypto: ${crypto_value:,.2f} ({crypto_value/total_positions_value*100:.1f}%)")
        logger.info(f"  Cash: ${account_info['cash']:,.2f} ({account_info['cash']/account_info['equity']*100:.1f}%)")

    # Save current portfolio value
    storage.save_portfolio_value(account_info['equity'], account_info['cash'])

    # Get and save benchmark prices
    benchmark_prices = get_benchmark_prices(api)
    storage.save_benchmark_prices(
        spy=benchmark_prices['spy'],
        qqq=benchmark_prices['qqq'],
        vti=benchmark_prices['vti']
    )
    logger.info(f"Benchmarks: SPY=${benchmark_prices['spy']:.2f}, QQQ=${benchmark_prices['qqq']:.2f}, VTI=${benchmark_prices['vti']:.2f}")

    # Get performance metrics (now includes benchmark comparison)
    performance = storage.get_performance_metrics()

    # Get chart data for benchmark comparison
    benchmark_data = storage.get_benchmark_chart_data()

    # Get rebalance state
    state = storage.get_rebalance_state()
    days_since = state.get('days_since_rebalance', 0)
    days_until_rebalance = max(0, 20 - days_since)

    # Calculate next rebalance date
    next_rebalance = date.today()
    days_added = 0
    while days_added < days_until_rebalance:
        next_rebalance += timedelta(days=1)
        if next_rebalance.weekday() < 5:
            days_added += 1
    next_rebalance_str = next_rebalance.strftime('%Y-%m-%d')

    logger.info(f"\nPerformance:")
    if performance:
        logger.info(f"  Portfolio Return: {performance.get('total_return_pct', 0):+.2f}%")
        logger.info(f"  vs SPY: {performance.get('total_return_pct', 0) - performance.get('spy_return_pct', 0):+.2f}%")
        logger.info(f"  vs QQQ: {performance.get('total_return_pct', 0) - performance.get('qqq_return_pct', 0):+.2f}%")
        logger.info(f"  vs VTI: {performance.get('total_return_pct', 0) - performance.get('vti_return_pct', 0):+.2f}%")
        logger.info(f"  Max Drawdown: {performance.get('max_drawdown_pct', 0):.2f}%")
    logger.info(f"  Next rebalance: {next_rebalance_str} ({days_until_rebalance} days)")

    # Send enhanced Discord notification with chart
    # Combine positions for backward compatibility with existing Discord method
    all_positions = equity_positions + crypto_positions

    discord.send_portfolio_with_chart(
        account_info=account_info,
        positions=all_positions,
        performance=performance,
        benchmark_data=benchmark_data,
        crypto_positions=crypto_positions  # Pass separately for special display
    )

    # Also send a quick status about upcoming rebalance
    if days_until_rebalance <= 3:
        discord.send_alert(
            title="Rebalance Coming Soon",
            message=f"Portfolio will rebalance in **{days_until_rebalance}** trading days ({next_rebalance_str}).",
            level="info"
        )

    logger.info("\n" + "=" * 60)
    logger.info("Portfolio update sent to Discord!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
