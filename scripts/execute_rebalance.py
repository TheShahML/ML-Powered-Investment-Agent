#!/usr/bin/env python3
"""
Rebalance Execution Script for GitHub Actions.
Executes trades via Alpaca API based on latest signals.
Runs every 20 trading days (monthly rebalancing).
"""

import os
import sys
import argparse
from datetime import date, datetime, timedelta
from loguru import logger
import alpaca_trade_api as tradeapi

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.file_storage import FileStorage
from src.config import load_config
from src.discord_notifier import DiscordNotifier
from src.crypto_strategy import BitcoinMomentumStrategy, get_crypto_orders
from src.data_service import DataService


def get_alpaca_api(config):
    """Initialize Alpaca API client."""
    return tradeapi.REST(
        config['ALPACA_API_KEY'],
        config['ALPACA_SECRET_KEY'],
        config['ALPACA_BASE_URL']
    )


def get_current_positions(api) -> dict:
    """Get current portfolio positions as {symbol: qty}."""
    positions = api.list_positions()
    return {p.symbol: float(p.qty) for p in positions}


def get_account_info(api) -> dict:
    """Get account equity and cash."""
    account = api.get_account()
    return {
        'equity': float(account.equity),
        'cash': float(account.cash),
        'buying_power': float(account.buying_power)
    }


def calculate_target_positions(signals, account_info, config, btc_allocation=0.0) -> dict:
    """
    Calculate target positions based on signals and account value.

    Args:
        btc_allocation: Percentage of equity reserved for Bitcoin (0-0.15)

    Returns: {symbol: target_qty}
    """
    n_holdings = config.get('n_holdings', 15)
    max_weight = config.get('max_position_weight', 0.12)

    # Reserve Bitcoin allocation from equity
    equity = account_info['equity']
    equity_for_stocks = equity * (1.0 - btc_allocation)

    target_weight = min(1.0 / n_holdings, max_weight)

    # Get top N stocks from signals
    top_stocks = signals.head(n_holdings).index.tolist()

    target_positions = {}
    for symbol in top_stocks:
        # Get current price
        try:
            api = tradeapi.REST(
                os.environ['ALPACA_API_KEY'],
                os.environ['ALPACA_SECRET_KEY'],
                os.environ['ALPACA_BASE_URL']
            )
            quote = api.get_latest_trade(symbol, feed='iex')  # Use IEX feed (free tier)
            price = float(quote.price)

            target_value = equity_for_stocks * target_weight
            target_qty = target_value / price

            # Round to reasonable precision (Alpaca supports fractional)
            target_qty = round(target_qty, 4)

            if target_qty > 0:
                target_positions[symbol] = target_qty

        except Exception as e:
            logger.warning(f"Could not get price for {symbol}: {e}")
            continue

    return target_positions


def execute_rebalance(api, current_positions, target_positions, dry_run=False) -> list:
    """
    Execute the rebalance by placing orders.

    Returns: List of orders placed
    """
    orders = []

    all_symbols = set(current_positions.keys()) | set(target_positions.keys())

    for symbol in all_symbols:
        current_qty = current_positions.get(symbol, 0)
        target_qty = target_positions.get(symbol, 0)
        diff = target_qty - current_qty

        # Skip tiny trades (less than $10 notional)
        try:
            quote = api.get_latest_trade(symbol, feed='iex')  # Use IEX feed (free tier)
            price = float(quote.price)
            notional = abs(diff * price)

            if notional < 10:
                continue

        except Exception as e:
            logger.warning(f"Skipping {symbol}: {e}")
            continue

        if abs(diff) < 0.01:
            continue

        side = 'buy' if diff > 0 else 'sell'
        qty = abs(diff)

        order_info = {
            'symbol': symbol,
            'side': side,
            'qty': round(qty, 4),
            'notional': round(notional, 2),
            'type': 'market',
            'time_in_force': 'day'
        }

        if dry_run:
            logger.info(f"[DRY RUN] Would {side} {qty:.4f} shares of {symbol} (${notional:.2f})")
            order_info['status'] = 'dry_run'
        else:
            try:
                order = api.submit_order(
                    symbol=symbol,
                    qty=qty,
                    side=side,
                    type='market',
                    time_in_force='day'
                )
                order_info['status'] = 'submitted'
                order_info['order_id'] = order.id
                logger.info(f"Submitted {side} order for {qty:.4f} shares of {symbol}")

            except Exception as e:
                order_info['status'] = 'failed'
                order_info['error'] = str(e)
                logger.error(f"Failed to submit order for {symbol}: {e}")

        orders.append(order_info)

    return orders


def main():
    parser = argparse.ArgumentParser(description='Execute portfolio rebalance')
    parser.add_argument('--dry-run', action='store_true', help='Run without placing orders')
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("MONTHLY REBALANCE EXECUTION")
    logger.info(f"Date: {date.today()}")
    logger.info(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE'}")
    logger.info("=" * 60)

    # Load config and initialize
    config = load_config()
    storage = FileStorage()
    api = get_alpaca_api(config)

    # Check broker mode safety
    broker_mode = os.environ.get('BROKER_MODE', 'paper')
    if broker_mode == 'live' and not args.dry_run:
        ack = os.environ.get('I_ACKNOWLEDGE_LIVE_TRADING', 'false')
        if ack.lower() != 'true':
            logger.error("BROKER_MODE is 'live' but I_ACKNOWLEDGE_LIVE_TRADING is not set!")
            logger.error("Set this secret to 'true' to enable live trading.")
            sys.exit(1)

    # Get latest signals
    signals = storage.get_latest_signals()
    if signals is None:
        logger.error("No signals found! Run generate_daily_signals.py first.")
        sys.exit(1)

    logger.info(f"Loaded signals with {len(signals)} stocks (using 20-day model)")

    # Get current state
    account_info = get_account_info(api)
    current_positions = get_current_positions(api)

    logger.info(f"\nAccount Info:")
    logger.info(f"  Equity: ${account_info['equity']:,.2f}")
    logger.info(f"  Cash: ${account_info['cash']:,.2f}")
    logger.info(f"  Current positions: {len(current_positions)}")

    # === BITCOIN MOMENTUM ALLOCATION ===
    btc_allocation = 0.0
    btc_orders = []

    if config.get('trade_crypto', False):
        logger.info("\n" + "=" * 60)
        logger.info("BITCOIN MOMENTUM ALLOCATION")
        logger.info("=" * 60)

        try:
            # Initialize Bitcoin momentum strategy
            btc_strategy = BitcoinMomentumStrategy(config)
            data_service = DataService(config)

            # Fetch Bitcoin price history (need 200 days for SMA)
            end_date = date.today()
            start_date = end_date - timedelta(days=365)  # 1 year to ensure 200 trading days

            logger.info("Fetching Bitcoin price history for momentum calculation...")
            btc_df = data_service.get_historical_data(
                ['BTC/USD'],
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            )

            if not btc_df.empty:
                # Extract BTC prices
                btc_prices = btc_df['close']

                # Calculate dynamic allocation (5-15% based on momentum)
                btc_allocation = btc_strategy.get_dynamic_allocation(btc_prices)

                logger.info(f"Dynamic Bitcoin allocation: {btc_allocation*100:.1f}% of portfolio")

                # Calculate target Bitcoin value
                target_btc_value = account_info['equity'] * btc_allocation

                # Execute Bitcoin orders
                btc_orders = get_crypto_orders(
                    api,
                    target_allocations={'BTC/USD': target_btc_value},
                    dry_run=args.dry_run
                )

                logger.info(f"Bitcoin target value: ${target_btc_value:,.2f}")
                if btc_orders:
                    for order in btc_orders:
                        if order['status'] in ['submitted', 'dry_run']:
                            logger.info(f"  Bitcoin {order['side']}: ${order['notional']:,.2f}")
            else:
                logger.warning("Could not fetch Bitcoin data, skipping crypto allocation")

        except Exception as e:
            logger.error(f"Error processing Bitcoin allocation: {e}")
            logger.warning("Continuing with equities only")

    # Calculate equity target positions (reserving Bitcoin allocation)
    logger.info(f"\nEquity allocation: {(1.0 - btc_allocation)*100:.1f}% of portfolio")
    target_positions = calculate_target_positions(signals, account_info, config, btc_allocation=btc_allocation)
    logger.info(f"  Target equity positions: {len(target_positions)}")

    # Execute rebalance (equities + Bitcoin)
    logger.info("\n" + "=" * 60)
    logger.info("EXECUTING REBALANCE")
    logger.info("=" * 60)

    # Execute equity orders
    logger.info("\nEquity orders...")
    equity_orders = execute_rebalance(api, current_positions, target_positions, dry_run=args.dry_run)

    # Combine all orders
    orders = equity_orders + btc_orders

    # Log results
    storage.log_orders(orders)

    # Save portfolio state
    if not args.dry_run:
        # Get updated positions
        new_positions = get_current_positions(api)
        positions_list = [{'symbol': s, 'qty': q} for s, q in new_positions.items()]
        storage.save_positions(positions_list)

        # Save portfolio value
        new_account = get_account_info(api)
        storage.save_portfolio_value(new_account['equity'], new_account['cash'])

        # Reset rebalance counter
        storage.reset_day_counter()

    # Summary
    equity_buy = [o for o in equity_orders if o['side'] == 'buy']
    equity_sell = [o for o in equity_orders if o['side'] == 'sell']
    btc_buy = [o for o in btc_orders if o.get('side') == 'buy']
    btc_sell = [o for o in btc_orders if o.get('side') == 'sell']

    logger.info("\n" + "=" * 60)
    logger.info("REBALANCE SUMMARY")
    logger.info(f"  Equity: {len(equity_buy)} buys, {len(equity_sell)} sells")
    logger.info(f"  Bitcoin: {len(btc_buy)} buys, {len(btc_sell)} sells")
    logger.info(f"  Total orders: {len(orders)}")

    performance = None
    if not args.dry_run:
        performance = storage.get_performance_metrics()
        if performance:
            logger.info(f"\nPerformance:")
            logger.info(f"  Total return: {performance.get('total_return_pct', 0):.2f}%")
            logger.info(f"  Max drawdown: {performance.get('max_drawdown_pct', 0):.2f}%")

    # Send Discord notification
    discord = DiscordNotifier()
    discord.send_rebalance_notification(
        orders=orders,
        account_info=account_info,
        performance=performance,
        dry_run=args.dry_run
    )

    logger.info("=" * 60)


if __name__ == "__main__":
    main()
