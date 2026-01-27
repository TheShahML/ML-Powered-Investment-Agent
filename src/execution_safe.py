"""Safe, idempotent order execution with multiple safety controls."""
import os
import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np
from loguru import logger
from typing import Dict, List, Tuple


def check_market_open(api: tradeapi.REST) -> bool:
    """Check if market is currently open."""
    try:
        clock = api.get_clock()
        is_open = clock.is_open

        if not is_open:
            logger.warning("Market is CLOSED - cannot place orders")
        else:
            logger.info("Market is OPEN")

        return is_open
    except Exception as e:
        logger.error(f"Error checking market status: {e}")
        return False


def check_kill_switch() -> Tuple[bool, str]:
    """
    Check kill switch status from environment variable.

    Returns:
        (enabled: bool, reason: str)
    """
    enabled_str = os.environ.get('KILL_SWITCH_ENABLED', 'false').lower()
    enabled = enabled_str in ['true', '1', 'yes']

    reason = os.environ.get('KILL_SWITCH_REASON', 'Manual override')

    if enabled:
        logger.warning(f"KILL SWITCH ENABLED: {reason}")
    else:
        logger.info("Kill switch: Disabled")

    return enabled, reason


def compute_target_weights_inverse_vol(
    signals: pd.DataFrame,
    data: pd.DataFrame,
    top_n: int,
    max_weight: float,
    as_of_date,
    vol_window: int = 60
) -> Dict[str, float]:
    """
    Compute inverse-volatility weights for top N stocks.

    Args:
        signals: DataFrame with 'score' and 'rank' columns, indexed by symbol
        data: Historical data (MultiIndex: timestamp, symbol)
        top_n: Number of stocks to select
        max_weight: Maximum weight per position
        as_of_date: As-of date for data
        vol_window: Lookback window for volatility

    Returns:
        Dict of {symbol: weight}
    """
    top_stocks = signals.head(top_n).index.tolist()

    # Compute realized volatility for each
    vols = {}
    for symbol in top_stocks:
        try:
            symbol_data = data.xs(symbol, level=1)
            symbol_data = symbol_data[symbol_data.index <= as_of_date]

            if len(symbol_data) < vol_window:
                vols[symbol] = 0.20  # Default 20% annualized
                continue

            returns = symbol_data['close'].pct_change()
            vol = returns.tail(vol_window).std() * np.sqrt(252)
            vols[symbol] = max(vol, 0.01)  # Floor at 1%

        except Exception as e:
            logger.warning(f"Could not compute vol for {symbol}: {e}")
            vols[symbol] = 0.20

    # Inverse vol weights
    inv_vols = {s: 1.0 / v for s, v in vols.items()}
    total_inv_vol = sum(inv_vols.values())

    weights = {s: iv / total_inv_vol for s, iv in inv_vols.items()}

    # Cap at max_weight
    weights = {s: min(w, max_weight) for s, w in weights.items()}

    # Renormalize
    total = sum(weights.values())
    if total > 0:
        weights = {s: w / total for s, w in weights.items()}

    logger.info(f"Target weights computed for {len(weights)} stocks (inverse-vol)")
    return weights


def execute_orders_safe(
    api: tradeapi.REST,
    target_weights: Dict[str, float],
    current_positions: Dict[str, float],
    portfolio_value: float,
    config: Dict,
    dry_run: bool = False
) -> List[Dict]:
    """
    Execute orders safely with caps and checks.

    Args:
        api: Alpaca REST client
        target_weights: {symbol: weight} target
        current_positions: {symbol: qty} current
        portfolio_value: Total portfolio equity
        config: Config with execution limits
        dry_run: If True, don't actually submit orders

    Returns:
        List of order dicts
    """
    max_orders = config.get('max_orders_per_rebalance', 100)
    max_notional = config.get('max_daily_notional', 1_000_000)
    min_notional = config.get('min_trade_notional', 10.0)
    turnover_buffer = config.get('turnover_buffer_pct', 1.0) / 100.0

    # Get current prices
    prices = {}
    for symbol in set(list(target_weights.keys()) + list(current_positions.keys())):
        try:
            quote = api.get_latest_trade(symbol, feed='iex')
            prices[symbol] = float(quote.price)
        except Exception as e:
            logger.warning(f"Could not get price for {symbol}: {e}")

    # Compute target dollar amounts
    target_dollars = {s: w * portfolio_value for s, w in target_weights.items()}

    # Compute current dollar amounts
    current_dollars = {
        s: current_positions.get(s, 0) * prices.get(s, 0)
        for s in set(list(current_positions.keys()) + list(target_dollars.keys()))
    }

    # Compute trades
    trades = []
    total_notional = 0.0

    for symbol in set(list(current_dollars.keys()) + list(target_dollars.keys())):
        current = current_dollars.get(symbol, 0)
        target = target_dollars.get(symbol, 0)
        diff = target - current

        # Check turnover buffer
        if abs(diff / portfolio_value) < turnover_buffer:
            continue

        # Check min notional
        if abs(diff) < min_notional:
            continue

        price = prices.get(symbol)
        if not price or price <= 0:
            continue

        # Compute shares
        shares = diff / price
        side = 'buy' if shares > 0 else 'sell'
        qty = abs(shares)

        trades.append({
            'symbol': symbol,
            'side': side,
            'qty': round(qty, 4),
            'notional': abs(diff),
            'price': price
        })

        total_notional += abs(diff)

    # Check max_orders cap
    if len(trades) > max_orders:
        logger.warning(f"Too many trades ({len(trades)}), capping at {max_orders}")
        trades = sorted(trades, key=lambda x: x['notional'], reverse=True)[:max_orders]
        total_notional = sum(t['notional'] for t in trades)

    # Check max_notional cap
    if total_notional > max_notional:
        logger.warning(f"Total notional ${total_notional:.0f} exceeds cap ${max_notional:.0f}")
        scale = max_notional / total_notional
        for trade in trades:
            trade['qty'] *= scale
            trade['notional'] *= scale

    logger.info(f"Executing {len(trades)} orders, total notional: ${total_notional:,.2f}")

    # Execute
    orders = []
    for trade in trades:
        order_info = trade.copy()

        if dry_run:
            logger.info(f"[DRY RUN] Would {trade['side']} {trade['qty']:.4f} {trade['symbol']} @ ${trade['price']:.2f}")
            order_info['status'] = 'dry_run'
        else:
            try:
                order = api.submit_order(
                    symbol=trade['symbol'],
                    qty=trade['qty'],
                    side=trade['side'],
                    type='market',
                    time_in_force='day'
                )
                order_info['status'] = 'submitted'
                order_info['order_id'] = order.id
                logger.info(f"Submitted: {trade['side']} {trade['qty']:.4f} {trade['symbol']}")

            except Exception as e:
                order_info['status'] = 'failed'
                order_info['error'] = str(e)
                logger.error(f"Failed to submit order for {trade['symbol']}: {e}")

        orders.append(order_info)

    return orders


def get_current_positions(api: tradeapi.REST) -> Dict[str, float]:
    """Get current positions as {symbol: qty}."""
    try:
        positions = api.list_positions()
        return {p.symbol: float(p.qty) for p in positions}
    except Exception as e:
        logger.error(f"Error getting positions: {e}")
        return {}


def get_account_equity(api: tradeapi.REST) -> float:
    """Get account equity."""
    try:
        account = api.get_account()
        return float(account.equity)
    except Exception as e:
        logger.error(f"Error getting account equity: {e}")
        return 0.0
