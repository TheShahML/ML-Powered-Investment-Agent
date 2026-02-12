"""Safe, idempotent order execution with multiple safety controls."""
import os
import re
# Fix alpaca import issue
from . import alpaca_fix
import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np
from loguru import logger
from typing import Dict, List, Tuple, Optional


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
    dry_run: bool = False,
    open_orders_by_symbol: Optional[Dict[str, List[Dict]]] = None,
    client_order_prefix: Optional[str] = None
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
    order_type = str(config.get('order_type', 'market') or 'market').lower()
    limit_offset_bps = float(config.get('limit_offset_bps', 10.0) or 10.0)
    if order_type not in {'market', 'limit'}:
        logger.warning(f"Invalid order_type={order_type}; defaulting to market")
        order_type = 'market'

    open_orders_by_symbol = open_orders_by_symbol or {}

    def _latest_price(symbol: str) -> Optional[float]:
        # Primary path: equities trade feed
        try:
            quote = api.get_latest_trade(symbol, feed='iex')
            return float(quote.price)
        except Exception:
            pass

        # Fallback path: crypto pairs (e.g., BTCUSD -> BTC/USD)
        crypto_symbol = None
        if '/' in symbol:
            crypto_symbol = symbol
        elif symbol.endswith('USD') and symbol.isalpha() and len(symbol) > 3:
            crypto_symbol = f"{symbol[:-3]}/USD"

        if crypto_symbol:
            try:
                q = api.get_latest_crypto_quote(crypto_symbol)
                ask = float(getattr(q, 'ap', 0.0) or 0.0)
                bid = float(getattr(q, 'bp', 0.0) or 0.0)
                px = ask if ask > 0 else bid
                if px > 0:
                    return px
            except Exception:
                pass
        return None

    # Get current prices
    prices = {}
    for symbol in set(list(target_weights.keys()) + list(current_positions.keys())):
        px = _latest_price(symbol)
        if px is not None and px > 0:
            prices[symbol] = float(px)
        else:
            logger.warning(f"Could not get price for {symbol}")

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

        trade_info = {
            'symbol': symbol,
            'side': side,
            'qty': round(qty, 4),
            'notional': abs(diff),
            'price': price
        }

        if open_orders_by_symbol.get(symbol):
            trade_info['status'] = 'skipped_existing_open_order'
            trade_info['error'] = f"{len(open_orders_by_symbol[symbol])} open order(s) already exist for {symbol}"
            trades.append(trade_info)
            continue

        trades.append(trade_info)

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

    logger.info(f"Computed {len(trades)} target trade(s), executable notional: ${total_notional:,.2f}")

    # Execute
    orders = []
    for trade in trades:
        order_info = trade.copy()
        if order_info.get('status') == 'skipped_existing_open_order':
            logger.warning(
                f"Skipping {trade['symbol']} due to existing open orders: "
                f"{order_info.get('error', 'unknown')}"
            )
            orders.append(order_info)
            continue

        if dry_run:
            logger.info(f"[DRY RUN] Would {trade['side']} {trade['qty']:.4f} {trade['symbol']} @ ${trade['price']:.2f}")
            order_info['status'] = 'dry_run'
        else:
            try:
                client_order_id = None
                if client_order_prefix:
                    notional_cents = int(round(float(trade['notional']) * 100))
                    raw = f"{client_order_prefix}-{trade['symbol']}-{trade['side']}-{notional_cents}"
                    raw = re.sub(r'[^A-Za-z0-9_-]', '', raw)
                    client_order_id = raw[:48]

                order = api.submit_order(
                    symbol=trade['symbol'],
                    qty=trade['qty'],
                    side=trade['side'],
                    type=order_type,
                    time_in_force='day',
                    client_order_id=client_order_id,
                    **(
                        {
                            'limit_price': round(
                                trade['price'] * (1.0 + (limit_offset_bps / 10000.0))
                                if trade['side'] == 'buy'
                                else trade['price'] * (1.0 - (limit_offset_bps / 10000.0)),
                                2
                            )
                        }
                        if order_type == 'limit' else {}
                    )
                )
                order_id = getattr(order, 'id', None)
                order_status = getattr(order, 'status', 'unknown')
                if not order_id:
                    raise RuntimeError(f"Order submitted but missing order id (status={order_status})")

                order_info['status'] = 'submitted'
                order_info['order_id'] = order_id
                order_info['alpaca_status'] = order_status
                if client_order_id:
                    order_info['client_order_id'] = client_order_id
                if order_type == 'limit':
                    order_info['limit_price'] = round(
                        trade['price'] * (1.0 + (limit_offset_bps / 10000.0))
                        if trade['side'] == 'buy'
                        else trade['price'] * (1.0 - (limit_offset_bps / 10000.0)),
                        2
                    )
                order_info['order_type'] = order_type
                logger.info(
                    f"Submitted: {trade['side']} {trade['qty']:.4f} {trade['symbol']} "
                    f"(order_id={order_id}, status={order_status})"
                )

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


def get_open_orders(api: tradeapi.REST) -> List[Dict]:
    """Get currently open orders."""
    try:
        orders = api.list_orders(status='open', nested=False)
        serialized = []
        for o in orders:
            serialized.append({
                'id': getattr(o, 'id', None),
                'client_order_id': getattr(o, 'client_order_id', None),
                'symbol': getattr(o, 'symbol', None),
                'side': getattr(o, 'side', None),
                'qty': getattr(o, 'qty', None),
                'notional': getattr(o, 'notional', None),
                'status': getattr(o, 'status', None)
            })
        return serialized
    except Exception as e:
        logger.error(f"Error getting open orders: {e}")
        return []


def cancel_open_orders(api: tradeapi.REST, open_orders: List[Dict]) -> List[Dict]:
    """Cancel provided open orders and return cancellation results."""
    results: List[Dict] = []
    for order in open_orders:
        order_id = order.get('id')
        symbol = order.get('symbol')
        try:
            if not order_id:
                raise RuntimeError("Missing order id")
            api.cancel_order(order_id)
            results.append({
                'id': order_id,
                'symbol': symbol,
                'status': 'cancelled'
            })
            logger.info(f"Cancelled open order {order_id} ({symbol})")
        except Exception as e:
            results.append({
                'id': order_id,
                'symbol': symbol,
                'status': 'cancel_failed',
                'error': str(e)
            })
            logger.error(f"Failed to cancel open order {order_id} ({symbol}): {e}")
    return results


def get_account_equity(api: tradeapi.REST) -> float:
    """Get account equity."""
    try:
        account = api.get_account()
        return float(account.equity)
    except Exception as e:
        logger.error(f"Error getting account equity: {e}")
        return 0.0
