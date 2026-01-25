"""
Bitcoin/Crypto Strategy for the investment bot.

Note: Unlike equities where we rank and pick top stocks,
crypto is handled as a simple allocation strategy.

Options:
1. Fixed allocation (e.g., always 10% in BTC)
2. Momentum-based (buy more when trending up)
3. DCA (dollar-cost average regardless of price)
"""

import pandas as pd
import numpy as np
from loguru import logger
from typing import Dict, List
from datetime import date, timedelta


class CryptoStrategy:
    """
    Simple Bitcoin allocation strategy.
    Maintains a fixed percentage of portfolio in BTC.
    """

    def __init__(self, config: Dict):
        self.config = config
        self.crypto_allocation = config.get('crypto_allocation', 0.10)  # 10% default
        self.crypto_tickers = config.get('crypto_tickers', ['BTC/USD'])

    def get_target_allocation(self, portfolio_value: float) -> Dict[str, float]:
        """
        Calculate target crypto allocation.

        Args:
            portfolio_value: Total portfolio value in USD

        Returns:
            Dict of {symbol: target_usd_value}
        """
        crypto_budget = portfolio_value * self.crypto_allocation

        # Split evenly among crypto assets
        per_asset = crypto_budget / len(self.crypto_tickers)

        allocations = {ticker: per_asset for ticker in self.crypto_tickers}

        logger.info(f"Crypto allocation: ${crypto_budget:.2f} ({self.crypto_allocation*100:.0f}% of portfolio)")
        for ticker, value in allocations.items():
            logger.info(f"  {ticker}: ${value:.2f}")

        return allocations

    def should_rebalance_crypto(self, current_value: float, target_value: float, threshold: float = 0.05) -> bool:
        """
        Check if crypto needs rebalancing.

        Only rebalance if allocation drifted more than threshold.
        """
        if target_value == 0:
            return current_value > 0

        drift = abs(current_value - target_value) / target_value

        if drift > threshold:
            logger.info(f"Crypto drift: {drift*100:.1f}% (threshold: {threshold*100:.1f}%)")
            return True

        return False


class BitcoinMomentumStrategy:
    """
    Bitcoin momentum strategy.
    Increases allocation when BTC is trending up, decreases when down.

    WARNING: More complex = more ways to fool yourself.
    The simple fixed allocation is probably better.
    """

    def __init__(self, config: Dict):
        self.config = config
        self.base_allocation = config.get('crypto_allocation', 0.10)
        self.min_allocation = 0.05  # Never less than 5%
        self.max_allocation = 0.15  # Never more than 15%

    def compute_btc_momentum(self, btc_prices: pd.Series) -> float:
        """
        Compute BTC momentum signal.
        Returns a value between -1 (bearish) and +1 (bullish).
        """
        if len(btc_prices) < 50:
            return 0.0

        # 50-day vs 200-day moving average crossover
        sma_50 = btc_prices.rolling(50).mean().iloc[-1]
        sma_200 = btc_prices.rolling(200).mean().iloc[-1] if len(btc_prices) >= 200 else sma_50

        if sma_200 == 0:
            return 0.0

        # Normalize: +1 if 50 SMA is 20% above 200 SMA, -1 if 20% below
        momentum = (sma_50 / sma_200 - 1) / 0.20
        momentum = max(-1, min(1, momentum))  # Clamp to [-1, 1]

        return momentum

    def get_dynamic_allocation(self, btc_prices: pd.Series) -> float:
        """
        Get dynamic BTC allocation based on momentum.
        """
        momentum = self.compute_btc_momentum(btc_prices)

        # Scale allocation based on momentum
        # momentum = 0 -> base allocation
        # momentum = 1 -> max allocation
        # momentum = -1 -> min allocation
        if momentum >= 0:
            allocation = self.base_allocation + (self.max_allocation - self.base_allocation) * momentum
        else:
            allocation = self.base_allocation + (self.base_allocation - self.min_allocation) * momentum

        logger.info(f"BTC momentum: {momentum:.2f}, allocation: {allocation*100:.1f}%")

        return allocation


def get_crypto_orders(
    api,
    target_allocations: Dict[str, float],
    dry_run: bool = False
) -> List[Dict]:
    """
    Generate crypto orders to reach target allocations.

    Args:
        api: Alpaca API client
        target_allocations: {symbol: target_usd_value}
        dry_run: If True, don't place orders

    Returns:
        List of order info dicts
    """
    orders = []

    for symbol, target_value in target_allocations.items():
        try:
            # Get current position
            try:
                position = api.get_position(symbol.replace('/', ''))
                current_value = float(position.market_value)
            except Exception:
                current_value = 0.0

            # Get current price
            quote = api.get_latest_crypto_quote(symbol)
            price = float(quote.ap)  # Ask price

            # Calculate order
            diff_value = target_value - current_value

            if abs(diff_value) < 10:  # Min $10 trade
                continue

            diff_qty = diff_value / price
            side = 'buy' if diff_qty > 0 else 'sell'

            order_info = {
                'symbol': symbol,
                'side': side,
                'notional': abs(diff_value),
                'qty': abs(diff_qty),
                'type': 'market',
                'time_in_force': 'gtc'  # Good til cancelled for crypto
            }

            if dry_run:
                logger.info(f"[DRY RUN] Would {side} ${abs(diff_value):.2f} of {symbol}")
                order_info['status'] = 'dry_run'
            else:
                # Place crypto order
                order = api.submit_order(
                    symbol=symbol.replace('/', ''),
                    notional=abs(diff_value),
                    side=side,
                    type='market',
                    time_in_force='gtc'
                )
                order_info['status'] = 'submitted'
                order_info['order_id'] = order.id
                logger.info(f"Crypto order: {side} ${abs(diff_value):.2f} of {symbol}")

            orders.append(order_info)

        except Exception as e:
            logger.error(f"Error processing crypto order for {symbol}: {e}")
            orders.append({
                'symbol': symbol,
                'status': 'failed',
                'error': str(e)
            })

    return orders
