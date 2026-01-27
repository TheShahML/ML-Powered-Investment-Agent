"""Canary baseline portfolio - pure momentum shadow (no ML, no Alpaca orders)."""
import pandas as pd
import numpy as np
from loguru import logger
from typing import Dict


class CanaryTracker:
    """Track pure momentum baseline portfolio (shadow, no actual trading)."""

    def __init__(self, config: Dict):
        self.config = config
        self.holdings = {}
        self.portfolio_value = 0.0

    def compute_momentum_signals(self, df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
        """
        Compute pure 12-1 momentum rankings.

        Args:
            df: Historical data with MultiIndex (timestamp, symbol)
            top_n: Number of stocks to select

        Returns:
            DataFrame with symbol, score, rank
        """
        df = df.copy().sort_index()
        grouped = df.groupby(level=1)

        # 12-1 momentum
        ret_12m = grouped['close'].pct_change(252)
        ret_1m = grouped['close'].pct_change(21)
        df['momentum'] = ret_12m - ret_1m

        # Get latest for each symbol
        latest = df.groupby(level=1).tail(1).copy()

        signals = pd.DataFrame({
            'ticker': latest.index.get_level_values(1),
            'score': latest['momentum'].values
        })

        signals = signals.dropna()
        signals['rank'] = signals['score'].rank(ascending=False)

        return signals.sort_values('rank').set_index('ticker')

    def simulate_rebalance(
        self,
        signals: pd.DataFrame,
        portfolio_value: float,
        top_n: int = 20,
        cost_bps: float = 15.0
    ) -> Dict:
        """
        Simulate rebalance (paper only, no actual orders).

        Args:
            signals: Momentum signals
            portfolio_value: Total portfolio value
            top_n: Number of holdings
            cost_bps: Transaction cost in basis points

        Returns:
            Dict with turnover and cost estimate
        """
        # Equal weight among top N
        target_weight = 1.0 / top_n
        targets = signals.head(top_n)

        # Simple equal weight (for now, can enhance with inverse-vol later)
        new_holdings = {
            ticker: target_weight * portfolio_value
            for ticker in targets.index
        }

        # Compute turnover
        if not self.holdings:
            turnover = 1.0  # First rebalance = 100% turnover
        else:
            # Sum of absolute weight changes
            all_symbols = set(self.holdings.keys()) | set(new_holdings.keys())
            total_diff = sum(
                abs(new_holdings.get(s, 0) - self.holdings.get(s, 0))
                for s in all_symbols
            )
            turnover = total_diff / (2 * portfolio_value) if portfolio_value > 0 else 0

        cost = turnover * (cost_bps / 10000.0) * portfolio_value

        self.holdings = new_holdings
        self.portfolio_value = portfolio_value - cost

        logger.info(f"Canary rebalance: {len(new_holdings)} holdings, turnover {turnover:.2%}, cost ${cost:.2f}")

        return {
            'turnover': turnover,
            'cost': cost,
            'holdings_count': len(new_holdings)
        }

    def mark_to_market(self, latest_prices: Dict[str, float]) -> float:
        """
        Update portfolio value based on latest prices.

        Args:
            latest_prices: {symbol: price}

        Returns:
            Updated portfolio value
        """
        if not self.holdings:
            return self.portfolio_value

        # Update each holding value
        total_value = 0.0
        for symbol, target_value in self.holdings.items():
            if symbol in latest_prices:
                # Assume we held shares = target_value / original_price
                # This is approximate; for shadow we just scale
                total_value += target_value  # Simplified

        self.portfolio_value = total_value
        return total_value

    def get_return_since_last_rebalance(self, initial_value: float) -> float:
        """Compute return % since last rebalance."""
        if initial_value <= 0:
            return 0.0
        return (self.portfolio_value - initial_value) / initial_value
