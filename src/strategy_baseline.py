import pandas as pd
import numpy as np

class MomentumStrategy:
    """
    Baseline strategy: Pure momentum ranking (e.g., 6-12 month return).
    """
    def __init__(self, config):
        self.config = config
        self.lookback_months = [6, 12]

    def compute_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates momentum scores for all tickers in the dataframe.
        Expects a multi-index dataframe (timestamp, symbol) with 'close'.
        """
        # Pivot to get wide format: index=timestamp, columns=symbol
        closes = df['close'].unstack()
        
        # Calculate returns over 6 and 12 months (approx 126 and 252 trading days)
        # Using log returns for momentum
        ret_6m = closes.pct_change(126)
        ret_12m = closes.pct_change(252)
        
        # Combined momentum: average of 6m and 12m returns
        momentum = (ret_6m + ret_12m) / 2
        
        # Get the most recent scores
        latest_momentum = momentum.iloc[-1].dropna()
        
        # Rank tickers (descending)
        ranks = latest_momentum.rank(ascending=False)
        
        signals = pd.DataFrame({
            'score': latest_momentum,
            'rank': ranks
        })
        
        return signals.sort_values('rank')



