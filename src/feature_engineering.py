import pandas as pd
import numpy as np

class FeatureEngineer:
    def __init__(self, config):
        self.config = config

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes features for ML model.
        df: Multi-index (timestamp, symbol) with 'close' and 'volume'.
        """
        # Ensure we are working with a copy
        df = df.copy()
        
        # Sort index to ensure correct rolling calculations
        df = df.sort_index()
        
        # Group by symbol for feature calculation
        grouped = df.groupby(level=1)
        
        # 1. Momentum returns (1w, 1m, 3m, 6m, 12m)
        # Using approx trading days: 1w=5, 1m=21, 3m=63, 6m=126, 12m=252
        df['ret_1w'] = grouped['close'].pct_change(5)
        df['ret_1m'] = grouped['close'].pct_change(21)
        df['ret_3m'] = grouped['close'].pct_change(63)
        df['ret_6m'] = grouped['close'].pct_change(126)
        df['ret_12m'] = grouped['close'].pct_change(252)
        
        # 2. Volatility (1m, 3m realized vol)
        df['vol_1m'] = grouped['close'].apply(lambda x: x.pct_change().rolling(21).std())
        df['vol_3m'] = grouped['close'].apply(lambda x: x.pct_change().rolling(63).std())
        
        # 3. Trend (distance to 50/100/200-day SMA)
        df['sma_50'] = grouped['close'].apply(lambda x: x.rolling(50).mean())
        df['sma_100'] = grouped['close'].apply(lambda x: x.rolling(100).mean())
        df['sma_200'] = grouped['close'].apply(lambda x: x.rolling(200).mean())
        
        df['dist_sma_50'] = (df['close'] / df['sma_50']) - 1
        df['dist_sma_100'] = (df['close'] / df['sma_100']) - 1
        df['dist_sma_200'] = (df['close'] / df['sma_200']) - 1
        
        # 4. Liquidity (dollar volume proxy)
        df['dollar_vol'] = df['close'] * df['volume']
        df['avg_dollar_vol_1m'] = grouped['dollar_vol'].apply(lambda x: x.rolling(21).mean())
        
        # 5. Short-term reversal (last week return - same as ret_1w but conceptualized differently)
        # Often used as a negative signal in some regimes.
        
        # Drop columns used for intermediate calculations
        cols_to_drop = ['sma_50', 'sma_100', 'sma_200', 'dollar_vol']
        df = df.drop(columns=cols_to_drop)
        
        # Drop rows with NaN features
        return df.dropna()



