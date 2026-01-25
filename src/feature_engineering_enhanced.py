"""
Enhanced Feature Engineering - Combines original features with Stock-Prediction-Models techniques.
Includes technical indicators: RSI, MACD, Bollinger Bands, OBV, and more.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional


class EnhancedFeatureEngineer:
    """
    Enhanced feature engineering that adds technical indicators
    commonly used in the Stock-Prediction-Models repo.
    """

    def __init__(self, config: Dict):
        self.config = config

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes all features for ML model.

        Args:
            df: Multi-index (timestamp, symbol) with 'close', 'volume', 'high', 'low', 'open'

        Returns:
            DataFrame with all features computed
        """
        df = df.copy()
        df = df.sort_index()

        grouped = df.groupby(level=1)

        # ===== MOMENTUM FEATURES =====
        df['ret_1w'] = grouped['close'].pct_change(5)
        df['ret_1m'] = grouped['close'].pct_change(21)
        df['ret_3m'] = grouped['close'].pct_change(63)
        df['ret_6m'] = grouped['close'].pct_change(126)
        df['ret_12m'] = grouped['close'].pct_change(252)

        # ===== VOLATILITY FEATURES =====
        df['vol_1m'] = grouped['close'].apply(
            lambda x: x.pct_change().rolling(21).std()
        )
        df['vol_3m'] = grouped['close'].apply(
            lambda x: x.pct_change().rolling(63).std()
        )

        # ===== TREND FEATURES (SMA Distance) =====
        df['sma_50'] = grouped['close'].transform(lambda x: x.rolling(50).mean())
        df['sma_100'] = grouped['close'].transform(lambda x: x.rolling(100).mean())
        df['sma_200'] = grouped['close'].transform(lambda x: x.rolling(200).mean())

        df['dist_sma_50'] = (df['close'] / df['sma_50']) - 1
        df['dist_sma_100'] = (df['close'] / df['sma_100']) - 1
        df['dist_sma_200'] = (df['close'] / df['sma_200']) - 1

        # ===== LIQUIDITY FEATURES =====
        df['dollar_vol'] = df['close'] * df['volume']
        df['avg_dollar_vol_1m'] = grouped['dollar_vol'].transform(
            lambda x: x.rolling(21).mean()
        )

        # ===== RSI (Relative Strength Index) =====
        df['rsi_14'] = grouped['close'].apply(self._compute_rsi, periods=14)

        # ===== MACD =====
        df['ema_12'] = grouped['close'].transform(lambda x: x.ewm(span=12).mean())
        df['ema_26'] = grouped['close'].transform(lambda x: x.ewm(span=26).mean())
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = grouped['macd'].transform(lambda x: x.ewm(span=9).mean())
        df['macd_hist'] = df['macd'] - df['macd_signal']

        # ===== BOLLINGER BANDS =====
        df['bb_middle'] = grouped['close'].transform(lambda x: x.rolling(20).mean())
        df['bb_std'] = grouped['close'].transform(lambda x: x.rolling(20).std())
        df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
        df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
        # Position within bands (0 = lower, 1 = upper)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

        # ===== VOLUME FEATURES =====
        df['volume_sma_20'] = grouped['volume'].transform(lambda x: x.rolling(20).mean())
        df['volume_sma_ratio'] = df['volume'] / df['volume_sma_20']

        # OBV (On-Balance Volume) slope
        df['obv'] = grouped.apply(self._compute_obv).reset_index(level=0, drop=True)
        df['obv_slope'] = grouped['obv'].transform(
            lambda x: x.rolling(10).apply(self._slope, raw=True)
        )

        # ===== ATR (Average True Range) =====
        if 'high' in df.columns and 'low' in df.columns:
            df['tr'] = self._compute_true_range(df, grouped)
            df['atr_14'] = grouped['tr'].transform(lambda x: x.rolling(14).mean())
            df['atr_pct'] = df['atr_14'] / df['close']

        # ===== MOMENTUM OSCILLATORS =====
        # Stochastic %K
        df['low_14'] = grouped['low'].transform(lambda x: x.rolling(14).min()) if 'low' in df.columns else grouped['close'].transform(lambda x: x.rolling(14).min())
        df['high_14'] = grouped['high'].transform(lambda x: x.rolling(14).max()) if 'high' in df.columns else grouped['close'].transform(lambda x: x.rolling(14).max())
        df['stoch_k'] = 100 * (df['close'] - df['low_14']) / (df['high_14'] - df['low_14'] + 1e-10)
        df['stoch_d'] = grouped['stoch_k'].transform(lambda x: x.rolling(3).mean())

        # ===== SECTOR MOMENTUM (placeholder - needs sector data) =====
        # This would compare stock momentum to its sector
        df['sector_momentum'] = df['ret_1m']  # Placeholder

        # ===== CLEANUP =====
        cols_to_drop = [
            'sma_50', 'sma_100', 'sma_200', 'dollar_vol',
            'ema_12', 'ema_26', 'bb_middle', 'bb_std', 'bb_upper', 'bb_lower',
            'volume_sma_20', 'obv', 'low_14', 'high_14'
        ]
        # Only drop columns that exist
        cols_to_drop = [c for c in cols_to_drop if c in df.columns]
        df = df.drop(columns=cols_to_drop)

        # Also drop intermediate columns if they exist
        for col in ['tr']:
            if col in df.columns:
                df = df.drop(columns=[col])

        return df.dropna()

    @staticmethod
    def _compute_rsi(prices: pd.Series, periods: int = 14) -> pd.Series:
        """Compute RSI indicator."""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=periods, min_periods=1).mean()
        avg_loss = loss.rolling(window=periods, min_periods=1).mean()

        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def _compute_obv(group: pd.DataFrame) -> pd.Series:
        """Compute On-Balance Volume."""
        close = group['close']
        volume = group['volume']

        obv = pd.Series(index=close.index, dtype=float)
        obv.iloc[0] = 0

        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
            elif close.iloc[i] < close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]

        return obv

    @staticmethod
    def _slope(arr: np.ndarray) -> float:
        """Calculate slope of array using linear regression."""
        if len(arr) < 2:
            return 0.0
        x = np.arange(len(arr))
        slope = np.polyfit(x, arr, 1)[0]
        return slope

    def _compute_true_range(self, df: pd.DataFrame, grouped) -> pd.Series:
        """Compute True Range for ATR."""
        high = df['high'] if 'high' in df.columns else df['close']
        low = df['low'] if 'low' in df.columns else df['close']

        prev_close = grouped['close'].shift(1)

        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)

        return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    def get_feature_list(self) -> list:
        """Return list of all computed features."""
        return [
            # Momentum
            'ret_1w', 'ret_1m', 'ret_3m', 'ret_6m', 'ret_12m',
            # Volatility
            'vol_1m', 'vol_3m',
            # Trend
            'dist_sma_50', 'dist_sma_100', 'dist_sma_200',
            # Liquidity
            'avg_dollar_vol_1m',
            # Technical indicators
            'rsi_14', 'macd', 'macd_signal', 'macd_hist',
            'bb_position', 'volume_sma_ratio', 'obv_slope',
            'stoch_k', 'stoch_d',
            # ATR (if high/low available)
            'atr_pct',
            # Sector
            'sector_momentum'
        ]
