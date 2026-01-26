"""
Simplified Strategy - Academically-backed momentum with single XGBoost model.

Key differences from stacking approach:
1. Single model (less overfitting)
2. Only 5 features (academically proven)
3. Proper time-series cross-validation
4. Focus on 6-12 month momentum (Jegadeesh & Titman, 1993)
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
import joblib
import os
from loguru import logger
from typing import Dict, List, Tuple


class SimpleStrategy:
    """
    Simple momentum strategy using single XGBoost model.

    Based on academic research:
    - Jegadeesh & Titman (1993): 6-12 month momentum works
    - Gu, Kelly & Xiu (2020): Simple features often beat complex ones
    """

    def __init__(self, config: Dict, model_dir: str = 'models', horizon: str = '5d'):
        self.config = config
        self.model_dir = model_dir
        self.horizon = horizon  # '1d', '5d', or '20d'
        self.model = None

        # Only 5 features - all academically documented
        # Same features used across all horizons to avoid overfitting
        self.feature_cols = [
            'momentum_12_1',   # 12-month return minus 1-month (classic momentum)
            'momentum_6m',     # 6-month return
            'volatility_3m',   # Risk adjustment
            'volume_trend',    # Liquidity signal
            'mean_reversion',  # Short-term reversal
        ]

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute only academically-backed features.

        Minimal feature set to reduce overfitting.
        """
        df = df.copy()
        df = df.sort_index()

        grouped = df.groupby(level=1)

        # 1. Classic Momentum (12-1): Jegadeesh & Titman
        # Skip the most recent month (short-term reversal)
        ret_12m = grouped['close'].pct_change(252)
        ret_1m = grouped['close'].pct_change(21)
        df['momentum_12_1'] = ret_12m - ret_1m

        # 2. 6-month momentum
        df['momentum_6m'] = grouped['close'].pct_change(126)

        # 3. Volatility (risk-adjusted)
        df['returns'] = grouped['close'].pct_change()
        df['volatility_3m'] = grouped['returns'].transform(lambda x: x.rolling(63).std())

        # 4. Volume trend (liquidity)
        vol_sma = grouped['volume'].transform(lambda x: x.rolling(21).mean())
        vol_sma_long = grouped['volume'].transform(lambda x: x.rolling(63).mean())
        df['volume_trend'] = vol_sma / vol_sma_long - 1

        # 5. Short-term mean reversion (1-week return, used negatively)
        df['mean_reversion'] = -grouped['close'].pct_change(5)

        return df.dropna()

    def train(self, df: pd.DataFrame, target_col: str = 'target') -> Dict:
        """
        Train with proper time-series cross-validation.
        """
        logger.info("Training simple XGBoost model...")

        available_features = [c for c in self.feature_cols if c in df.columns]
        logger.info(f"Using {len(available_features)} features: {available_features}")

        X = df[available_features].values
        y = df[target_col].values

        # Remove NaN/inf
        mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
        X = X[mask]
        y = y[mask]

        if len(X) < 500:
            raise ValueError(f"Insufficient data: {len(X)} samples")

        # Time-series cross-validation (5 folds)
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Simple XGBoost - deliberately underfit to generalize
            model = xgb.XGBRegressor(
                n_estimators=50,      # Low to prevent overfitting
                max_depth=3,          # Shallow trees
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,        # L1 regularization
                reg_lambda=1.0,       # L2 regularization
                random_state=42,
                n_jobs=-1
            )

            model.fit(X_train, y_train)

            val_pred = model.predict(X_val)
            corr = np.corrcoef(val_pred, y_val)[0, 1]
            cv_scores.append(corr)
            logger.info(f"Fold {fold+1} correlation: {corr:.4f}")

        # Final model on all data
        self.model = xgb.XGBRegressor(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X, y)

        # Save with horizon suffix
        os.makedirs(self.model_dir, exist_ok=True)
        model_path = os.path.join(self.model_dir, f'simple_xgb_{self.horizon}.pkl')
        features_path = os.path.join(self.model_dir, f'simple_features_{self.horizon}.pkl')
        joblib.dump(self.model, model_path)
        joblib.dump(available_features, features_path)
        logger.info(f"Model saved: {model_path}")

        metrics = {
            'cv_mean_corr': np.mean(cv_scores),
            'cv_std_corr': np.std(cv_scores),
            'cv_min_corr': np.min(cv_scores),
            'cv_max_corr': np.max(cv_scores),
            'n_samples': len(X),
            'n_features': len(available_features)
        }

        logger.info(f"CV Mean Correlation: {metrics['cv_mean_corr']:.4f} ± {metrics['cv_std_corr']:.4f}")

        return metrics

    def load_model(self, horizon: str = None) -> bool:
        """Load trained model for specific horizon."""
        if horizon:
            self.horizon = horizon

        try:
            model_path = os.path.join(self.model_dir, f'simple_xgb_{self.horizon}.pkl')
            features_path = os.path.join(self.model_dir, f'simple_features_{self.horizon}.pkl')

            if os.path.exists(model_path):
                self.model = joblib.load(model_path)
                if os.path.exists(features_path):
                    self.feature_cols = joblib.load(features_path)
                logger.info(f"Loaded {self.horizon} model from {model_path}")
                return True
        except Exception as e:
            logger.error(f"Error loading {self.horizon} model: {e}")
        return False

    def compute_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate signals (stock rankings)."""
        if self.model is None:
            if not self.load_model():
                raise ValueError("Model not trained or loaded")

        # Compute features if not already done
        if 'momentum_12_1' not in df.columns:
            df = self.compute_features(df)

        # Get latest data for each ticker
        latest = df.groupby(level=1).tail(1).copy()

        available_features = [c for c in self.feature_cols if c in latest.columns]
        X = latest[available_features].values

        # Handle missing values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        scores = self.model.predict(X)

        signals = pd.DataFrame({
            'ticker': latest.index.get_level_values(1),
            'score': scores
        })

        signals['rank'] = signals['score'].rank(ascending=False)
        return signals.sort_values('rank').set_index('ticker')

    def compute_multi_horizon_signals(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Generate signals from all 3 time horizons.

        Returns:
            Dict with keys '1d', '5d', '20d' mapping to signal DataFrames
        """
        # Compute features once
        if 'momentum_12_1' not in df.columns:
            df = self.compute_features(df)

        signals = {}
        for horizon in ['1d', '5d', '20d']:
            logger.info(f"Computing {horizon} signals...")
            if self.load_model(horizon):
                signals[horizon] = self.compute_signals(df)
                logger.info(f"  {horizon}: {len(signals[horizon])} stocks ranked")
            else:
                logger.warning(f"Could not load {horizon} model")

        return signals

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance."""
        if self.model is None:
            return pd.DataFrame()

        importance = self.model.feature_importances_
        return pd.DataFrame({
            'feature': self.feature_cols[:len(importance)],
            'importance': importance
        }).sort_values('importance', ascending=False)


class PureMomentumStrategy:
    """
    Even simpler: pure momentum ranking, no ML.

    Just rank stocks by 12-1 momentum and buy the top N.
    This is the academic baseline that's hard to beat.
    """

    def __init__(self, config: Dict):
        self.config = config

    def compute_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Rank stocks by classic 12-1 momentum.
        No ML, no overfitting possible.
        """
        df = df.copy()
        df = df.sort_index()

        grouped = df.groupby(level=1)

        # 12-1 momentum: 12-month return minus 1-month return
        ret_12m = grouped['close'].pct_change(252)
        ret_1m = grouped['close'].pct_change(21)
        df['momentum'] = ret_12m - ret_1m

        # Get latest momentum for each ticker
        latest = df.groupby(level=1).tail(1).copy()

        signals = pd.DataFrame({
            'ticker': latest.index.get_level_values(1),
            'score': latest['momentum'].values
        })

        # Remove NaN
        signals = signals.dropna()

        signals['rank'] = signals['score'].rank(ascending=False)
        return signals.sort_values('rank').set_index('ticker')
