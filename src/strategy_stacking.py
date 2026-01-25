"""
Stacking Ensemble Strategy - Combines multiple models for improved predictions.
Based on the Stock-Prediction-Models repo approach with XGBoost, LightGBM, and ensemble methods.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
    ExtraTreesRegressor,
    BaggingRegressor,
    VotingRegressor
)
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
import joblib
import os
from loguru import logger
from typing import Dict, List, Optional, Tuple


class StackingEnsembleStrategy:
    """
    Stacking ensemble that combines:
    - XGBoost
    - LightGBM
    - Random Forest
    - Gradient Boosting
    - Extra Trees

    Uses a Ridge meta-learner to combine predictions.
    """

    def __init__(self, config: Dict, model_dir: str = 'models'):
        self.config = config
        self.model_dir = model_dir
        self.scaler = StandardScaler()

        self.feature_cols = [
            # Momentum features
            'ret_1w', 'ret_1m', 'ret_3m', 'ret_6m', 'ret_12m',
            # Volatility features
            'vol_1m', 'vol_3m',
            # Trend features
            'dist_sma_50', 'dist_sma_100', 'dist_sma_200',
            # Liquidity
            'avg_dollar_vol_1m',
            # Technical indicators (new)
            'rsi_14', 'macd', 'macd_signal', 'bb_position',
            # Volume features (new)
            'volume_sma_ratio', 'obv_slope',
            # Sector momentum (new)
            'sector_momentum'
        ]

        # Base models with fast parameters for GitHub Actions
        self.base_models = {
            'xgboost': xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            ),
            'lightgbm': lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbosity=-1
            ),
            'random_forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=10,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boost': GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42
            ),
            'extra_trees': ExtraTreesRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=10,
                random_state=42,
                n_jobs=-1
            )
        }

        # Meta learner
        self.meta_learner = Ridge(alpha=1.0)

        self.is_trained = False

    def _get_available_features(self, df: pd.DataFrame) -> List[str]:
        """Get features that exist in the dataframe."""
        return [col for col in self.feature_cols if col in df.columns]

    def train(self, df: pd.DataFrame, target_col: str = 'target') -> Dict:
        """
        Train the stacking ensemble using walk-forward validation.

        Args:
            df: Feature-engineered dataframe with target column
            target_col: Name of the target column

        Returns:
            Dictionary with training metrics
        """
        logger.info("Training stacking ensemble model...")

        available_features = self._get_available_features(df)
        logger.info(f"Using {len(available_features)} features: {available_features}")

        X = df[available_features].values
        y = df[target_col].values

        # Remove any remaining NaN/inf values
        mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
        X = X[mask]
        y = y[mask]

        if len(X) < 100:
            raise ValueError(f"Insufficient data for training: {len(X)} samples")

        # Walk-forward split (80/20)
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        # Train base models and collect OOF predictions
        base_preds_train = np.zeros((len(X_train), len(self.base_models)))
        base_preds_val = np.zeros((len(X_val), len(self.base_models)))

        metrics = {}

        for i, (name, model) in enumerate(self.base_models.items()):
            logger.info(f"Training {name}...")

            model.fit(X_train_scaled, y_train)

            # Predictions
            base_preds_train[:, i] = model.predict(X_train_scaled)
            base_preds_val[:, i] = model.predict(X_val_scaled)

            # Track individual model performance
            val_corr = np.corrcoef(base_preds_val[:, i], y_val)[0, 1]
            metrics[f'{name}_corr'] = val_corr
            logger.info(f"{name} validation correlation: {val_corr:.4f}")

        # Train meta-learner on base model predictions
        logger.info("Training meta-learner...")
        self.meta_learner.fit(base_preds_train, y_train)

        # Final ensemble predictions
        final_preds = self.meta_learner.predict(base_preds_val)
        ensemble_corr = np.corrcoef(final_preds, y_val)[0, 1]
        metrics['ensemble_corr'] = ensemble_corr
        logger.info(f"Ensemble validation correlation: {ensemble_corr:.4f}")

        # Save models
        self._save_models(available_features)

        self.is_trained = True
        return metrics

    def _save_models(self, feature_cols: List[str]):
        """Save all models to disk."""
        os.makedirs(self.model_dir, exist_ok=True)

        # Save base models
        for name, model in self.base_models.items():
            path = os.path.join(self.model_dir, f'{name}.pkl')
            joblib.dump(model, path)

        # Save meta-learner
        joblib.dump(self.meta_learner, os.path.join(self.model_dir, 'meta_learner.pkl'))

        # Save scaler
        joblib.dump(self.scaler, os.path.join(self.model_dir, 'scaler.pkl'))

        # Save feature list
        joblib.dump(feature_cols, os.path.join(self.model_dir, 'feature_cols.pkl'))

        logger.info(f"Models saved to {self.model_dir}")

    def load_models(self) -> bool:
        """Load all models from disk."""
        try:
            # Load base models
            for name in self.base_models.keys():
                path = os.path.join(self.model_dir, f'{name}.pkl')
                if os.path.exists(path):
                    self.base_models[name] = joblib.load(path)

            # Load meta-learner
            meta_path = os.path.join(self.model_dir, 'meta_learner.pkl')
            if os.path.exists(meta_path):
                self.meta_learner = joblib.load(meta_path)

            # Load scaler
            scaler_path = os.path.join(self.model_dir, 'scaler.pkl')
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)

            # Load feature columns
            features_path = os.path.join(self.model_dir, 'feature_cols.pkl')
            if os.path.exists(features_path):
                self.feature_cols = joblib.load(features_path)

            self.is_trained = True
            logger.info("Models loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False

    def compute_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals (stock rankings) for the latest data.

        Args:
            df: Feature-engineered dataframe

        Returns:
            DataFrame with ticker, score, and rank columns
        """
        if not self.is_trained:
            if not self.load_models():
                raise ValueError("Models not trained or loaded")

        # Get latest data point for each ticker
        latest = df.groupby(level=1).tail(1).copy()

        available_features = self._get_available_features(latest)
        X = latest[available_features].values

        # Handle missing values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Scale
        X_scaled = self.scaler.transform(X)

        # Get base model predictions
        base_preds = np.zeros((len(X), len(self.base_models)))
        for i, (name, model) in enumerate(self.base_models.items()):
            base_preds[:, i] = model.predict(X_scaled)

        # Meta-learner ensemble prediction
        scores = self.meta_learner.predict(base_preds)

        # Create signals dataframe
        signals = pd.DataFrame({
            'ticker': latest.index.get_level_values(1),
            'score': scores
        })

        signals['rank'] = signals['score'].rank(ascending=False)
        signals = signals.sort_values('rank').set_index('ticker')

        return signals

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from XGBoost model."""
        if not self.is_trained:
            return pd.DataFrame()

        xgb_model = self.base_models['xgboost']
        importance = xgb_model.feature_importances_

        return pd.DataFrame({
            'feature': self.feature_cols[:len(importance)],
            'importance': importance
        }).sort_values('importance', ascending=False)
