import lightgbm as lgb
import pandas as pd
import numpy as np
from loguru import logger
import os
import joblib

class MLStrategy:
    """
    ML strategy using LightGBM to rank stocks.
    """
    def __init__(self, config, model_path='research/models/latest_model.pkl'):
        self.config = config
        self.model_path = model_path
        self.model = None
        self.feature_cols = [
            'ret_1w', 'ret_1m', 'ret_3m', 'ret_6m', 'ret_12m',
            'vol_1m', 'vol_3m', 'dist_sma_50', 'dist_sma_100', 'dist_sma_200',
            'avg_dollar_vol_1m'
        ]

    def train(self, df: pd.DataFrame):
        """
        Trains the LightGBM model.
        df: Feature-engineered dataframe with 'target' column.
        """
        logger.info("Training LightGBM model...")
        
        # Simple target: next week's return
        # The user's requirement is cross-sectional ranking.
        # We can use LGBMRanker or LGBMRegressor. For simplicity, Regressor with ranking targets.
        
        X = df[self.feature_cols]
        y = df['target']
        
        # Train-test split (walk-forward style: last 20% for validation)
        split_idx = int(len(df) * 0.8)
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
        
        self.model = lgb.LGBMRegressor(
            n_estimators=100,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=10)]
        )
        
        # Save model
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.model, self.model_path)
        logger.info(f"Model saved to {self.model_path}")

    def load_model(self):
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            return True
        return False

    def compute_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predicts scores and ranks for the given features.
        """
        if self.model is None:
            if not self.load_model():
                raise ValueError("Model not trained or loaded.")
        
        # Expects latest features for all tickers
        latest_features = df.groupby(level=1).tail(1)
        
        X = latest_features[self.feature_cols]
        scores = self.model.predict(X)
        
        signals = pd.DataFrame({
            'ticker': latest_features.index.get_level_values(1),
            'score': scores
        })
        
        signals['rank'] = signals['score'].rank(ascending=False)
        return signals.sort_values('rank').set_index('ticker')



