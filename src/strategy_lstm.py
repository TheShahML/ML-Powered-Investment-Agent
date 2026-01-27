"""
LSTM-based Neural Network Strategy

Uses LSTM (Long Short-Term Memory) networks to learn temporal patterns in stock data.
Based on proven architectures from stock prediction research.

Reference: https://github.com/pranityadav19/Stock-Prediction-Models
"""

import numpy as np
import pandas as pd
import os
import joblib
from loguru import logger
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')
import time

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logger.warning("TensorFlow not available - LSTM strategy will not work. Install with: pip install tensorflow")

from .strategy_base import BaseStrategy
from .leakage_safe_cv import train_with_leakage_safe_cv
from .tf_runtime import configure_tensorflow


class LSTMStrategy(BaseStrategy):
    """
    LSTM-based strategy for stock prediction.
    
    Uses sequence-to-sequence LSTM architecture to learn temporal patterns.
    Processes sequences of features over time windows.
    """
    
    def __init__(self, config: Dict, model_dir: str = 'models', horizon: str = '20d'):
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow required for LSTM strategy. Install with: pip install tensorflow")
        
        super().__init__(config, model_dir, horizon)

        # Prefer GPU if available, otherwise CPU (safe fallback)
        try:
            self.tf_runtime = configure_tensorflow(tf)
        except Exception as e:
            # Never hard-fail on runtime config; training can proceed on CPU.
            logger.warning(f"TensorFlow runtime configuration failed; continuing with defaults: {e}")
            self.tf_runtime = None
        
        # LSTM hyperparameters
        self.sequence_length = config.get('lstm_sequence_length', 20)  # Look back 20 days
        self.lstm_units = config.get('lstm_units', 64)
        self.dropout_rate = config.get('lstm_dropout', 0.2)
        self.learning_rate = config.get('lstm_learning_rate', 0.001)
        self.batch_size = config.get('lstm_batch_size', 32)
        self.epochs = config.get('lstm_epochs', 50)
        
        # Same features as XGBoost for consistency
        self.feature_cols = [
            'momentum_12_1',
            'momentum_6m',
            'volatility_3m',
            'volume_trend',
            'mean_reversion',
        ]
    
    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute features (same as SimpleStrategy for consistency).
        
        Uses the same academically-backed features to ensure fair comparison.
        """
        df = df.copy()
        df = df.sort_index()
        
        grouped = df.groupby(level=1)
        
        # 1. Classic Momentum (12-1)
        ret_12m = grouped['close'].pct_change(252)
        ret_1m = grouped['close'].pct_change(21)
        df['momentum_12_1'] = ret_12m - ret_1m
        
        # 2. 6-month momentum
        df['momentum_6m'] = grouped['close'].pct_change(126)
        
        # 3. Volatility
        df['returns'] = grouped['close'].pct_change()
        df['volatility_3m'] = grouped['returns'].transform(lambda x: x.rolling(63).std())
        
        # 4. Volume trend
        vol_sma = grouped['volume'].transform(lambda x: x.rolling(21).mean())
        vol_sma_long = grouped['volume'].transform(lambda x: x.rolling(63).mean())
        df['volume_trend'] = vol_sma / vol_sma_long - 1
        
        # 5. Mean reversion
        df['mean_reversion'] = -grouped['close'].pct_change(5)
        
        return df.dropna()
    
    def _create_sequences(self, df: pd.DataFrame, target_col: str = 'target') -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM training.
        
        Args:
            df: DataFrame with MultiIndex (timestamp, symbol)
            target_col: Target column name
            
        Returns:
            (X, y) where X is (n_samples, sequence_length, n_features) and y is (n_samples,)
        """
        sequences = []
        targets = []
        
        # Group by symbol
        for symbol, group in df.groupby(level=1):
            group = group.sort_index()
            
            # Get features
            feature_data = group[self.feature_cols].values
            target_data = group[target_col].values if target_col in group.columns else None
            
            # Create sequences
            for i in range(len(feature_data) - self.sequence_length):
                seq = feature_data[i:i + self.sequence_length]
                sequences.append(seq)
                
                if target_data is not None:
                    targets.append(target_data[i + self.sequence_length])
        
        if len(sequences) == 0:
            return np.array([]), np.array([])
        
        X = np.array(sequences)
        y = np.array(targets) if targets else None
        
        return X, y
    
    def _build_model(self, input_shape: Tuple[int, int]) -> keras.Model:
        """Build LSTM model architecture."""
        model = keras.Sequential([
            layers.LSTM(self.lstm_units, return_sequences=True, input_shape=input_shape),
            layers.Dropout(self.dropout_rate),
            layers.LSTM(self.lstm_units // 2, return_sequences=False),
            layers.Dropout(self.dropout_rate),
            layers.Dense(32, activation='relu'),
            layers.Dense(1)  # Regression output
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train(self, df: pd.DataFrame, target_col: str = 'target', embargo_days: int = None) -> Dict:
        """
        Train LSTM model with leakage-safe cross-validation.
        
        Note: LSTM requires sequences, so we adapt the CV to work with sequences.
        """
        logger.info(f"Training LSTM model (horizon: {self.horizon})...")
        
        if embargo_days is None:
            embargo_map = {'1d': 1, '5d': 5, '20d': 20}
            embargo_days = embargo_map.get(self.horizon, 20)
        
        logger.info(f"Using embargo period: {embargo_days} days")
        logger.info(f"Using {len(self.feature_cols)} features: {self.feature_cols}")
        
        # Compute features if needed
        if 'momentum_12_1' not in df.columns:
            df = self.compute_features(df)
        
        # Ensure target exists
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")
        
        # Remove NaN
        df_clean = df.dropna(subset=self.feature_cols + [target_col])
        
        if len(df_clean) == 0:
            raise ValueError("No valid data after removing NaN")
        
        # Create sequences
        X, y = self._create_sequences(df_clean, target_col)
        
        if len(X) == 0:
            raise ValueError("Could not create sequences - need at least sequence_length days of data")
        
        logger.info(f"Created {len(X)} sequences (shape: {X.shape})")
        
        # For CV, we need to group by date to avoid leakage
        # Get date indices for each sequence
        date_indices = []
        for symbol, group in df_clean.groupby(level=1):
            group = group.sort_index()
            dates = group.index.get_level_values(0).values
            for i in range(len(group) - self.sequence_length):
                date_indices.append(dates[i + self.sequence_length])
        
        date_indices = pd.DatetimeIndex(date_indices)
        
        # Build model
        input_shape = (self.sequence_length, len(self.feature_cols))
        model = self._build_model(input_shape)
        
        # Simple train/validation split for now (can enhance with proper CV later)
        # Use last 20% for validation
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Train
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )

        # Hard time limit for training to avoid stalling weekly job
        # Default: 10 minutes; configurable via config['strategy']['lstm_max_train_seconds']
        max_train_seconds = (
            self.config.get('strategy', {}).get('lstm_max_train_seconds')
            if isinstance(self.config, dict) else None
        )
        if max_train_seconds is None:
            max_train_seconds = 600

        class _TimeLimitCallback(keras.callbacks.Callback):
            def __init__(self, max_seconds: int):
                super().__init__()
                self.max_seconds = max_seconds
                self.start = None
                self.timed_out = False

            def on_train_begin(self, logs=None):
                self.start = time.monotonic()

            def on_epoch_end(self, epoch, logs=None):
                if self.start is None:
                    return
                if (time.monotonic() - self.start) > self.max_seconds:
                    self.timed_out = True
                    self.model.stop_training = True

        time_limit_cb = _TimeLimitCallback(int(max_train_seconds))
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=self.batch_size,
            epochs=self.epochs,
            callbacks=[early_stopping, time_limit_cb],
            verbose=0
        )

        if getattr(time_limit_cb, 'timed_out', False):
            # Do not save partial model; let caller skip LSTM and continue with XGBoost/momentum.
            raise TimeoutError(f"LSTM training exceeded {max_train_seconds}s; skipping LSTM strategy")
        
        # Calculate IC (Information Coefficient) for consistency with XGBoost
        val_pred = model.predict(X_val, verbose=0).flatten()
        ic = np.corrcoef(val_pred, y_val)[0, 1] if len(y_val) > 1 else 0.0
        
        # Save model
        model_path = os.path.join(self.model_dir, f'lstm_{self.horizon}.h5')
        model.save(model_path)
        
        # Save feature columns
        features_path = os.path.join(self.model_dir, f'lstm_features_{self.horizon}.pkl')
        joblib.dump(self.feature_cols, features_path)
        
        logger.info(f"Model saved: {model_path}")
        logger.info(f"Validation IC: {ic:.4f}")
        
        self.model = model
        
        return {
            'cv_mean_ic': ic,
            'cv_std_ic': 0.0,  # Single fold for now
            'n_samples': len(X_train),
            'n_features': len(self.feature_cols),
            'embargo_days': embargo_days,
            'val_loss': float(min(history.history['val_loss']))
        }
    
    def load_model(self, horizon: str = None) -> bool:
        """Load trained LSTM model."""
        if horizon:
            self.horizon = horizon
        
        try:
            model_path = os.path.join(self.model_dir, f'lstm_{self.horizon}.h5')
            features_path = os.path.join(self.model_dir, f'lstm_features_{self.horizon}.pkl')
            
            if os.path.exists(model_path):
                self.model = keras.models.load_model(model_path)
                if os.path.exists(features_path):
                    self.feature_cols = joblib.load(features_path)
                logger.info(f"Loaded LSTM {self.horizon} model from {model_path}")
                return True
        except Exception as e:
            logger.error(f"Error loading LSTM {self.horizon} model: {e}")
        return False
    
    def compute_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate signals using LSTM model."""
        if self.model is None:
            if not self.load_model():
                raise ValueError("LSTM model not trained or loaded")
        
        # Compute features if needed
        if 'momentum_12_1' not in df.columns:
            df = self.compute_features(df)
        
        # Get latest sequence for each ticker
        signals_list = []
        
        for symbol, group in df.groupby(level=1):
            group = group.sort_index().tail(self.sequence_length)
            
            if len(group) < self.sequence_length:
                continue  # Skip if not enough history
            
            # Get features
            feature_data = group[self.feature_cols].values
            
            # Pad or truncate to sequence_length
            if len(feature_data) < self.sequence_length:
                # Pad with last value
                padding = np.tile(feature_data[-1:], (self.sequence_length - len(feature_data), 1))
                feature_data = np.vstack([padding, feature_data])
            elif len(feature_data) > self.sequence_length:
                feature_data = feature_data[-self.sequence_length:]
            
            # Reshape for LSTM: (1, sequence_length, n_features)
            X_seq = feature_data.reshape(1, self.sequence_length, len(self.feature_cols))
            
            # Predict
            score = self.model.predict(X_seq, verbose=0)[0, 0]
            
            signals_list.append({
                'ticker': symbol,
                'score': float(score)
            })
        
        if len(signals_list) == 0:
            logger.warning("LSTMStrategy: No valid signals generated")
            return pd.DataFrame(columns=['score', 'rank'])
        
        signals = pd.DataFrame(signals_list)
        signals['rank'] = signals['score'].rank(ascending=False)
        return signals.set_index('ticker').sort_values('rank')

