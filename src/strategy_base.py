"""
Base Strategy Interface

All strategies must implement this interface to enable dynamic switching.
"""
from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, List, Tuple
from loguru import logger


class BaseStrategy(ABC):
    """
    Base class for all trading strategies.
    
    All strategies must implement:
    - compute_features: Transform raw price/volume data into features
    - train: Train the model with leakage-safe cross-validation
    - compute_signals: Generate stock rankings/signals
    - load_model: Load a previously trained model
    """
    
    def __init__(self, config: Dict, model_dir: str = 'models', horizon: str = '20d'):
        self.config = config
        self.model_dir = model_dir
        self.horizon = horizon
        self.model = None
        self.feature_cols = []
    
    @abstractmethod
    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute features from raw price/volume data.
        
        Args:
            df: DataFrame with MultiIndex (timestamp, symbol) and columns: open, high, low, close, volume
            
        Returns:
            DataFrame with computed features added
        """
        pass
    
    @abstractmethod
    def train(self, df: pd.DataFrame, target_col: str = 'target', embargo_days: int = None) -> Dict:
        """
        Train the model with leakage-safe cross-validation.
        
        Args:
            df: Training data with MultiIndex (timestamp, symbol) and features + target
            target_col: Name of target column
            embargo_days: Days to embargo between train/test
            
        Returns:
            Dict with CV metrics (must include 'cv_mean_ic' and 'cv_std_ic')
        """
        pass
    
    @abstractmethod
    def compute_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals (stock rankings).
        
        Args:
            df: DataFrame with features computed
            
        Returns:
            DataFrame with index=ticker, columns=['score', 'rank']
        """
        pass
    
    @abstractmethod
    def load_model(self, horizon: str = None) -> bool:
        """
        Load a previously trained model.
        
        Args:
            horizon: Optional horizon override
            
        Returns:
            True if loaded successfully, False otherwise
        """
        pass
    
    def get_strategy_name(self) -> str:
        """Return strategy name for identification."""
        return self.__class__.__name__
    
    def compute_multi_horizon_signals(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Generate signals from all 3 time horizons (optional method).
        
        Default implementation uses the primary horizon only.
        Strategies can override to provide multi-horizon signals.
        
        Returns:
            Dict with keys '1d', '5d', '20d' mapping to signal DataFrames
        """
        # Default: use primary horizon for all
        signals = {}
        primary_horizon = self.horizon
        
        for horizon in ['1d', '5d', '20d']:
            if self.load_model(horizon):
                signals[horizon] = self.compute_signals(df)
            else:
                # Fallback: use primary horizon model
                if horizon == primary_horizon and self.load_model(primary_horizon):
                    signals[horizon] = self.compute_signals(df)
        
        return signals

