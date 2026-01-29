# Dynamic Strategy Switching

This document explains how to use the new dynamic strategy switching system that allows you to train and compare multiple ML strategies, including neural networks.

## Overview

The system now supports:
1. **Multiple Strategies**: XGBoost, LSTM Neural Networks, Pure Momentum
2. **Dynamic Switching**: Automatically selects the best performing strategy
3. **Unified Interface**: All strategies implement the same API for easy comparison

## Available Strategies

### 1. XGBoost (Simple Strategy)
- **Type**: `simple`
- **Model**: Gradient Boosted Trees
- **Features**: 5 academically-backed features
- **Pros**: Fast training, interpretable, proven performance
- **Cons**: May miss complex temporal patterns

### 2. LSTM Neural Network
- **Type**: `lstm`
- **Model**: Long Short-Term Memory network
- **Features**: Same 5 features, but learns temporal sequences
- **Pros**: Can learn complex temporal patterns, good for time series
- **Cons**: Slower training, requires TensorFlow, more hyperparameters

### 3. Pure Momentum (Baseline)
- **Type**: `pure_momentum`
- **Model**: Rule-based (no ML)
- **Features**: None (uses price momentum directly)
- **Pros**: Zero overfitting, fast, academic baseline
- **Cons**: No ML enhancement

## Installation

### Basic Setup (XGBoost only)
```bash
pip install -r requirements.txt
```

### With LSTM Support
```bash
pip install -r requirements.txt
pip install tensorflow>=2.10.0
```

## Usage

### Training Multiple Strategies

Train and compare all available strategies:

```bash
python scripts/train_multiple_strategies.py
```

This will:
1. Train XGBoost models (1d, 5d, 20d horizons)
2. Train LSTM models (if TensorFlow installed)
3. Run walk-forward backtests for each
4. Compare against Pure Momentum baseline
5. Select the best performing strategy

### Using a Specific Strategy

In your code:

```python
from src.strategy_selector import StrategySelector
from src.config import load_config

config = load_config()
selector = StrategySelector(config)

# Get XGBoost strategy
strategy = selector.get_strategy('simple', horizon='20d')

# Get LSTM strategy
strategy = selector.get_strategy('lstm', horizon='20d')

# Get Pure Momentum
strategy = selector.get_strategy('pure_momentum', horizon='20d')
```

### Dynamic Strategy Selection

The system can automatically select the best strategy based on backtest performance:

```python
from src.strategy_selector import StrategySelector

selector = StrategySelector(config)

# Results from backtests
strategy_results = {
    'simple': {'sharpe': 2.31, 'max_drawdown': -0.202, 'cagr': 1.063},
    'lstm': {'sharpe': 2.45, 'max_drawdown': -0.185, 'cagr': 1.12}
}

baselines = {
    'pure_momentum': {'sharpe': 2.55, 'max_drawdown': -0.093}
}

# Select best
best_strategy = selector.select_best_strategy(strategy_results, baselines)
# Returns: 'lstm' (if it beats XGBoost)
```

## Configuration

Edit `config/strategy.yaml`:

```yaml
strategy:
  strategy_type: "simple"  # Default strategy
  enable_dynamic_switching: false  # Set to true for auto-switching
  
  # LSTM-specific settings
  lstm_sequence_length: 20  # Look back 20 days
  lstm_units: 64  # LSTM units
  lstm_dropout: 0.2
  lstm_learning_rate: 0.001
  lstm_batch_size: 32
  lstm_epochs: 50
```

## Strategy Comparison

The system compares strategies on:
1. **Sharpe Ratio**: Risk-adjusted returns
2. **CAGR**: Compound annual growth rate
3. **Max Drawdown**: Maximum peak-to-trough decline
4. **Information Coefficient (IC)**: Prediction accuracy

## Promotion Gate

Each strategy must pass the promotion gate:
- **Sharpe Margin**: Must beat baseline by +0.20
- **MaxDD Tolerance**: Can be at most +0.05 worse than baseline

If a strategy fails, the system continues using the last approved strategy.

## Adding New Strategies

To add a new strategy:

1. Create a new file: `src/strategy_yourname.py`
2. Inherit from `BaseStrategy`:

```python
from src.strategy_base import BaseStrategy

class YourStrategy(BaseStrategy):
    def compute_features(self, df):
        # Your feature computation
        pass
    
    def train(self, df, target_col, embargo_days):
        # Your training logic
        pass
    
    def compute_signals(self, df):
        # Your signal generation
        pass
    
    def load_model(self, horizon):
        # Your model loading
        pass
```

3. Register in `StrategySelector._register_strategies()`

## Performance Notes

- **XGBoost**: ~1-2 minutes per horizon
- **LSTM**: ~5-10 minutes per horizon (depends on GPU)
- **Pure Momentum**: Instant (no training)

## References

- LSTM architecture based on: https://github.com/pranityadav19/Stock-Prediction-Models
- Original repo: https://github.com/huseinzol05/Stock-Prediction-Models





