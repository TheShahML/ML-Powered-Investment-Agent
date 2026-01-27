"""
Strategy Selector - Dynamically switches between strategies based on performance.

Uses promotion gate results to select the best performing strategy.
"""
import pandas as pd
from loguru import logger
from typing import Dict, Optional, List
from .strategy_base import BaseStrategy
from .strategy_simple import SimpleStrategy
from .strategy_simple import PureMomentumStrategy


class StrategySelector:
    """
    Selects and manages multiple strategies dynamically.
    
    Tracks performance of each strategy and switches to the best one
    based on promotion gate results.
    """
    
    def __init__(self, config: Dict, model_dir: str = 'models'):
        self.config = config
        self.model_dir = model_dir
        self.available_strategies = {}
        self.current_strategy: Optional[BaseStrategy] = None
        self.strategy_performance = {}  # Track performance history
        
        # Register available strategies
        self._register_strategies()
    
    def _register_strategies(self):
        """Register all available strategies."""
        strategy_config = self.config.get('strategy', {})
        
        # Always available: Simple XGBoost
        self.available_strategies['simple'] = {
            'class': SimpleStrategy,
            'name': 'XGBoost Multi-Horizon',
            'description': 'Gradient boosted trees with 5 features'
        }
        
        # LSTM disabled for now (can be re-enabled later)
        # If you want to re-enable, add it back here and ensure the runtime has GPU/TF configured.
        
        # Pure momentum baseline
        self.available_strategies['pure_momentum'] = {
            'class': PureMomentumStrategy,
            'name': 'Pure Momentum',
            'description': 'Baseline momentum strategy (no ML)'
        }
    
    def get_strategy(self, strategy_type: str = None, horizon: str = '20d') -> BaseStrategy:
        """
        Get a strategy instance.
        
        Args:
            strategy_type: 'simple' or 'pure_momentum'. If None, uses config default.
            horizon: '1d', '5d', or '20d'
            
        Returns:
            Strategy instance
        """
        if strategy_type is None:
            strategy_type = self.config.get('strategy', {}).get('strategy_type', 'simple')
        
        if strategy_type not in self.available_strategies:
            logger.warning(f"Strategy '{strategy_type}' not available, defaulting to 'simple'")
            strategy_type = 'simple'
        
        strategy_info = self.available_strategies[strategy_type]
        strategy_class = strategy_info['class']
        
        return strategy_class(self.config, self.model_dir, horizon)
    
    def select_best_strategy(
        self,
        strategy_results: Dict[str, Dict],
        baselines: Dict[str, Dict]
    ) -> str:
        """
        Select the best strategy based on promotion gate results.
        
        Args:
            strategy_results: Dict mapping strategy_type -> {sharpe, cagr, max_drawdown, ...}
            baselines: Dict mapping baseline_name -> {sharpe, cagr, max_drawdown, ...}
            
        Returns:
            Best strategy type name
        """
        if not strategy_results:
            logger.warning("No strategy results provided, defaulting to 'simple'")
            return 'simple'
        
        # Get best baseline for comparison
        baseline_sharpes = [m.get('sharpe', 0) for m in baselines.values() if 'sharpe' in m]
        best_baseline_sharpe = max(baseline_sharpes) if baseline_sharpes else 0
        
        # Score each strategy
        strategy_scores = {}
        
        for strategy_type, results in strategy_results.items():
            sharpe = results.get('sharpe', 0)
            maxdd = results.get('max_drawdown', -1.0)
            
            # Score = Sharpe improvement over baseline - penalty for worse MaxDD
            sharpe_improvement = sharpe - best_baseline_sharpe
            maxdd_penalty = max(0, maxdd - (-0.1))  # Penalize if MaxDD worse than -10%
            
            score = sharpe_improvement - (maxdd_penalty * 0.5)  # Weight MaxDD penalty
            
            strategy_scores[strategy_type] = {
                'score': score,
                'sharpe': sharpe,
                'maxdd': maxdd,
                'sharpe_improvement': sharpe_improvement
            }
        
        # Select best
        best_strategy = max(strategy_scores.items(), key=lambda x: x[1]['score'])
        best_type, best_metrics = best_strategy
        
        logger.info(f"Selected best strategy: {best_type}")
        logger.info(f"  Score: {best_metrics['score']:.3f}")
        logger.info(f"  Sharpe: {best_metrics['sharpe']:.3f} (improvement: {best_metrics['sharpe_improvement']:+.3f})")
        logger.info(f"  MaxDD: {best_metrics['maxdd']:.3f}")
        
        return best_type
    
    def update_performance(self, strategy_type: str, metrics: Dict):
        """Update performance tracking for a strategy."""
        if strategy_type not in self.strategy_performance:
            self.strategy_performance[strategy_type] = []
        
        self.strategy_performance[strategy_type].append(metrics)
    
    def get_performance_summary(self) -> pd.DataFrame:
        """Get performance summary of all strategies."""
        if not self.strategy_performance:
            return pd.DataFrame()
        
        summaries = []
        for strategy_type, history in self.strategy_performance.items():
            if not history:
                continue
            
            latest = history[-1]
            summaries.append({
                'strategy': strategy_type,
                'sharpe': latest.get('sharpe', 0),
                'cagr': latest.get('cagr', 0),
                'maxdd': latest.get('max_drawdown', 0),
                'n_evaluations': len(history)
            })
        
        return pd.DataFrame(summaries).sort_values('sharpe', ascending=False)

