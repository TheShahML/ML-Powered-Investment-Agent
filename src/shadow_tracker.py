"""
Shadow portfolio tracker for ML, canary, and benchmarks.
Tracks performance without placing actual orders (except ML actual portfolio).
"""
import pandas as pd
import numpy as np
from loguru import logger
from typing import Dict, List, Optional, Tuple
from datetime import date, datetime, timedelta
from .portfolio_constructor import PortfolioConstructor


class ShadowPortfolioTracker:
    """
    Track shadow portfolios for ML, canary, and benchmarks.
    
    Maintains:
    - ML shadow portfolio (weights/holdings and PnL)
    - Canary momentum portfolio (weights/holdings and PnL)
    - Benchmarks: SPY, BTC (if enabled)
    
    State persisted in latest_state.json:
    - last_rebalance_date
    - current shadow holdings/weights for ML and canary
    - daily equity curve points (rolling 90 days)
    - turnover and cost-drag at rebalance
    - rolling 30d/90d returns
    """

    def __init__(self, config: Dict):
        self.config = config
        self.top_n = config.get('top_n', 25)
        self.cost_bps = config.get('cost_bps', 15.0)
        
        # Portfolio constructors
        self.ml_constructor = PortfolioConstructor(
            top_n=self.top_n,
            max_weight=0.10,
            turnover_buffer_pct=1.0,
            regime_filter=True
        )
        
        self.canary_constructor = PortfolioConstructor(
            top_n=self.top_n,
            max_weight=0.10,
            turnover_buffer_pct=1.0,
            regime_filter=True
        )

    def initialize_from_state(self, state: Dict) -> Dict:
        """Initialize tracker state from latest_state.json."""
        shadow = state.get('shadow_portfolios', {})
        
        ml_shadow = shadow.get('ml', {
            'holdings': {},
            'weights': {},
            'equity_curve': [],
            'dates': [],
            'last_rebalance_date': None,
            'initial_value': 100000.0
        })
        
        canary_shadow = shadow.get('canary', {
            'holdings': {},
            'weights': {},
            'equity_curve': [],
            'dates': [],
            'last_rebalance_date': None,
            'initial_value': 100000.0
        })
        
        benchmarks = shadow.get('benchmarks', {
            'spy': {'equity_curve': [], 'dates': []},
            'btc': {'equity_curve': [], 'dates': []}
        })
        
        return {
            'ml': ml_shadow,
            'canary': canary_shadow,
            'benchmarks': benchmarks
        }

    def update_daily(
        self,
        shadow_state: Dict,
        as_of_date: date,
        prices: Dict[str, float],
        spy_price: Optional[float] = None,
        btc_price: Optional[float] = None
    ) -> Dict:
        """
        Update shadow portfolios with daily prices.
        
        Args:
            shadow_state: Current shadow state
            as_of_date: Date for update
            prices: {symbol: price} for all holdings
            spy_price: SPY price (if available)
            btc_price: BTC price (if available)
        
        Returns:
            Updated shadow_state
        """
        # Update ML shadow
        ml_value = self._mark_to_market(shadow_state['ml'], prices)
        shadow_state['ml']['equity_curve'].append(ml_value)
        shadow_state['ml']['dates'].append(as_of_date.isoformat())
        
        # Update canary shadow
        canary_value = self._mark_to_market(shadow_state['canary'], prices)
        shadow_state['canary']['equity_curve'].append(canary_value)
        shadow_state['canary']['dates'].append(as_of_date.isoformat())
        
        # Update benchmarks
        if spy_price:
            spy_state = shadow_state['benchmarks']['spy']
            if not spy_state['equity_curve']:
                spy_state['initial_value'] = spy_price
            spy_normalized = spy_price / spy_state.get('initial_value', spy_price)
            spy_state['equity_curve'].append(spy_normalized)
            spy_state['dates'].append(as_of_date.isoformat())
        
        if btc_price:
            btc_state = shadow_state['benchmarks']['btc']
            if not btc_state['equity_curve']:
                btc_state['initial_value'] = btc_price
            btc_normalized = btc_price / btc_state.get('initial_value', btc_price)
            btc_state['equity_curve'].append(btc_normalized)
            btc_state['dates'].append(as_of_date.isoformat())
        
        # Keep only last 90 days
        for portfolio in [shadow_state['ml'], shadow_state['canary']]:
            if len(portfolio['equity_curve']) > 90:
                portfolio['equity_curve'] = portfolio['equity_curve'][-90:]
                portfolio['dates'] = portfolio['dates'][-90:]
        
        for bench in shadow_state['benchmarks'].values():
            if len(bench['equity_curve']) > 90:
                bench['equity_curve'] = bench['equity_curve'][-90:]
                bench['dates'] = bench['dates'][-90:]
        
        return shadow_state

    def rebalance_shadow(
        self,
        shadow_state: Dict,
        ml_signals: pd.DataFrame,
        canary_signals: pd.DataFrame,
        data: pd.DataFrame,
        as_of_date: date,
        spy_data: Optional[pd.DataFrame] = None
    ) -> Dict:
        """
        Rebalance shadow portfolios.
        
        Args:
            shadow_state: Current shadow state
            ml_signals: ML model signals
            canary_signals: Canary momentum signals
            data: Historical data
            as_of_date: Rebalance date
            spy_data: SPY data for regime filter
        
        Returns:
            Updated shadow_state with rebalance info
        """
        # Get current portfolio values
        ml_current_value = shadow_state['ml'].get('equity_curve', [100000.0])[-1]
        canary_current_value = shadow_state['canary'].get('equity_curve', [100000.0])[-1]
        
        # Get current weights
        ml_current_weights = shadow_state['ml'].get('weights', {})
        canary_current_weights = shadow_state['canary'].get('weights', {})
        
        # Compute target weights
        ml_targets, ml_metadata = self.ml_constructor.compute_target_weights(
            ml_signals,
            data,
            as_of_date,
            current_weights=ml_current_weights,
            spy_data=spy_data
        )
        
        canary_targets, canary_metadata = self.canary_constructor.compute_target_weights(
            canary_signals,
            data,
            as_of_date,
            current_weights=canary_current_weights,
            spy_data=spy_data
        )
        
        # Compute turnover
        ml_turnover = self.ml_constructor.compute_turnover(
            ml_targets, ml_current_weights, ml_current_value
        )
        
        canary_turnover = self.canary_constructor.compute_turnover(
            canary_targets, canary_current_weights, canary_current_value
        )
        
        # Apply transaction costs
        ml_cost = ml_turnover['turnover_dollars'] * (self.cost_bps / 10000.0)
        canary_cost = canary_turnover['turnover_dollars'] * (self.cost_bps / 10000.0)
        
        # Update shadow state
        shadow_state['ml']['weights'] = ml_targets
        shadow_state['ml']['holdings'] = ml_targets  # Simplified - weights represent holdings
        shadow_state['ml']['last_rebalance_date'] = as_of_date.isoformat()
        shadow_state['ml']['last_turnover'] = ml_turnover['turnover_pct']
        shadow_state['ml']['last_cost'] = ml_cost
        
        shadow_state['canary']['weights'] = canary_targets
        shadow_state['canary']['holdings'] = canary_targets
        shadow_state['canary']['last_rebalance_date'] = as_of_date.isoformat()
        shadow_state['canary']['last_turnover'] = canary_turnover['turnover_pct']
        shadow_state['canary']['last_cost'] = canary_cost
        
        logger.info(
            f"Shadow rebalance at {as_of_date}: "
            f"ML turnover {ml_turnover['turnover_pct']:.1%}, cost ${ml_cost:.2f}; "
            f"Canary turnover {canary_turnover['turnover_pct']:.1%}, cost ${canary_cost:.2f}"
        )
        
        return shadow_state

    def _mark_to_market(
        self,
        portfolio_state: Dict,
        prices: Dict[str, float]
    ) -> float:
        """Mark portfolio to market."""
        weights = portfolio_state.get('weights', {})
        if not weights:
            # Return last value if no holdings
            equity_curve = portfolio_state.get('equity_curve', [])
            return equity_curve[-1] if equity_curve else portfolio_state.get('initial_value', 100000.0)
        
        # Compute value from weights (assuming normalized to 1.0)
        total_value = 0.0
        initial_value = portfolio_state.get('initial_value', 100000.0)
        
        # Get last known value to scale from
        equity_curve = portfolio_state.get('equity_curve', [])
        last_value = equity_curve[-1] if equity_curve else initial_value
        
        # For shadow, we track relative performance
        # Assume we rebalance to target weights, so value scales with weighted average return
        # Simplified: track as if we hold equal dollar amounts
        for symbol, weight in weights.items():
            if symbol in prices:
                # This is simplified - in reality we'd track actual shares
                total_value += weight * prices[symbol]
        
        # Normalize to maintain continuity
        if total_value > 0:
            return last_value * (total_value / sum(weights.values())) if weights else last_value
        
        return last_value

    def compute_performance_metrics(
        self,
        shadow_state: Dict,
        as_of_date: date
    ) -> Dict:
        """
        Compute performance metrics.
        
        Returns:
            Dict with returns, Sharpe, etc.
        """
        ml_curve = shadow_state['ml'].get('equity_curve', [])
        canary_curve = shadow_state['canary'].get('equity_curve', [])
        spy_curve = shadow_state['benchmarks']['spy'].get('equity_curve', [])
        btc_curve = shadow_state['benchmarks']['btc'].get('equity_curve', [])
        
        metrics = {}
        
        # Since last rebalance
        ml_last_rebal = shadow_state['ml'].get('last_rebalance_date')
        if ml_last_rebal and ml_curve:
            try:
                rebal_date = datetime.fromisoformat(ml_last_rebal).date()
                # Find index at rebalance
                dates = [datetime.fromisoformat(d).date() for d in shadow_state['ml'].get('dates', [])]
                if rebal_date in dates:
                    rebal_idx = dates.index(rebal_date)
                    if len(ml_curve) > rebal_idx:
                        ml_since = (ml_curve[-1] / ml_curve[rebal_idx] - 1) * 100
                        canary_since = (canary_curve[-1] / canary_curve[rebal_idx] - 1) * 100 if len(canary_curve) > rebal_idx else 0.0
                        spy_since = (spy_curve[-1] / spy_curve[rebal_idx] - 1) * 100 if len(spy_curve) > rebal_idx else 0.0
                        
                        metrics['since_rebalance'] = {
                            'ml': ml_since,
                            'canary': canary_since,
                            'spy': spy_since,
                            'days': (as_of_date - rebal_date).days
                        }
            except:
                pass
        
        # Rolling 30d/90d returns
        if len(ml_curve) >= 30:
            ml_30d = (ml_curve[-1] / ml_curve[-30] - 1) * 100
            canary_30d = (canary_curve[-1] / canary_curve[-30] - 1) * 100 if len(canary_curve) >= 30 else 0.0
            spy_30d = (spy_curve[-1] / spy_curve[-30] - 1) * 100 if len(spy_curve) >= 30 else 0.0
            
            metrics['rolling_30d'] = {
                'ml': ml_30d,
                'canary': canary_30d,
                'spy': spy_30d
            }
        
        if len(ml_curve) >= 90:
            ml_90d = (ml_curve[-1] / ml_curve[-90] - 1) * 100
            canary_90d = (canary_curve[-1] / canary_curve[-90] - 1) * 100 if len(canary_curve) >= 90 else 0.0
            spy_90d = (spy_curve[-1] / spy_curve[-90] - 1) * 100 if len(spy_curve) >= 90 else 0.0
            
            metrics['rolling_90d'] = {
                'ml': ml_90d,
                'canary': canary_90d,
                'spy': spy_90d
            }
        
        return metrics

