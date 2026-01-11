import pandas as pd
import numpy as np
from loguru import logger

class Backtester:
    def __init__(self, config):
        self.config = config
        self.slippage_bps = config.get('slippage_bps', 10)
        self.initial_capital = config.get('initial_capital', 1000)
        
    def run(self, data: pd.DataFrame, strategy) -> pd.DataFrame:
        """
        Runs the backtest on historical data.
        data: Multi-index (timestamp, symbol) with 'close' and 'open'.
        """
        logger.info("Starting backtest...")
        
        # Get unique trading days (Mondays and Tuesdays)
        # In reality, we filter for the rebalance cycle.
        # We assume 'data' is EOD bars.
        
        dates = data.index.get_level_values(0).unique().sort_values()
        
        # Identify rebalance dates (Mondays) and execution dates (Tuesdays)
        # This is simplified: in practice, we'd handle holidays.
        rebalance_dates = dates[dates.weekday == 0] # Monday
        
        portfolio_value = self.initial_capital
        holdings = {} # symbol -> quantity
        history = []
        
        for i, rebalance_date in enumerate(rebalance_dates):
            # 1. Compute signals as of Monday close
            # We need data up to this Monday
            historical_data = data.loc[data.index.get_level_values(0) <= rebalance_date]
            signals = strategy.compute_signals(historical_data)
            
            # Select top N
            top_n = signals.head(self.config['n_holdings']).index.tolist()
            
            # 2. Execution on Tuesday open
            # Find the next available date after Monday
            try:
                execution_date = dates[dates > rebalance_date][0]
                if execution_date.weekday() != 1: # Not a Tuesday
                    logger.warning(f"Next trading day after {rebalance_date} is not Tuesday: {execution_date}")
            except IndexError:
                break
                
            # Current prices at Tuesday open
            open_prices = data.loc[execution_date]['open']
            
            # Rebalance logic
            target_weight = 1.0 / self.config['n_holdings']
            max_weight = self.config.get('max_position_weight', 0.12)
            weight = min(target_weight, max_weight)
            
            # Calculate target quantities
            target_holdings = {}
            cash_buffer = portfolio_value * (1 - self.config.get('always_invested_threshold', 0.95))
            investable_capital = portfolio_value - cash_buffer
            
            for ticker in top_n:
                if ticker in open_prices:
                    price = open_prices[ticker]
                    # Apply slippage to buy price
                    buy_price = price * (1 + self.slippage_bps / 10000)
                    qty = (investable_capital * weight) / buy_price
                    target_holdings[ticker] = qty
            
            # Simplified rebalance: sell all, buy new (could be optimized)
            # In a real model, we'd only trade the difference.
            
            # Calculate turnover and costs
            # (Skipping detailed cost accounting for MVP, but accounting for slippage)
            
            holdings = target_holdings
            
            # 3. Update portfolio value as of Tuesday close or next rebalance
            # For simplicity, we track value at the next rebalance date
            if i + 1 < len(rebalance_dates):
                next_rebalance = rebalance_dates[i+1]
                next_close_prices = data.loc[next_rebalance]['close']
                
                new_value = 0
                for ticker, qty in holdings.items():
                    if ticker in next_close_prices:
                        new_value += qty * next_close_prices[ticker]
                
                # Add back the cash buffer (simplified)
                portfolio_value = new_value + cash_buffer
                
                history.append({
                    'date': next_rebalance,
                    'portfolio_value': portfolio_value
                })
        
        return pd.DataFrame(history).set_index('date')



