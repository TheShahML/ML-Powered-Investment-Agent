import alpaca_trade_api as tradeapi
from loguru import logger
from typing import Dict, List
import pandas as pd
from decimal import Decimal

class ExecutionService:
    def __init__(self, config):
        self.config = config
        self.api = tradeapi.REST(
            self.config['ALPACA_API_KEY'],
            self.config['ALPACA_SECRET_KEY'],
            self.config['ALPACA_BASE_URL']
        )
        self.mode = self.config.get('BROKER_MODE', 'paper')
        self.live_ack = self.config.get('I_ACKNOWLEDGE_LIVE_TRADING', False)
        
        # Verify safety for live trading
        if self.mode == 'live' and not self.live_ack:
            raise ValueError("CRITICAL: BROKER_MODE is 'live' but I_ACKNOWLEDGE_LIVE_TRADING is false. Aborting.")
        
        logger.info(f"Execution Service initialized in {self.mode.upper()} mode.")

    def get_account_info(self) -> Dict:
        account = self.api.get_account()
        return {
            'equity': float(account.equity),
            'cash': float(account.cash),
            'buying_power': float(account.buying_power)
        }

    def get_positions(self) -> pd.DataFrame:
        positions = self.api.list_positions()
        if not positions:
            return pd.DataFrame()
        
        data = []
        for p in positions:
            data.append({
                'ticker': p.symbol,
                'qty': float(p.qty),
                'market_value': float(p.market_value),
                'cost_basis': float(p.cost_basis),
                'unrealized_pl': float(p.unrealized_pl)
            })
        return pd.DataFrame(data).set_index('ticker')

    def execute_rebalance(self, target_weights: Dict[str, float], dry_run: bool = False):
        """
        Executes trades to match target weights.
        target_weights: ticker -> weight
        """
        account = self.get_account_info()
        equity = account['equity']
        current_positions = self.get_positions()
        
        logger.info(f"Starting rebalance for equity: ${equity:.2f}")
        
        # 1. Calculate target notional per ticker
        target_notionals = {ticker: weight * equity for ticker, weight in target_weights.items()}
        
        # 2. Determine trades
        # Sell tickers not in targets or if weight decreased significantly
        for ticker, pos in current_positions.iterrows():
            if ticker not in target_notionals:
                logger.info(f"Selling entire position in {ticker}")
                if not dry_run:
                    self._submit_order(ticker, qty=pos['qty'], side='sell')
            else:
                # Optional: optimize by only trading if delta > threshold
                pass

        # 3. Buy/Adjust positions
        for ticker, target_value in target_notionals.items():
            current_value = current_positions.loc[ticker, 'market_value'] if ticker in current_positions.index else 0
            delta = target_value - current_value
            
            if abs(delta) < self.config.get('min_trade_notional', 2.0):
                logger.debug(f"Skipping trade for {ticker}: delta ${delta:.2f} below threshold")
                continue
                
            if delta > 0:
                logger.info(f"Buying ${delta:.2f} of {ticker}")
                if not dry_run:
                    self._submit_order(ticker, notional=delta, side='buy')
            elif delta < 0:
                # Sell part of position
                # Alpaca doesn't support 'notional' for sells usually, need qty
                # But for rebalance we might just sell the whole thing and rebuy or calc qty
                price = self.api.get_latest_trade(ticker).price
                qty_to_sell = abs(delta) / price
                logger.info(f"Selling {qty_to_sell:.4f} shares of {ticker} (${abs(delta):.2f})")
                if not dry_run:
                    self._submit_order(ticker, qty=qty_to_sell, side='sell')

    def _submit_order(self, ticker: str, qty: float = None, notional: float = None, side: str = 'buy'):
        try:
            params = {
                'symbol': ticker,
                'side': side,
                'type': self.config.get('order_type', 'market'),
                'time_in_force': 'day'
            }
            if notional:
                params['notional'] = round(notional, 2)
            elif qty:
                params['qty'] = round(qty, 4) # Fractional support
            
            logger.info(f"Submitting {side} order for {ticker}: {params}")
            order = self.api.submit_order(**params)
            return order
        except Exception as e:
            logger.error(f"Error submitting order for {ticker}: {e}")
            return None



