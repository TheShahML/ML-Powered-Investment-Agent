import alpaca_trade_api as tradeapi
import pandas as pd
import datetime
from loguru import logger
from typing import List, Optional
import time

class DataService:
    def __init__(self, config):
        self.config = config
        self.api = tradeapi.REST(
            self.config['ALPACA_API_KEY'],
            self.config['ALPACA_SECRET_KEY'],
            self.config['ALPACA_BASE_URL']
        )

    def get_universe(self) -> List[str]:
        """
        Fetches the tradeable universe based on filters in config.
        """
        logger.info("Fetching tradeable universe...")
        
        # In a real scenario, we might use a third-party API or Alpaca's assets endpoint 
        # but Alpaca assets don't include market cap/volume easily.
        # For this implementation, we will use Alpaca's active assets and filter them.
        # Note: Filtering by market cap/volume usually requires a separate data provider 
        # (like Polygon, IEX, or YFinance). For Alpaca-only, we'll demonstrate the logic.
        
        assets = self.api.list_assets(status='active', asset_class='us_equity')
        
        # Filter for tradeable and fractional if required
        universe = [
            a.symbol for a in assets 
            if a.tradable and a.shortable and a.marginable
            # Alpaca-specific: check for fractional support if configured
            and (not self.config.get('fractional_shares') or a.fractionable)
        ]
        
        # TODO: Implement strict filtering (Cap > $10B, Vol > $50M) 
        # This typically requires an external data source or Alpaca's Screener API if available.
        # For now, we'll return a subset or placeholder logic.
        logger.info(f"Initial universe size: {len(universe)}")
        return universe[:self.config.get('target_size', 500)]

    def get_historical_data(self, tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetches historical EOD bars for the given tickers.
        """
        logger.info(f"Fetching historical data for {len(tickers)} tickers from {start_date} to {end_date}...")
        
        all_bars = []
        # Chunk tickers to avoid long URLs/timeouts
        chunk_size = 50
        for i in range(0, len(tickers), chunk_size):
            chunk = tickers[i:i + chunk_size]
            try:
                bars = self.api.get_bars(
                    chunk, 
                    tradeapi.rest.TimeFrame.Day, 
                    start_date, 
                    end_date, 
                    adjustment='all'
                ).df
                if not bars.empty:
                    all_bars.append(bars)
            except Exception as e:
                logger.error(f"Error fetching data for chunk {i}: {e}")
            time.sleep(0.1) # Rate limiting
            
        if not all_bars:
            return pd.DataFrame()
            
        df = pd.concat(all_bars)
        df.index = pd.to_datetime(df.index)
        return df

    def check_data_freshness(self, df: pd.DataFrame, expected_date: datetime.date) -> bool:
        """
        Ensures the data includes the most recent expected trading day.
        """
        if df.empty:
            return False
            
        last_date = df.index.max().date()
        if last_date < expected_date:
            logger.warning(f"Data is stale. Last date: {last_date}, Expected: {expected_date}")
            return False
        return True



