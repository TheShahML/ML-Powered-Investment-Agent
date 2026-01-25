import alpaca_trade_api as tradeapi
import pandas as pd
import datetime
from loguru import logger
from typing import List, Optional
import time
import requests


# S&P 500 tickers (as of 2024 - update periodically)
SP500_TICKERS = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "TSLA", "BRK.B", "UNH", "XOM",
    "JNJ", "JPM", "V", "PG", "MA", "HD", "CVX", "MRK", "ABBV", "LLY",
    "PEP", "KO", "COST", "AVGO", "WMT", "MCD", "CSCO", "TMO", "ABT", "ACN",
    "DHR", "NEE", "DIS", "VZ", "ADBE", "WFC", "PM", "CMCSA", "TXN", "CRM",
    "NKE", "BMY", "RTX", "UPS", "QCOM", "HON", "T", "LOW", "ORCL", "MS",
    "SPGI", "UNP", "INTC", "IBM", "GS", "CAT", "BA", "SBUX", "AMD", "DE",
    "INTU", "PLD", "AMGN", "BLK", "GILD", "AXP", "MDLZ", "ADI", "CVS", "SYK",
    "ISRG", "LMT", "BKNG", "MMC", "ADP", "TJX", "CI", "REGN", "SCHW", "NOW",
    "MO", "ZTS", "C", "VRTX", "PNC", "CB", "SO", "TMUS", "BDX", "CME",
    "DUK", "ITW", "CL", "EOG", "BSX", "SLB", "NOC", "FI", "AON", "ICE",
    # Add more as needed - this is a subset of top 100
]


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
        Fetches the tradeable universe based on config.
        Supports: 'sp500', 'all', or custom list.
        """
        universe_type = self.config.get('universe_type', 'sp500')
        logger.info(f"Fetching universe (type: {universe_type})...")

        if universe_type == 'sp500':
            return self._get_sp500_universe()
        elif universe_type == 'all':
            return self._get_all_equities()
        elif universe_type == 'custom':
            return self.config.get('custom_tickers', SP500_TICKERS[:50])
        else:
            return self._get_sp500_universe()

    def _get_sp500_universe(self) -> List[str]:
        """Get S&P 500 stocks that are tradeable on Alpaca."""
        logger.info("Using S&P 500 universe...")

        # Try to fetch live S&P 500 list
        try:
            sp500 = self._fetch_sp500_from_wikipedia()
            if sp500:
                logger.info(f"Fetched {len(sp500)} S&P 500 tickers from Wikipedia")
        except Exception as e:
            logger.warning(f"Could not fetch S&P 500 list: {e}")
            sp500 = SP500_TICKERS

        # Filter for tradeable on Alpaca
        tradeable = []
        assets = self.api.list_assets(status='active', asset_class='us_equity')
        tradeable_symbols = {a.symbol for a in assets if a.tradable}

        for ticker in sp500:
            # Handle BRK.B -> BRKB format if needed
            alpaca_ticker = ticker.replace('.', '')
            if ticker in tradeable_symbols or alpaca_ticker in tradeable_symbols:
                tradeable.append(ticker if ticker in tradeable_symbols else alpaca_ticker)

        logger.info(f"S&P 500 tradeable on Alpaca: {len(tradeable)} stocks")
        return tradeable[:self.config.get('target_size', 500)]

    def _fetch_sp500_from_wikipedia(self) -> List[str]:
        """Fetch current S&P 500 list from Wikipedia."""
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        try:
            tables = pd.read_html(url)
            df = tables[0]
            tickers = df['Symbol'].tolist()
            return [t.replace('.', '-') for t in tickers]  # BRK.B -> BRK-B
        except Exception:
            return SP500_TICKERS

    def _get_all_equities(self) -> List[str]:
        """Get all tradeable US equities."""
        logger.info("Using all US equities universe...")

        assets = self.api.list_assets(status='active', asset_class='us_equity')

        universe = [
            a.symbol for a in assets
            if a.tradable and a.shortable and a.marginable
            and (not self.config.get('fractional_shares') or a.fractionable)
        ]

        logger.info(f"All equities universe: {len(universe)} stocks")
        return universe[:self.config.get('target_size', 500)]

    def get_historical_data(self, tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetches historical EOD bars for the given tickers.
        """
        logger.info(f"Fetching historical data for {len(tickers)} tickers from {start_date} to {end_date}...")

        all_bars = []
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
            time.sleep(0.1)

        if not all_bars:
            return pd.DataFrame()

        df = pd.concat(all_bars)
        df.index = pd.to_datetime(df.index)
        return df

    def check_data_freshness(self, df: pd.DataFrame, expected_date: datetime.date) -> bool:
        """Ensures the data includes the most recent expected trading day."""
        if df.empty:
            return False

        last_date = df.index.max().date()
        if last_date < expected_date:
            logger.warning(f"Data is stale. Last date: {last_date}, Expected: {expected_date}")
            return False
        return True


class CryptoDataService:
    """
    Handles cryptocurrency data and trading via Alpaca.
    Alpaca supports BTC/USD, ETH/USD, and other crypto pairs.
    """

    # Supported crypto on Alpaca
    SUPPORTED_CRYPTO = ['BTC/USD', 'ETH/USD', 'LTC/USD', 'BCH/USD', 'LINK/USD', 'UNI/USD', 'AAVE/USD']

    def __init__(self, config):
        self.config = config
        self.api = tradeapi.REST(
            self.config['ALPACA_API_KEY'],
            self.config['ALPACA_SECRET_KEY'],
            self.config['ALPACA_BASE_URL']
        )

    def get_crypto_universe(self) -> List[str]:
        """Get list of supported crypto assets."""
        crypto_config = self.config.get('crypto_tickers', ['BTC/USD'])
        logger.info(f"Crypto universe: {crypto_config}")
        return crypto_config

    def get_crypto_data(self, symbols: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch historical crypto data.
        Note: Alpaca crypto uses different endpoint.
        """
        logger.info(f"Fetching crypto data for {symbols}...")

        all_bars = []
        for symbol in symbols:
            try:
                # Alpaca crypto bars endpoint
                bars = self.api.get_crypto_bars(
                    symbol,
                    tradeapi.rest.TimeFrame.Day,
                    start=start_date,
                    end=end_date
                ).df

                if not bars.empty:
                    bars['symbol'] = symbol.replace('/', '')  # BTC/USD -> BTCUSD
                    all_bars.append(bars)

            except Exception as e:
                logger.error(f"Error fetching crypto data for {symbol}: {e}")

        if not all_bars:
            return pd.DataFrame()

        df = pd.concat(all_bars)
        return df

    def get_latest_crypto_price(self, symbol: str) -> float:
        """Get latest crypto price."""
        try:
            quote = self.api.get_latest_crypto_quote(symbol)
            return float(quote.ap)  # Ask price
        except Exception as e:
            logger.error(f"Error getting crypto price for {symbol}: {e}")
            return 0.0


class CombinedDataService:
    """
    Combined service that handles both equities and crypto.
    """

    def __init__(self, config):
        self.config = config
        self.equity_service = DataService(config)
        self.crypto_service = CryptoDataService(config)

    def get_full_universe(self) -> dict:
        """
        Get both equity and crypto universes.
        Returns: {'equities': [...], 'crypto': [...]}
        """
        result = {
            'equities': [],
            'crypto': []
        }

        if self.config.get('trade_equities', True):
            result['equities'] = self.equity_service.get_universe()

        if self.config.get('trade_crypto', False):
            result['crypto'] = self.crypto_service.get_crypto_universe()

        logger.info(f"Full universe: {len(result['equities'])} equities, {len(result['crypto'])} crypto")
        return result
