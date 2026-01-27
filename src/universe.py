"""
Universe Construction: S&P 500 ∪ NASDAQ-100 with filters.

Provides:
- Symbol normalization (handle BRK-B, BRK.B, BRKB variations)
- S&P 500 constituent fetching
- NASDAQ-100 constituent fetching
- Union + deduplication
- Alpaca tradeable filtering
- Price/volume/liquidity filters
"""

import alpaca_trade_api as tradeapi
import requests
import pandas as pd
from loguru import logger
from typing import List, Dict, Set
from datetime import date, timedelta


# NASDAQ-100 constituents (as of 2024-2025)
# Source: https://www.nasdaq.com/market-activity/quotes/nasdaq-ndx-index
NASDAQ_100_TICKERS = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "GOOG", "META", "TSLA", "AVGO", "COST",
    "NFLX", "AMD", "PEP", "ADBE", "CSCO", "TMUS", "CMCSA", "INTC", "INTU", "TXN",
    "QCOM", "AMGN", "HON", "AMAT", "SBUX", "BKNG", "ISRG", "ADI", "VRTX", "ADP",
    "GILD", "REGN", "MDLZ", "LRCX", "PANW", "MU", "PYPL", "SNPS", "KLAC", "CDNS",
    "MELI", "NXPI", "ASML", "ABNB", "MAR", "CRWD", "MRVL", "ORLY", "FTNT", "CSX",
    "DASH", "ADSK", "WDAY", "MNST", "ROP", "PCAR", "CHTR", "PAYX", "AEP", "CPRT",
    "ROST", "ODFL", "FAST", "KDP", "EA", "VRSK", "DXCM", "BKR", "CTSH", "EXC",
    "GEHC", "IDXX", "TEAM", "XEL", "CTAS", "KHC", "LULU", "CCEP", "TTD", "FANG",
    "ZS", "ANSS", "ON", "DDOG", "CDW", "BIIB", "WBD", "GFS", "ILMN", "MDB",
    "MRNA", "ARM", "SMCI", "WBA", "DLTR", "PDD", "ZM", "RIVN", "LCID", "ENPH"
]


class Universe:
    """Manages investment universe construction and symbol normalization."""

    def __init__(self, api: tradeapi.REST, config: dict):
        self.api = api
        self.config = config

    def normalize_symbol_for_alpaca(self, symbol: str) -> str:
        """
        Normalize symbol for Alpaca API compatibility.

        Handles common variations:
        - BRK-B, BRK.B -> BRKB
        - BF-B, BF.B -> BFB
        - Strips whitespace
        - Uppercases

        Args:
            symbol: Raw symbol

        Returns:
            Normalized symbol for Alpaca
        """
        # Uppercase and strip
        normalized = symbol.upper().strip()

        # Handle Class B shares: X-B or X.B -> XB
        if '-' in normalized:
            normalized = normalized.replace('-', '')
        if '.' in normalized and normalized.endswith('.B'):
            normalized = normalized.replace('.', '')

        return normalized

    def get_sp500_constituents(self) -> List[str]:
        """
        Fetch S&P 500 constituent symbols from Wikipedia.

        Returns:
            List of S&P 500 symbols (normalized)
        """
        try:
            logger.info("Fetching S&P 500 constituents...")
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            tables = pd.read_html(url)
            df = tables[0]

            # Symbol is in 'Symbol' column
            symbols = df['Symbol'].tolist()

            # Normalize all symbols
            normalized = [self.normalize_symbol_for_alpaca(s) for s in symbols]

            logger.info(f"Fetched {len(normalized)} S&P 500 symbols")
            return normalized

        except Exception as e:
            logger.error(f"Error fetching S&P 500 constituents: {e}")
            # Fallback to a core set
            return [self.normalize_symbol_for_alpaca(s) for s in [
                "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "TSLA", "BRK-B", "UNH", "XOM",
                "JPM", "V", "PG", "MA", "HD", "CVX", "MRK", "ABBV", "LLY", "PEP",
                "KO", "COST", "AVGO", "WMT", "MCD", "CSCO", "TMO", "ABT", "ACN", "DHR"
            ]]

    def get_nasdaq100_constituents(self) -> List[str]:
        """
        Get NASDAQ-100 constituent symbols.

        Returns:
            List of NASDAQ-100 symbols (normalized)
        """
        logger.info(f"Using NASDAQ-100 list: {len(NASDAQ_100_TICKERS)} symbols")
        return [self.normalize_symbol_for_alpaca(s) for s in NASDAQ_100_TICKERS]

    def get_union_universe(self) -> List[str]:
        """
        Get union of S&P 500 and NASDAQ-100 (deduplicated).

        Returns:
            List of unique symbols from both indices
        """
        sp500 = set(self.get_sp500_constituents())
        ndx100 = set(self.get_nasdaq100_constituents())

        union = sp500 | ndx100
        logger.info(f"Universe union: {len(union)} unique symbols (SP500: {len(sp500)}, NDX100: {len(ndx100)})")

        return sorted(list(union))

    def filter_tradeable(self, symbols: List[str]) -> tuple[List[str], Dict]:
        """
        Filter to Alpaca-tradeable assets.

        Args:
            symbols: List of symbols to filter

        Returns:
            (tradeable_symbols, stats_dict)
        """
        stats = {
            'input_count': len(symbols),
            'tradeable_count': 0,
            'non_tradeable': [],
            'not_found': []
        }

        tradeable = []

        try:
            # Get all assets from Alpaca
            all_assets = self.api.list_assets(status='active')
            alpaca_symbols = {a.symbol: a for a in all_assets}

            for symbol in symbols:
                if symbol not in alpaca_symbols:
                    stats['not_found'].append(symbol)
                    continue

                asset = alpaca_symbols[symbol]

                # Check if tradeable
                if asset.tradable and asset.shortable and asset.marginable and asset.fractionable:
                    tradeable.append(symbol)
                else:
                    stats['non_tradeable'].append({
                        'symbol': symbol,
                        'tradable': asset.tradable,
                        'shortable': asset.shortable,
                        'marginable': asset.marginable,
                        'fractionable': asset.fractionable
                    })

            stats['tradeable_count'] = len(tradeable)

            logger.info(f"Tradeable filter: {stats['tradeable_count']}/{stats['input_count']} passed")
            if stats['not_found']:
                logger.warning(f"Symbols not found on Alpaca: {stats['not_found'][:10]}...")
            if stats['non_tradeable']:
                logger.warning(f"Non-tradeable assets: {len(stats['non_tradeable'])}")

            return tradeable, stats

        except Exception as e:
            logger.error(f"Error filtering tradeable assets: {e}")
            return symbols, stats  # Return original on error

    def apply_liquidity_filters(
        self,
        symbols: List[str],
        min_price: float,
        min_adv: float,
        lookback_days: int = 30
    ) -> tuple[List[str], Dict]:
        """
        Apply price and average dollar volume (ADV) filters.

        Args:
            symbols: Symbols to filter
            min_price: Minimum price (e.g., 5.0)
            min_adv: Minimum average dollar volume (e.g., 20_000_000)
            lookback_days: Days to compute ADV (default: 30)

        Returns:
            (filtered_symbols, stats_dict)
        """
        from .trading_calendar import TradingCalendar
        from .data_service import DataService

        stats = {
            'input_count': len(symbols),
            'passed_count': 0,
            'failed_price': [],
            'failed_adv': [],
            'failed_data': []
        }

        try:
            # Get last completed trading day
            calendar = TradingCalendar(self.api)
            end_date = calendar.get_last_completed_trading_day()
            start_date = end_date - timedelta(days=lookback_days + 10)  # Extra buffer

            logger.info(f"Fetching {lookback_days}-day data for liquidity filters...")

            # Fetch data
            data_service = DataService(self.config)
            df = data_service.get_historical_data(
                symbols,
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            )

            if df.empty:
                logger.error("No data returned for liquidity filtering")
                return symbols, stats

            passed = []

            for symbol in symbols:
                try:
                    # Get symbol data
                    symbol_data = df.xs(symbol, level=1)

                    if len(symbol_data) < lookback_days // 2:
                        stats['failed_data'].append(symbol)
                        continue

                    # Check latest price
                    latest_price = symbol_data['close'].iloc[-1]
                    if latest_price < min_price:
                        stats['failed_price'].append({'symbol': symbol, 'price': latest_price})
                        continue

                    # Compute ADV
                    dollar_volume = symbol_data['close'] * symbol_data['volume']
                    adv = dollar_volume.mean()

                    if adv < min_adv:
                        stats['failed_adv'].append({'symbol': symbol, 'adv': adv})
                        continue

                    passed.append(symbol)

                except Exception as e:
                    logger.warning(f"Error processing {symbol} for filters: {e}")
                    stats['failed_data'].append(symbol)

            stats['passed_count'] = len(passed)

            logger.info(f"Liquidity filters: {stats['passed_count']}/{stats['input_count']} passed")
            logger.info(f"  Failed price filter: {len(stats['failed_price'])}")
            logger.info(f"  Failed ADV filter: {len(stats['failed_adv'])}")
            logger.info(f"  Insufficient data: {len(stats['failed_data'])}")

            return passed, stats

        except Exception as e:
            logger.error(f"Error applying liquidity filters: {e}")
            return symbols, stats

    def build_universe(self) -> tuple[List[str], Dict]:
        """
        Build full filtered universe: S&P 500 ∪ NASDAQ-100 with all filters applied.

        Returns:
            (final_symbols, detailed_stats)
        """
        logger.info("=" * 60)
        logger.info("BUILDING INVESTMENT UNIVERSE")
        logger.info("=" * 60)

        all_stats = {}

        # Step 1: Get union
        raw_symbols = self.get_union_universe()
        all_stats['raw_count'] = len(raw_symbols)

        # Step 2: Filter tradeable
        tradeable, tradeable_stats = self.filter_tradeable(raw_symbols)
        all_stats['tradeable_stats'] = tradeable_stats

        # Step 3: Apply liquidity filters
        min_price = self.config.get('min_price', 5.0)
        min_adv = self.config.get('min_avg_dollar_volume', 20_000_000)

        final, liquidity_stats = self.apply_liquidity_filters(
            tradeable,
            min_price=min_price,
            min_adv=min_adv
        )
        all_stats['liquidity_stats'] = liquidity_stats
        all_stats['final_count'] = len(final)

        logger.info("=" * 60)
        logger.info(f"UNIVERSE BUILD COMPLETE: {all_stats['final_count']} symbols")
        logger.info("=" * 60)

        return final, all_stats
