"""
Trading Calendar and As-Of Date Control.

Prevents lookahead bias by enforcing strict "as-of" semantics:
- Training, signals, and rebalancing use ONLY data available up to last completed trading day.
- Market data freshness checks prevent trading on stale data.
"""

import alpaca_trade_api as tradeapi
from datetime import date, datetime, timedelta
from loguru import logger
from typing import Optional
import pandas as pd


class TradingCalendar:
    """
    Manages trading calendar and as-of date semantics.

    Key principle: We can only use data from trading days that have COMPLETED.
    - If today is a trading day but market hasn't closed: last completed = previous trading day
    - If today is non-trading day: last completed = previous trading day
    """

    def __init__(self, api: tradeapi.REST):
        self.api = api
        self._cache = {}

    def get_last_completed_trading_day(self, reference_date: Optional[date] = None) -> date:
        """
        Get the last trading day that has completed (market closed).

        Args:
            reference_date: Reference date (default: today)

        Returns:
            Last completed trading day as date object
        """
        if reference_date is None:
            reference_date = date.today()

        # Check cache
        cache_key = f"last_completed_{reference_date}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            # Get market clock
            clock = self.api.get_clock()

            # If market is currently open, last completed day is previous trading day
            if clock.is_open:
                logger.info("Market is currently open - using previous trading day")
                result = self._get_previous_trading_day(reference_date)
            else:
                # Market is closed - check if today is a trading day
                calendar = self.api.get_calendar(
                    start=reference_date.strftime('%Y-%m-%d'),
                    end=reference_date.strftime('%Y-%m-%d')
                )

                if calendar and len(calendar) > 0:
                    # Today is a trading day and market is closed
                    # Check if we're AFTER market close
                    today_calendar = calendar[0]
                    market_close = datetime.combine(
                        reference_date,
                        datetime.strptime(today_calendar.close, '%H:%M').time()
                    )

                    if datetime.now() >= market_close:
                        # We're after market close - today is completed
                        result = reference_date
                    else:
                        # We're before market close - use previous day
                        result = self._get_previous_trading_day(reference_date)
                else:
                    # Today is not a trading day
                    result = self._get_previous_trading_day(reference_date)

            self._cache[cache_key] = result
            logger.info(f"Last completed trading day: {result}")
            return result

        except Exception as e:
            logger.error(f"Error getting last completed trading day: {e}")
            # Fallback: go back 1-3 days to be safe
            result = self._get_previous_trading_day(reference_date)
            logger.warning(f"Using fallback: {result}")
            return result

    def _get_previous_trading_day(self, from_date: date) -> date:
        """Get the trading day before from_date."""
        try:
            # Go back up to 7 days to find a trading day
            start_date = from_date - timedelta(days=7)
            end_date = from_date - timedelta(days=1)

            calendar = self.api.get_calendar(
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d')
            )

            if calendar and len(calendar) > 0:
                # Return the last trading day in the range
                last_day = calendar[-1].date
                return datetime.strptime(last_day, '%Y-%m-%d').date()
            else:
                # Fallback: subtract days until we hit a weekday
                candidate = end_date
                while candidate.weekday() >= 5:  # Skip weekends
                    candidate -= timedelta(days=1)
                return candidate

        except Exception as e:
            logger.error(f"Error getting previous trading day: {e}")
            # Simple fallback
            candidate = from_date - timedelta(days=1)
            while candidate.weekday() >= 5:
                candidate -= timedelta(days=1)
            return candidate

    def verify_data_freshness(
        self,
        data_df: pd.DataFrame,
        expected_date: date,
        symbols: Optional[list] = None,
        tolerance_days: int = 0
    ) -> tuple[bool, dict]:
        """
        Verify that data is fresh enough for trading.

        Args:
            data_df: DataFrame with MultiIndex (timestamp, symbol) or just timestamp index
            expected_date: Expected last trading day
            symbols: Optional list of symbols to check (checks all if None)
            tolerance_days: Allow data to be this many days old (default: 0 = must be exact)

        Returns:
            (is_fresh: bool, details: dict with per-symbol info)
        """
        details = {
            'expected_date': expected_date,
            'tolerance_days': tolerance_days,
            'stale_symbols': [],
            'missing_symbols': [],
            'fresh_count': 0
        }

        try:
            # Get last date in data
            if isinstance(data_df.index, pd.MultiIndex):
                # Extract timestamp level
                last_dates_by_symbol = data_df.groupby(level=1).apply(
                    lambda x: x.index.get_level_values(0).max()
                )

                if symbols is None:
                    symbols = last_dates_by_symbol.index.tolist()

                for symbol in symbols:
                    if symbol not in last_dates_by_symbol.index:
                        details['missing_symbols'].append(symbol)
                        continue

                    last_date = pd.to_datetime(last_dates_by_symbol[symbol]).date()
                    days_old = (expected_date - last_date).days

                    if days_old > tolerance_days:
                        details['stale_symbols'].append({
                            'symbol': symbol,
                            'last_date': last_date,
                            'days_old': days_old
                        })
                    else:
                        details['fresh_count'] += 1
            else:
                # Simple timestamp index
                last_date = pd.to_datetime(data_df.index.max()).date()
                days_old = (expected_date - last_date).days

                if days_old > tolerance_days:
                    details['stale_symbols'].append({
                        'symbol': 'ALL',
                        'last_date': last_date,
                        'days_old': days_old
                    })
                else:
                    details['fresh_count'] = 1

            is_fresh = (
                len(details['missing_symbols']) == 0 and
                len(details['stale_symbols']) == 0
            )

            if not is_fresh:
                logger.warning(f"Data freshness check FAILED: {details}")
            else:
                logger.info(f"Data freshness check PASSED: {details['fresh_count']} symbols fresh")

            return is_fresh, details

        except Exception as e:
            logger.error(f"Error verifying data freshness: {e}")
            return False, {'error': str(e)}

    def get_trading_days_between(self, start_date: date, end_date: date) -> list[date]:
        """Get list of trading days between start and end (inclusive)."""
        try:
            calendar = self.api.get_calendar(
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d')
            )
            return [datetime.strptime(day.date, '%Y-%m-%d').date() for day in calendar]
        except Exception as e:
            logger.error(f"Error getting trading days: {e}")
            return []

    def is_market_open(self) -> bool:
        """Check if market is currently open."""
        try:
            clock = self.api.get_clock()
            return clock.is_open
        except Exception as e:
            logger.error(f"Error checking market status: {e}")
            return False
