#!/usr/bin/env python3
"""Debug script to check data fetching and universe building."""
import os
import sys
from datetime import timedelta
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config
from src.trading_calendar import TradingCalendar
from src.universe import Universe
from src.data_service import DataService
import alpaca_trade_api as tradeapi


def main():
    logger.info("=" * 60)
    logger.info("DATA DEBUGGING")
    logger.info("=" * 60)

    # Load config
    config = load_config()
    api = tradeapi.REST(
        config['ALPACA_API_KEY'],
        config['ALPACA_SECRET_KEY'],
        config['ALPACA_BASE_URL']
    )

    # Get as-of date
    calendar = TradingCalendar(api)
    as_of_date = calendar.get_last_completed_trading_day()
    logger.info(f"As-of date: {as_of_date}")

    # Build universe
    universe_builder = Universe(api, config)
    symbols, universe_stats = universe_builder.build_universe()
    logger.info(f"\nUniverse Stats:")
    logger.info(f"  Total symbols: {len(symbols)}")
    logger.info(f"  Stats: {universe_stats}")
    logger.info(f"  First 10: {symbols[:10]}")

    # Fetch small sample of data
    test_symbols = symbols[:5]
    end_date = as_of_date
    start_date = end_date - timedelta(days=30)

    logger.info(f"\nFetching 30 days of data for {len(test_symbols)} symbols...")
    data_service = DataService(config)
    df = data_service.get_historical_data(
        test_symbols,
        start_date.strftime('%Y-%m-%d'),
        end_date.strftime('%Y-%m-%d')
    )

    if df.empty:
        logger.error("DataFrame is EMPTY!")
    else:
        logger.info(f"✓ Data fetched successfully")
        logger.info(f"  Shape: {df.shape}")
        logger.info(f"  Date range: {df.index.get_level_values(0).min()} to {df.index.get_level_values(0).max()}")
        logger.info(f"  Symbols: {df.index.get_level_values(1).unique().tolist()}")
        logger.info(f"\nFirst few rows:")
        logger.info(f"\n{df.head(10)}")

        # Check for NaNs
        logger.info(f"\nNaN counts:")
        logger.info(f"\n{df.isnull().sum()}")

        # Check price ranges
        logger.info(f"\nPrice statistics:")
        logger.info(f"\n{df['close'].describe()}")


if __name__ == "__main__":
    main()
