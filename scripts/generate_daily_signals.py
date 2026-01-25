#!/usr/bin/env python3
"""
Daily Signal Generation Script for GitHub Actions.
Generates ML predictions and saves them to the data/ directory.
"""

import os
import sys
from datetime import date, datetime, timedelta
from loguru import logger

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_service import DataService
from src.feature_engineering_enhanced import EnhancedFeatureEngineer
from src.strategy_stacking import StackingEnsembleStrategy
from src.file_storage import FileStorage
from src.config import load_config
from src.discord_notifier import DiscordNotifier


def main():
    logger.info("=" * 60)
    logger.info("DAILY SIGNAL GENERATION")
    logger.info(f"Date: {date.today()}")
    logger.info("=" * 60)

    # Load config
    config = load_config()

    # Initialize components
    data_service = DataService(config)
    feature_engineer = EnhancedFeatureEngineer(config)
    storage = FileStorage()

    # Try to use stacking model, fall back to basic if not trained
    strategy = StackingEnsembleStrategy(config)
    if not strategy.load_models():
        logger.warning("Stacking model not found, using basic LightGBM")
        from src.strategy_ml import MLStrategy
        strategy = MLStrategy(config)
        if not strategy.load_model():
            logger.error("No trained model found! Run train_stacking_model.py first.")
            sys.exit(1)

    # Get universe of stocks
    logger.info("Fetching tradeable universe...")
    universe = data_service.get_universe()
    logger.info(f"Universe size: {len(universe)} stocks")

    # Fetch historical data (need enough for feature calculation)
    end_date = date.today()
    start_date = end_date - timedelta(days=365)  # 1 year for features

    logger.info(f"Fetching historical data from {start_date} to {end_date}...")
    df = data_service.get_historical_data(
        universe,
        start_date.strftime('%Y-%m-%d'),
        end_date.strftime('%Y-%m-%d')
    )

    if df.empty:
        logger.error("No historical data retrieved!")
        sys.exit(1)

    logger.info(f"Retrieved {len(df)} data points")

    # Compute features
    logger.info("Computing features...")
    df_features = feature_engineer.compute_features(df)
    logger.info(f"Features computed: {len(df_features)} rows")

    # Generate signals
    logger.info("Generating ML signals...")
    signals = strategy.compute_signals(df_features)

    # Get top picks
    n_holdings = config.get('n_holdings', 15)
    top_picks = signals.head(n_holdings)

    logger.info(f"\nTop {n_holdings} stocks for today:")
    logger.info("-" * 40)
    for ticker, row in top_picks.iterrows():
        logger.info(f"  {int(row['rank']):2d}. {ticker:6s} | Score: {row['score']:.4f}")

    # Save signals
    storage.save_signals(signals, date.today())

    # Send Discord notification
    discord = DiscordNotifier()
    discord.send_daily_signals(signals, date.today().strftime('%Y-%m-%d'))

    logger.info("\n" + "=" * 60)
    logger.info("Signal generation complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
