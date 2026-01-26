#!/usr/bin/env python3
"""
Daily Signal Generation Script for GitHub Actions.
Supports multiple strategies: simple (recommended), stacking, or pure momentum.
"""

import os
import sys
from datetime import date, datetime, timedelta
from loguru import logger

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_service import DataService
from src.file_storage import FileStorage
from src.config import load_config
from src.discord_notifier import DiscordNotifier


def get_strategy(config):
    """
    Load the appropriate strategy based on config.
    Priority: simple > stacking > pure_momentum
    """
    strategy_type = config.get('strategy_type', 'simple')

    if strategy_type == 'simple':
        from src.strategy_simple import SimpleStrategy
        strategy = SimpleStrategy(config)
        if strategy.load_model():
            logger.info("Using SIMPLE XGBoost strategy (recommended)")
            return strategy, 'simple'

    if strategy_type in ['simple', 'stacking']:
        from src.strategy_stacking import StackingEnsembleStrategy
        strategy = StackingEnsembleStrategy(config)
        if strategy.load_models():
            logger.info("Using STACKING ensemble strategy")
            return strategy, 'stacking'

    # Fallback to pure momentum (no ML, no overfitting)
    from src.strategy_simple import PureMomentumStrategy
    logger.warning("No trained model found. Using PURE MOMENTUM (no ML)")
    return PureMomentumStrategy(config), 'pure_momentum'


def main():
    logger.info("=" * 60)
    logger.info("DAILY SIGNAL GENERATION")
    logger.info(f"Date: {date.today()}")
    logger.info("=" * 60)

    # Load config
    config = load_config()

    # Initialize components
    data_service = DataService(config)
    storage = FileStorage()

    # Get strategy
    strategy, strategy_type = get_strategy(config)

    # Get universe of stocks
    logger.info("Fetching tradeable universe...")
    universe = data_service.get_universe()
    logger.info(f"Universe size: {len(universe)} stocks")

    # Fetch historical data (need enough for feature calculation)
    # Need 2 years: 1 year for data + 1 year lookback for 12-month momentum
    end_date = date.today()
    start_date = end_date - timedelta(days=365 * 2)

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

    # Compute features if using simple strategy
    if strategy_type == 'simple':
        logger.info("Computing simple features (5 only - less overfitting)...")
        df = strategy.compute_features(df)
    elif strategy_type == 'stacking':
        from src.feature_engineering_enhanced import EnhancedFeatureEngineer
        feature_engineer = EnhancedFeatureEngineer(config)
        logger.info("Computing enhanced features...")
        df = feature_engineer.compute_features(df)

    logger.info(f"Features computed: {len(df)} rows")

    # Generate signals
    logger.info("Generating signals...")
    signals = strategy.compute_signals(df)

    # Get top picks
    n_holdings = config.get('n_holdings', 15)
    top_picks = signals.head(n_holdings)

    logger.info(f"\nTop {n_holdings} stocks for today ({strategy_type}):")
    logger.info("-" * 40)
    for ticker, row in top_picks.iterrows():
        logger.info(f"  {int(row['rank']):2d}. {ticker:6s} | Score: {row['score']:.4f}")

    # Save signals
    storage.save_signals(signals, date.today())

    # Get rebalance info for Discord notification
    state = storage.get_rebalance_state()
    days_since = state.get('days_since_rebalance', 0)
    days_until_rebalance = max(0, 20 - days_since)

    # Calculate next rebalance date (approximate - skips weekends)
    next_rebalance = date.today()
    days_added = 0
    while days_added < days_until_rebalance:
        next_rebalance += timedelta(days=1)
        if next_rebalance.weekday() < 5:  # Monday = 0, Friday = 4
            days_added += 1
    next_rebalance_str = next_rebalance.strftime('%Y-%m-%d')

    logger.info(f"Days until rebalance: {days_until_rebalance} (estimated: {next_rebalance_str})")

    # Send enhanced Discord notification
    discord = DiscordNotifier()
    discord.send_enhanced_signals(
        signals_df=signals,
        days_until_rebalance=days_until_rebalance,
        next_rebalance_date=next_rebalance_str,
        date=date.today().strftime('%Y-%m-%d')
    )

    logger.info("\n" + "=" * 60)
    logger.info(f"Signal generation complete! (Strategy: {strategy_type})")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
