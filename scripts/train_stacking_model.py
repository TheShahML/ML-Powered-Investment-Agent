#!/usr/bin/env python3
"""
Train the stacking ensemble model.
Run weekly via GitHub Actions or manually before first deployment.
"""

import os
import sys
from datetime import date, timedelta
from loguru import logger

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_service import DataService
from src.feature_engineering_enhanced import EnhancedFeatureEngineer
from src.strategy_stacking import StackingEnsembleStrategy
from src.config import load_config
from src.discord_notifier import DiscordNotifier


def main():
    logger.info("=" * 60)
    logger.info("STACKING ENSEMBLE MODEL TRAINING")
    logger.info(f"Date: {date.today()}")
    logger.info("=" * 60)

    # Load config
    config = load_config()

    # Initialize components
    data_service = DataService(config)
    feature_engineer = EnhancedFeatureEngineer(config)
    strategy = StackingEnsembleStrategy(config, model_dir='models')

    # Get universe
    logger.info("Fetching tradeable universe...")
    universe = data_service.get_universe()

    # Limit universe for training efficiency
    universe = universe[:200]  # Top 200 most liquid stocks
    logger.info(f"Training universe: {len(universe)} stocks")

    # Fetch 2 years of historical data
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

    # Compute features
    logger.info("Computing features...")
    df_features = feature_engineer.compute_features(df)
    logger.info(f"Features computed: {len(df_features)} rows")

    # Create target: forward 5-day returns (next week's return)
    logger.info("Computing target variable (5-day forward returns)...")
    grouped = df_features.groupby(level=1)
    df_features['target'] = grouped['close'].transform(
        lambda x: x.shift(-5) / x - 1
    )

    # Drop rows without target
    df_train = df_features.dropna(subset=['target'])
    logger.info(f"Training samples: {len(df_train)}")

    # Train the model
    logger.info("Training stacking ensemble...")
    metrics = strategy.train(df_train, target_col='target')

    # Print results
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING RESULTS")
    logger.info("=" * 60)

    for name, value in metrics.items():
        logger.info(f"  {name}: {value:.4f}")

    # Feature importance
    importance = strategy.get_feature_importance()
    if not importance.empty:
        logger.info("\nTop 10 Feature Importances (XGBoost):")
        for _, row in importance.head(10).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")

    # Send Discord notification
    discord = DiscordNotifier()
    discord.send_model_training_complete(metrics)

    logger.info("\n" + "=" * 60)
    logger.info("Training complete! Models saved to models/")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
