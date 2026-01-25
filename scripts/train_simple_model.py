#!/usr/bin/env python3
"""
Train the simple XGBoost model.
Less features, less overfitting, more robust.
"""

import os
import sys
from datetime import date, timedelta
from loguru import logger

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_service import DataService
from src.strategy_simple import SimpleStrategy
from src.config import load_config
from src.discord_notifier import DiscordNotifier


def main():
    logger.info("=" * 60)
    logger.info("SIMPLE XGBOOST MODEL TRAINING")
    logger.info("(Less overfitting, academically-backed features)")
    logger.info(f"Date: {date.today()}")
    logger.info("=" * 60)

    config = load_config()
    data_service = DataService(config)
    strategy = SimpleStrategy(config, model_dir='models')

    # Get universe
    logger.info("Fetching tradeable universe...")
    universe = data_service.get_universe()
    universe = universe[:200]  # Top 200 liquid stocks
    logger.info(f"Training universe: {len(universe)} stocks")

    # Fetch 2 years of data
    end_date = date.today()
    start_date = end_date - timedelta(days=365 * 2)

    logger.info(f"Fetching data from {start_date} to {end_date}...")
    df = data_service.get_historical_data(
        universe,
        start_date.strftime('%Y-%m-%d'),
        end_date.strftime('%Y-%m-%d')
    )

    if df.empty:
        logger.error("No data retrieved!")
        sys.exit(1)

    logger.info(f"Retrieved {len(df)} data points")

    # Compute features
    logger.info("Computing simple features (5 only)...")
    df_features = strategy.compute_features(df)
    logger.info(f"Features computed: {len(df_features)} rows")

    # Target: 5-day forward returns
    logger.info("Computing target (5-day forward returns)...")
    grouped = df_features.groupby(level=1)
    df_features['target'] = grouped['close'].transform(lambda x: x.shift(-5) / x - 1)

    df_train = df_features.dropna(subset=['target'])
    logger.info(f"Training samples: {len(df_train)}")

    # Train
    logger.info("Training simple XGBoost...")
    metrics = strategy.train(df_train, target_col='target')

    # Results
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING RESULTS")
    logger.info("=" * 60)
    for name, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"  {name}: {value:.4f}")
        else:
            logger.info(f"  {name}: {value}")

    # Feature importance
    importance = strategy.get_feature_importance()
    if not importance.empty:
        logger.info("\nFeature Importances:")
        for _, row in importance.iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")

    # Discord notification
    discord = DiscordNotifier()
    discord.send_model_training_complete(metrics)

    logger.info("\n" + "=" * 60)
    logger.info("Training complete! Model saved to models/simple_xgb.pkl")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
