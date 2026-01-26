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

    # Compute targets for all horizons
    logger.info("Computing targets for all horizons (1d, 5d, 20d)...")
    grouped = df_features.groupby(level=1)
    df_features['target_1d'] = grouped['close'].transform(lambda x: x.shift(-1) / x - 1)
    df_features['target_5d'] = grouped['close'].transform(lambda x: x.shift(-5) / x - 1)
    df_features['target_20d'] = grouped['close'].transform(lambda x: x.shift(-20) / x - 1)

    # Train 3 separate models
    horizons = [
        ('1d', 'target_1d', 1),
        ('5d', 'target_5d', 5),
        ('20d', 'target_20d', 20)
    ]

    all_metrics = {}

    for horizon_name, target_col, days in horizons:
        logger.info("\n" + "=" * 60)
        logger.info(f"TRAINING {horizon_name.upper()} MODEL ({days}-day forward returns)")
        logger.info("=" * 60)

        # Create strategy instance for this horizon
        strategy_h = SimpleStrategy(config, model_dir='models', horizon=horizon_name)

        # Filter out rows with missing target
        df_train = df_features.dropna(subset=[target_col])
        logger.info(f"Training samples for {horizon_name}: {len(df_train)}")

        # Train model
        metrics = strategy_h.train(df_train, target_col=target_col)
        all_metrics[horizon_name] = metrics

        # Feature importance
        importance = strategy_h.get_feature_importance()
        if not importance.empty:
            logger.info(f"\n{horizon_name.upper()} Feature Importances:")
            for _, row in importance.iterrows():
                logger.info(f"  {row['feature']}: {row['importance']:.4f}")

    # Results summary for all horizons
    logger.info("\n" + "=" * 60)
    logger.info("MULTI-HORIZON TRAINING RESULTS")
    logger.info("=" * 60)

    for horizon_name in ['1d', '5d', '20d']:
        if horizon_name in all_metrics:
            metrics = all_metrics[horizon_name]
            logger.info(f"\n{horizon_name.upper()} Model:")
            for name, value in metrics.items():
                if isinstance(value, float):
                    logger.info(f"  {name}: {value:.4f}")
                else:
                    logger.info(f"  {name}: {value}")

    # Discord notification
    discord = DiscordNotifier()
    discord.send_multi_horizon_training_complete(all_metrics)

    logger.info("\n" + "=" * 60)
    logger.info("Training complete! 3 models saved:")
    logger.info("  - models/simple_xgb_1d.pkl")
    logger.info("  - models/simple_xgb_5d.pkl")
    logger.info("  - models/simple_xgb_20d.pkl (PRIMARY for rebalancing)")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
