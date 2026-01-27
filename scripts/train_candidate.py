#!/usr/bin/env python3
"""Train candidate model with walk-forward backtest and promotion gate."""
import os
import sys
from datetime import date, timedelta
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config
from src.trading_calendar import TradingCalendar
from src.universe import Universe
from src.data_service import DataService
from src.strategy_simple import SimpleStrategy, PureMomentumStrategy
from src.backtest_engine import WalkForwardBacktest, check_promotion_gate, save_backtest_report
from src.state_manager import StateManager
from src.discord_prod import DiscordProductionNotifier
import alpaca_trade_api as tradeapi


def main():
    logger.info("=" * 60)
    logger.info("WEEKLY CANDIDATE TRAINING")
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
    logger.info(f"Universe: {len(symbols)} stocks")

    # Use top 200 for training (reduce compute)
    training_universe = symbols[:200]

    # Fetch 2 years of data
    end_date = as_of_date
    start_date = end_date - timedelta(days=365 * 2)

    data_service = DataService(config)
    logger.info(f"Fetching data: {start_date} to {end_date}")
    df = data_service.get_historical_data(
        training_universe,
        start_date.strftime('%Y-%m-%d'),
        end_date.strftime('%Y-%m-%d')
    )

    if df.empty:
        logger.error("No data retrieved!")
        sys.exit(1)

    # Compute features (once for all models)
    strategy = SimpleStrategy(config, horizon='20d')
    logger.info("Computing features...")
    df_features = strategy.compute_features(df)

    # Compute all 3 targets
    grouped = df_features.groupby(level=1)
    df_features['target_1d'] = grouped['close'].transform(lambda x: x.shift(-1) / x - 1)
    df_features['target_5d'] = grouped['close'].transform(lambda x: x.shift(-5) / x - 1)
    df_features['target_20d'] = grouped['close'].transform(lambda x: x.shift(-20) / x - 1)

    logger.info(f"Feature computation complete")

    # Train all 3 models
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING MULTI-HORIZON MODELS")
    logger.info("=" * 60)

    horizons = [
        ('1d', 'target_1d', 1),
        ('5d', 'target_5d', 5),
        ('20d', 'target_20d', 20)
    ]

    all_cv_metrics = {}
    for horizon_name, target_col, embargo_days in horizons:
        logger.info(f"\n--- Training {horizon_name} model ---")

        strategy_h = SimpleStrategy(config, horizon=horizon_name)
        df_train = df_features.dropna(subset=[target_col])

        logger.info(f"Training samples: {len(df_train)}")

        cv_metrics = strategy_h.train(df_train, target_col=target_col, embargo_days=embargo_days)
        all_cv_metrics[horizon_name] = cv_metrics

        logger.info(f"{horizon_name} model trained: IC={cv_metrics.get('cv_mean_ic', 0):.4f}")

    logger.info("\n" + "=" * 60)
    logger.info("All 3 models trained successfully")
    logger.info("=" * 60)

    # Run walk-forward backtest (using 20d model as PRIMARY)
    logger.info("\n" + "=" * 60)
    logger.info("WALK-FORWARD BACKTEST (20d model)")
    logger.info("=" * 60)

    backtest_config = {
        'top_n': 25,  # Increased from 20 to 25
        'max_weight': 0.10,  # Reduced from 0.08 to 0.10 (10% per position)
        'weight_method': 'inverse_vol',
        'cost_bps': 15,
        'slippage_bps': 5,
        'turnover_buffer_pct': 1.0,
        'rebalance_freq_days': 20,
        'regime_filter': True
    }

    backtester = WalkForwardBacktest(backtest_config)

    # Backtest candidate (20d model)
    backtest_start = (end_date - timedelta(days=365 * 2)).strftime('%Y-%m-%d')
    backtest_end = end_date.strftime('%Y-%m-%d')

    strategy_20d = SimpleStrategy(config, horizon='20d')
    strategy_20d.load_model('20d')

    candidate_results = backtester.run_backtest(
        strategy_20d,
        df_features,
        backtest_start,
        backtest_end
    )

    logger.info(f"Candidate (20d) CAGR: {candidate_results.get('cagr', 0)*100:.1f}%")
    logger.info(f"Candidate (20d) Sharpe: {candidate_results.get('sharpe', 0):.2f}")
    logger.info(f"Candidate (20d) MaxDD: {candidate_results.get('max_drawdown', 0)*100:.1f}%")

    # Backtest baselines
    logger.info("\nBacktesting baselines...")

    momentum_strategy = PureMomentumStrategy(config)
    momentum_results = backtester.run_backtest(
        momentum_strategy,
        df_features,
        backtest_start,
        backtest_end
    )

    baselines = {
        'pure_momentum': momentum_results
    }

    # Check promotion gate
    logger.info("\n" + "=" * 60)
    logger.info("PROMOTION GATE CHECK")
    logger.info("=" * 60)

    passed, gate_details = check_promotion_gate(
        candidate_results,
        baselines,
        sharpe_margin=0.2,
        maxdd_tolerance=0.05
    )

    # Save backtest report
    report_path = save_backtest_report(candidate_results, baselines, (passed, gate_details))

    # Update state
    state_manager = StateManager(state_file_path="state-repo/latest_state.json")
    state = state_manager.load_state()

    candidate_version = f"multi_horizon_{as_of_date.strftime('%Y%m%d')}"

    state['models']['candidate_model'] = {
        'version': candidate_version,
        'trained_date': as_of_date.isoformat(),
        'training_window_start': start_date.isoformat(),
        'training_window_end': end_date.isoformat(),
        # CV metrics for all 3 horizons
        'cv_1d_mean_ic': all_cv_metrics['1d'].get('cv_mean_ic', 0),
        'cv_5d_mean_ic': all_cv_metrics['5d'].get('cv_mean_ic', 0),
        'cv_20d_mean_ic': all_cv_metrics['20d'].get('cv_mean_ic', 0),
        # Backtest uses 20d model (primary)
        'backtest_sharpe': candidate_results.get('sharpe', 0),
        'backtest_cagr': candidate_results.get('cagr', 0),
        'backtest_maxdd': candidate_results.get('max_drawdown', 0),
        'approved_for_next_rebalance': passed,
        'horizons': ['1d', '5d', '20d'],
        'primary_horizon': '20d'
    }

    state_manager.save_state(state)

    # Send Discord notification
    discord = DiscordProductionNotifier()
    discord.send_weekly_training_report(
        candidate_version=candidate_version,
        training_window=(start_date.isoformat(), end_date.isoformat()),
        cv_metrics=all_cv_metrics,  # Pass all 3 horizons' metrics
        backtest_candidate=candidate_results,
        backtest_baselines=baselines,
        gate_passed=passed,
        gate_details=gate_details
    )

    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info(f"Candidate: {candidate_version}")
    logger.info(f"Gate: {'PASSED' if passed else 'FAILED'}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
