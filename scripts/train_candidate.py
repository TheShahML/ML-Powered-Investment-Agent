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
import numpy as np
import pandas as pd
from src.state_manager import StateManager
from src.discord_prod import DiscordProductionNotifier
# Fix alpaca import issue
from src import alpaca_fix
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

    # Use ALL stocks for training (we have 400, not 3000+)
    training_universe = symbols

    # Fetch 2 years of data
    end_date = as_of_date
    start_date = end_date - timedelta(days=365 * 2)

    data_service = DataService(config)
    logger.info(f"Fetching data: {start_date} to {end_date}")
    # Ensure SPY is included for regime filter and baseline
    df = data_service.get_historical_data(
        training_universe,
        start_date.strftime('%Y-%m-%d'),
        end_date.strftime('%Y-%m-%d'),
        ensure_spy=True
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
        'top_n': 25,
        'max_weight': 0.10,
        'weight_method': 'inverse_vol',
        'cost_bps': 15,
        'slippage_bps': 5,
        'turnover_buffer_pct': 1.0,
        'rebalance_freq_days': 20,
        'regime_filter': True,
        'retrain_each_rebalance': True  # Enable walk-forward retraining
    }

    backtester = WalkForwardBacktest(backtest_config)

    # Split into training warm-up and evaluation window
    # Use first year for warm-up, rest for evaluation
    warmup_days = 252
    backtest_start_date = start_date + timedelta(days=warmup_days)
    backtest_start = backtest_start_date.strftime('%Y-%m-%d')
    backtest_end = end_date.strftime('%Y-%m-%d')

    logger.info(f"Backtest evaluation window: {backtest_start} to {backtest_end}")
    logger.info(f"Training warm-up period: {start_date} to {backtest_start_date}")

    # Create trainer function for walk-forward retraining
    def train_strategy_at_date(train_data, as_of_date):
        """Train strategy on data up to as_of_date."""
        strategy = SimpleStrategy(config, horizon='20d')
        
        # Check if we have enough raw data before computing features
        # Need: 252 days for 12-month momentum + 20 days for target + 20 days embargo = ~292 days minimum
        dates = pd.to_datetime(train_data.index.get_level_values(0)).unique()
        n_dates = len(dates)
        
        if n_dates < 300:  # Require at least 300 trading days (~14 months)
            logger.warning(f"Insufficient historical data at {as_of_date}: {n_dates} days (need ~300), using existing model")
            strategy.load_model('20d')
            return strategy
        
        # Compute features
        df_train_features = strategy.compute_features(train_data)
        
        # Compute target
        grouped = df_train_features.groupby(level=1)
        df_train_features['target_20d'] = grouped['close'].transform(lambda x: x.shift(-20) / x - 1)
        
        # Train
        df_train = df_train_features.dropna(subset=['target_20d'])
        
        # More realistic threshold: need at least 50 stocks * 50 days = 2,500 samples
        # Or at least 30 days with data across multiple stocks
        n_stocks = len(df_train.index.get_level_values(1).unique())
        n_dates_with_data = len(df_train.index.get_level_values(0).unique())
        
        min_samples = max(2000, n_stocks * 30)  # At least 30 days per stock on average
        
        if len(df_train) < min_samples:
            logger.warning(
                f"Insufficient training data at {as_of_date}: "
                f"{len(df_train)} samples ({n_stocks} stocks, {n_dates_with_data} dates) "
                f"(need {min_samples}), using existing model"
            )
            strategy.load_model('20d')
            return strategy
        
        logger.info(f"Retraining with {len(df_train)} samples ({n_stocks} stocks, {n_dates_with_data} dates)")
        strategy.train(df_train, target_col='target_20d', embargo_days=20)
        return strategy

    # Initial model training
    strategy_20d = SimpleStrategy(config, horizon='20d')
    strategy_20d.load_model('20d')

    candidate_results = backtester.run_backtest(
        strategy_20d,
        df_features,
        backtest_start,
        backtest_end,
        trainer=train_strategy_at_date if backtest_config['retrain_each_rebalance'] else None
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

    # SPY buy-and-hold baseline
    spy_baseline = {}
    try:
        spy_data = df_features.xs('SPY', level=1)
        spy_data = spy_data[(spy_data.index >= backtest_start) & (spy_data.index <= backtest_end)]
        
        if len(spy_data) > 1:
            spy_returns = spy_data['close'].pct_change().dropna()
            spy_total_return = (spy_data['close'].iloc[-1] / spy_data['close'].iloc[0]) - 1
            years = len(spy_data) / 252
            spy_cagr = (1 + spy_total_return) ** (1 / years) - 1 if years > 0 else 0
            spy_sharpe = spy_returns.mean() / spy_returns.std() * np.sqrt(252) if spy_returns.std() > 0 else 0
            
            spy_equity = spy_data['close'] / spy_data['close'].iloc[0]
            spy_cummax = spy_equity.cummax()
            spy_drawdowns = (spy_equity - spy_cummax) / spy_cummax
            spy_maxdd = spy_drawdowns.min()
            
            spy_baseline = {
                'cagr': spy_cagr,
                'sharpe': spy_sharpe,
                'max_drawdown': spy_maxdd,
                'total_return': spy_total_return
            }
            logger.info(f"SPY baseline: CAGR {spy_cagr*100:.1f}%, Sharpe {spy_sharpe:.2f}, MaxDD {spy_maxdd*100:.1f}%")
    except Exception as e:
        logger.warning(f"Could not compute SPY baseline: {e}")

    baselines = {
        'pure_momentum': momentum_results
    }
    
    if spy_baseline:
        baselines['spy_buy_hold'] = spy_baseline

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

    # Update state (from workspace, not state-repo - workflow copies it)
    state_manager = StateManager(state_file_path="./latest_state.json")
    state = state_manager.load_state()

    candidate_version = f"multi_horizon_{as_of_date.strftime('%Y%m%d')}"
    
    # Get strategy type from config
    strategy_type = config.get('strategy', {}).get('strategy_type', 'simple')

    state['models']['candidate_model'] = {
        'version': candidate_version,
        'strategy_type': strategy_type,  # Include strategy type
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
        gate_details=gate_details,
        strategy_type='simple'  # Current strategy type
    )

    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info(f"Candidate: {candidate_version}")
    logger.info(f"Gate: {'PASSED' if passed else 'FAILED'}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
