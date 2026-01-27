#!/usr/bin/env python3
"""
Train multiple strategies and compare performance.

This script trains all available strategies (XGBoost, LSTM, etc.) and
compares their performance to enable dynamic strategy switching.
"""
import os
import sys
from datetime import datetime, timedelta
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config
from src.trading_calendar import TradingCalendar
from src.universe import Universe
from src.data_service import DataService
from src.strategy_selector import StrategySelector
from src.backtest_engine import WalkForwardBacktest, check_promotion_gate
from src.state_manager import StateManager
from src.discord_prod import DiscordProductionNotifier
# Fix alpaca import issue
from src import alpaca_fix
import alpaca_trade_api as tradeapi
import numpy as np

# Backtest data is large; keep a module-level reference for backtest_strategy()
df_global_for_backtest = None


def compute_buy_hold_metrics(
    df: 'pd.DataFrame',
    symbol: str,
    start_date: datetime,
    end_date: datetime,
    initial_capital: float = 100000.0
) -> dict:
    """
    Compute buy-and-hold metrics for a single symbol over the backtest window.
    Mirrors the metrics produced by WalkForwardBacktest._compute_metrics().
    """
    import pandas as pd

    try:
        sym = df.xs(symbol, level=1).copy()
    except Exception:
        return {}

    # Handle tz-aware vs tz-naive comparisons (Alpaca bars are often UTC tz-aware)
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    idx_tz = getattr(sym.index, "tz", None)
    if idx_tz is not None:
        if start_ts.tzinfo is None:
            start_ts = start_ts.tz_localize(idx_tz)
        if end_ts.tzinfo is None:
            end_ts = end_ts.tz_localize(idx_tz)

    sym = sym[(sym.index >= start_ts) & (sym.index <= end_ts)]
    if len(sym) < 2 or 'close' not in sym.columns:
        return {}

    close = sym['close'].astype(float)
    equity = (close / close.iloc[0]) * float(initial_capital)

    returns = equity.pct_change().dropna().to_numpy()
    if len(returns) < 1:
        return {}

    total_return = float((equity.iloc[-1] / equity.iloc[0]) - 1.0)
    years = len(equity) / 252.0
    cagr = float((1 + total_return) ** (1 / years) - 1) if years > 0 else 0.0

    vol = float(np.std(returns))
    sharpe = float(np.mean(returns) / vol * np.sqrt(252)) if vol > 0 else 0.0

    eq_np = equity.to_numpy()
    cummax = np.maximum.accumulate(eq_np)
    drawdowns = (eq_np - cummax) / cummax
    max_dd = float(np.min(drawdowns)) if len(drawdowns) else 0.0

    monthly_returns = []
    for i in range(0, len(returns), 21):
        chunk = returns[i:i+21]
        if len(chunk) > 0:
            monthly_returns.append(float(np.prod(1 + chunk) - 1))
    worst_month = float(np.min(monthly_returns)) if monthly_returns else 0.0

    return {
        'cagr': cagr,
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'worst_month': worst_month,
        'total_return': total_return,
        'avg_turnover': 0.0,
        'total_trades': 0,
        'cost_drag_cumulative': 0.0,
        'final_value': float(equity.iloc[-1]),
        'equity_curve': equity.tolist(),
        'dates': [str(d) for d in equity.index]
    }


def train_all_strategies(
    config: dict,
    df: dict,
    horizons: list = ['1d', '5d', '20d'],
    strategies: list = ['simple']
) -> dict:
    """
    Train all specified strategies on all horizons.
    
    Returns:
        Dict mapping strategy_type -> horizon -> metrics
    """
    selector = StrategySelector(config)
    all_results = {}
    
    for strategy_type in strategies:
        if strategy_type not in selector.available_strategies:
            logger.warning(f"Strategy '{strategy_type}' not available, skipping")
            continue
        
        logger.info(f"\n{'='*60}")
        logger.info(f"TRAINING STRATEGY: {strategy_type.upper()}")
        logger.info(f"{'='*60}")
        
        strategy_results = {}
        
        for horizon in horizons:
            logger.info(f"\n--- Training {strategy_type} {horizon} model ---")
            
            try:
                strategy = selector.get_strategy(strategy_type, horizon)
                
                # Train
                target_col = f'target_{horizon}'
                if target_col not in df.columns:
                    logger.warning(f"Target {target_col} not found, skipping")
                    continue
                
                metrics = strategy.train(df, target_col=target_col)
                strategy_results[horizon] = metrics
                
                logger.info(f"{strategy_type} {horizon}: IC={metrics.get('cv_mean_ic', 0):.4f}")
                
            except TimeoutError as e:
                # LSTM can be expensive; skip it if it exceeds time budget and continue with XGBoost/momentum.
                logger.warning(f"Timeout training {strategy_type} ({horizon}): {e} — skipping {strategy_type} entirely")
                strategy_results = {}
                break
            except Exception as e:
                logger.error(f"Error training {strategy_type} {horizon}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Only keep strategy if we trained at least one horizon successfully
        if strategy_results:
            all_results[strategy_type] = strategy_results
        else:
            logger.warning(f"Strategy '{strategy_type}' produced no trained models; excluding from comparison")
    
    return all_results


def backtest_strategy(
    config: dict,
    strategy_type: str,
    horizon: str,
    start_date: datetime,
    end_date: datetime,
    universe: list
) -> dict:
    """Run walk-forward backtest for a strategy."""
    selector = StrategySelector(config)
    strategy = selector.get_strategy(strategy_type, horizon)
    
    # Load model
    if not strategy.load_model(horizon):
        logger.error(f"Could not load {strategy_type} {horizon} model")
        return None

    # Run backtest using repo's backtest engine API
    bt_cfg = {
        'top_n': config.get('strategy', {}).get('n_holdings', 25),
        'max_weight': config.get('risk', {}).get('max_position_weight', 0.10),
        'weight_method': 'inverse_vol',
        'cost_bps': 15.0,
        'slippage_bps': 5.0,
        'turnover_buffer_pct': 1.0,
        'rebalance_freq_days': 20,
        'regime_filter': True,
        'retrain_each_rebalance': False
    }
    backtest = WalkForwardBacktest(bt_cfg)
    return backtest.run_backtest(
        strategy=strategy,
        data=df_global_for_backtest,  # set in main()
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d'),
        initial_capital=100000.0,
        trainer=None
    )


def main():
    logger.info("=" * 60)
    logger.info("MULTI-STRATEGY TRAINING & COMPARISON")
    logger.info("=" * 60)
    
    # Load config
    config = load_config()
    
    # Get as-of date
    api = tradeapi.REST(
        config['ALPACA_API_KEY'],
        config['ALPACA_SECRET_KEY'],
        config['ALPACA_BASE_URL']
    )
    calendar = TradingCalendar(api)
    as_of_date = calendar.get_last_completed_trading_day()
    logger.info(f"As-of date: {as_of_date}")
    
    # Build universe
    logger.info("\nBuilding universe...")
    universe_builder = Universe(api, config)
    symbols, _ = universe_builder.build_universe()
    logger.info(f"Universe: {len(symbols)} stocks")
    
    # Fetch data
    end_date = as_of_date
    start_date = end_date - timedelta(days=730)  # 2 years
    
    logger.info(f"\nFetching data: {start_date.date()} to {end_date.date()}")
    data_service = DataService(config)
    # Ensure benchmarks are present for reporting + gates
    symbols_with_benchmarks = list(dict.fromkeys(symbols + ['SPY', 'QQQ', 'VTI']))
    df = data_service.get_historical_data(
        symbols_with_benchmarks,
        start_date.strftime('%Y-%m-%d'),
        end_date.strftime('%Y-%m-%d'),
        ensure_spy=True
    )
    
    # Compute features and targets
    logger.info("\nComputing features...")
    from src.strategy_simple import SimpleStrategy
    temp_strategy = SimpleStrategy(config)
    df = temp_strategy.compute_features(df)
    
    # Compute targets
    logger.info("Computing targets...")
    grouped = df.groupby(level=1)
    df['target_1d'] = grouped['close'].pct_change(1).shift(-1)
    df['target_5d'] = grouped['close'].pct_change(5).shift(-5)
    df['target_20d'] = grouped['close'].pct_change(20).shift(-20)

    # Make data available for backtest_strategy()
    global df_global_for_backtest
    df_global_for_backtest = df
    
    # Train all strategies
    # LSTM disabled for now (CPU-only TensorFlow on Windows was causing confusion/slowdowns)
    strategies_to_train = ['simple']
    
    all_training_results = train_all_strategies(
        config,
        df,
        horizons=['1d', '5d', '20d'],
        strategies=strategies_to_train
    )
    
    # Backtest each strategy (using 20d model)
    logger.info("\n" + "=" * 60)
    logger.info("RUNNING BACKTESTS")
    logger.info("=" * 60)
    
    backtest_start = start_date + timedelta(days=252)  # 1 year warm-up
    backtest_end = end_date
    
    backtest_results = {}
    
    for strategy_type in strategies_to_train:
        logger.info(f"\nBacktesting {strategy_type}...")
        try:
            results = backtest_strategy(
                config,
                strategy_type,
                '20d',
                backtest_start,
                backtest_end,
                symbols
            )
            if results:
                backtest_results[strategy_type] = results
                logger.info(f"{strategy_type}: CAGR={results.get('cagr', 0)*100:.1f}%, "
                          f"Sharpe={results.get('sharpe', 0):.2f}, "
                          f"MaxDD={results.get('max_drawdown', 0)*100:.1f}%")
        except Exception as e:
            logger.error(f"Error backtesting {strategy_type}: {e}")
            import traceback
            traceback.print_exc()
    
    # Compare and select best
    logger.info("\n" + "=" * 60)
    logger.info("STRATEGY COMPARISON")
    logger.info("=" * 60)
    
    # Backtest Pure Momentum as a *candidate strategy* (so it can win vs XGBoost)
    logger.info("\nBacktesting Pure Momentum candidate...")
    from src.strategy_simple import PureMomentumStrategy
    momentum_strategy = PureMomentumStrategy(config)
    bt_cfg_baseline = {
        'top_n': config.get('strategy', {}).get('n_holdings', 25),
        'max_weight': config.get('risk', {}).get('max_position_weight', 0.10),
        'weight_method': 'inverse_vol',
        'cost_bps': 15.0,
        'slippage_bps': 5.0,
        'turnover_buffer_pct': 1.0,
        'rebalance_freq_days': 20,
        'regime_filter': True,
        'retrain_each_rebalance': False
    }
    momentum_backtest = WalkForwardBacktest(bt_cfg_baseline)
    momentum_results = momentum_backtest.run_backtest(
        strategy=momentum_strategy,
        data=df_global_for_backtest,
        start_date=backtest_start.strftime('%Y-%m-%d'),
        end_date=backtest_end.strftime('%Y-%m-%d'),
        initial_capital=100000.0,
        trainer=None
    )

    if momentum_results:
        backtest_results['pure_momentum'] = momentum_results

    # Benchmarks (buy & hold)
    logger.info("\nComputing benchmark (buy & hold) metrics...")
    benchmarks = {}
    for bm in ['SPY', 'QQQ', 'VTI']:
        m = compute_buy_hold_metrics(df_global_for_backtest, bm, backtest_start, backtest_end, 100000.0)
        if m:
            benchmarks[bm] = m
            logger.info(f"{bm}: CAGR={m.get('cagr', 0)*100:.1f}%, Sharpe={m.get('sharpe', 0):.2f}, MaxDD={m.get('max_drawdown', 0)*100:.1f}%")
        else:
            logger.warning(f"Could not compute benchmark metrics for {bm} (missing data?)")
    
    # Select best strategy
    selector = StrategySelector(config)
    best_strategy = selector.select_best_strategy(backtest_results, benchmarks)
    
    logger.info(f"\n✓ Best strategy: {best_strategy}")
    
    # Check promotion gates for each strategy
    logger.info("\n" + "=" * 60)
    logger.info("PROMOTION GATE CHECKS")
    logger.info("=" * 60)
    gate_by_strategy = {}

    for strategy_type, results in backtest_results.items():
        passed, details = check_promotion_gate(
            results,
            benchmarks,
            sharpe_margin=0.2,
            maxdd_tolerance=0.05
        )
        gate_by_strategy[strategy_type] = {'passed': passed, 'details': details}
        logger.info(f"\n{strategy_type}: {'PASS' if passed else 'FAIL'}")
        logger.info(f"  Sharpe: {results.get('sharpe', 0):.3f} vs {details.get('best_baseline_sharpe', 0):.3f}")
        logger.info(f"  MaxDD: {results.get('max_drawdown', 0):.3f} vs {details.get('best_baseline_maxdd', 0):.3f}")

    # Send Discord notification with multi-strategy comparison (after gate checks)
    best_gate = gate_by_strategy.get(best_strategy, {})
    best_gate_passed = bool(best_gate.get('passed', False))
    best_gate_details = best_gate.get('details', {})

    discord = DiscordProductionNotifier()
    discord.send_multi_strategy_comparison(
        as_of_date=as_of_date.isoformat(),
        strategy_results=backtest_results,
        baselines=benchmarks,
        best_strategy=best_strategy,
        training_window=(start_date.isoformat(), end_date.isoformat()),
        gate_passed=best_gate_passed,
        gate_details=best_gate_details
    )
    
    # Save state
    state_manager = StateManager(state_file_path="./latest_state.json")
    state = state_manager.load_state()
    
    state['strategies'] = {
        'available': list(backtest_results.keys()),
        'best': best_strategy,
        'results': {
            strategy_type: {
                'sharpe': results.get('sharpe', 0),
                'cagr': results.get('cagr', 0),
                'maxdd': results.get('max_drawdown', 0)
            }
            for strategy_type, results in backtest_results.items()
        },
        'last_comparison_date': as_of_date.isoformat()
    }
    
    # Also update candidate model with best strategy info
    if backtest_results:
        best_results = backtest_results.get(best_strategy, {})
        best_gate = gate_by_strategy.get(best_strategy, {})
        approved = bool(best_gate.get('passed', False))
        state['models']['candidate_model'] = {
            'version': f"multi_strategy_{as_of_date.strftime('%Y%m%d')}",
            'strategy_type': best_strategy,
            'trained_date': as_of_date.isoformat(),
            'backtest_sharpe': best_results.get('sharpe', 0),
            'backtest_cagr': best_results.get('cagr', 0),
            'backtest_maxdd': best_results.get('max_drawdown', 0),
            'approved_for_next_rebalance': approved,
            'promotion_gate': best_gate.get('details', {})
        }
    
    state_manager.save_state(state)
    
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

