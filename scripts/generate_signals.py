#!/usr/bin/env python3
"""Generate daily signals from configured trading strategy + canary baseline."""
import os
import sys
from datetime import timedelta
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config
from src.trading_calendar import TradingCalendar
from src.universe import Universe
from src.data_service import DataService
from src.canary_tracker import CanaryTracker
from src.shadow_tracker import ShadowPortfolioTracker
from src.state_manager import StateManager
from src.file_storage import FileStorage
from src.discord_prod import DiscordProductionNotifier
from src.reporting.dashboard import DashboardGenerator
from src.strategies.lc_reversal import LCReversalStrategy
# Fix alpaca import issue
from src import alpaca_fix
import alpaca_trade_api as tradeapi


def main():
    logger.info("=" * 60)
    logger.info("DAILY SIGNAL GENERATION")
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

    # Check data freshness
    universe_builder = Universe(api, config)
    symbols, _ = universe_builder.build_universe()

    # Fetch recent data
    end_date = as_of_date
    start_date = end_date - timedelta(days=365 * 2)

    data_service = DataService(config)
    # Ensure SPY is included for regime filter
    df = data_service.get_historical_data(
        symbols,
        start_date.strftime('%Y-%m-%d'),
        end_date.strftime('%Y-%m-%d'),
        ensure_spy=True
    )

    if df.empty:
        logger.error("No data!")
        sys.exit(1)

    # Verify freshness
    is_fresh, freshness_details = calendar.verify_data_freshness(
        df,
        expected_date=as_of_date,
        symbols=symbols,
        tolerance_days=0
    )

    if not is_fresh:
        logger.error("Data is STALE - blocking signal generation")
        discord = DiscordProductionNotifier()
        discord.send_health_alert([
            "Stale data detected",
            f"Stale symbols: {len(freshness_details.get('stale_symbols', []))}",
            f"Missing symbols: {len(freshness_details.get('missing_symbols', []))}"
        ], severity="error")
        sys.exit(1)

    # Load state (from workspace, not state-repo - workflow copies it)
    state_manager = StateManager(state_file_path="./latest_state.json")
    state = state_manager.load_state()

    # Load active model
    active_model = state.get('models', {}).get('active_model')
    if not active_model:
        logger.warning("No active model, using fallback")
        active_model = {'version': 'fallback', 'strategy_type': 'simple'}
    
    strategy_name = config.get('strategy_name', 'lc_reversal')
    logger.info(f"Configured strategy_name={strategy_name}")

    if strategy_name == 'lc_reversal':
        active_model = {'version': 'lc_reversal', 'strategy_type': 'lc_reversal'}
        strategy = LCReversalStrategy(config)
        df_features = strategy.compute_features(df)
        signals_20d = strategy.compute_signals(df_features)
        signals_1d = signals_20d
        signals_5d = signals_20d
        strategy_type = 'lc_reversal'
        logger.info(f"Generated LC-Reversal signals: {len(signals_20d)} rows")
    else:
        # Legacy ML momentum path
        strategy_type = active_model.get('strategy_type') or state.get('strategies', {}).get('best') or config.get('strategy', {}).get('strategy_type', 'simple')
        if strategy_name == 'ml_momentum':
            strategy_type = 'simple'
        active_model['strategy_type'] = strategy_type

        logger.info(f"Generating multi-horizon signals (active: {active_model.get('version')}, strategy: {strategy_type})")
        from src.strategy_selector import StrategySelector
        selector = StrategySelector(config)
        strategy = selector.get_strategy(strategy_type, horizon='20d')
        df_features = strategy.compute_features(df)

        if hasattr(strategy, 'compute_multi_horizon_signals'):
            multi_signals = strategy.compute_multi_horizon_signals(df_features)
            signals_1d = multi_signals.get('1d')
            signals_5d = multi_signals.get('5d')
            signals_20d = multi_signals.get('20d')  # PRIMARY for rebalancing
        else:
            signals_20d = strategy.compute_signals(df_features)
            signals_1d = signals_20d
            signals_5d = signals_20d

    if signals_20d is None or signals_20d.empty:
        logger.error("20-day model (primary) missing or failed!")
        sys.exit(1)

    logger.info(f"✓ 1d signals: {len(signals_1d) if signals_1d is not None else 0} stocks")
    logger.info(f"✓ 5d signals: {len(signals_5d) if signals_5d is not None else 0} stocks")
    logger.info(f"✓ 20d signals: {len(signals_20d)} stocks (PRIMARY)")

    # Generate benchmark momentum (canary) signals
    logger.info("Generating benchmark momentum (canary) signals")
    canary = CanaryTracker(config)
    canary_signals = canary.compute_momentum_signals(df, top_n=25)

    # Save signals (20d as primary, others for reference)
    storage = FileStorage()
    storage.save_signals(signals_20d, as_of_date)  # Primary for rebalancing

    # Update shadow portfolios
    shadow_tracker = ShadowPortfolioTracker(config)
    shadow_state = shadow_tracker.initialize_from_state(state)
    
    # Get latest prices for all symbols + SPY + BTC
    latest_prices = {}
    spy_price = None
    btc_price = None
    
    try:
        for symbol in symbols + ['SPY']:
            try:
                symbol_data = df.xs(symbol, level=1)
                if len(symbol_data) > 0:
                    latest_prices[symbol] = float(symbol_data['close'].iloc[-1])
                    if symbol == 'SPY':
                        spy_price = latest_prices[symbol]
            except:
                continue
        
        # Try to get BTC price if enabled
        if config.get('trade_crypto', False):
            try:
                btc_data = df.xs('BTC/USD', level=1) if 'BTC/USD' in df.index.get_level_values(1) else None
                if btc_data is not None and len(btc_data) > 0:
                    btc_price = float(btc_data['close'].iloc[-1])
            except:
                pass
    except Exception as e:
        logger.warning(f"Error getting latest prices: {e}")

    # Update shadow portfolios daily
    shadow_state = shadow_tracker.update_daily(
        shadow_state,
        as_of_date,
        latest_prices,
        spy_price=spy_price,
        btc_price=btc_price
    )
    
    # Compute performance metrics
    performance_metrics = shadow_tracker.compute_performance_metrics(shadow_state, as_of_date)
    
    # Update state with shadow portfolios
    state['shadow_portfolios'] = shadow_state
    state_manager.save_state(state)

    # Get performance metrics for Discord
    perf_since_rebal = performance_metrics.get('since_rebalance', {
        'ml': 0.0,
        'canary': 0.0,
        'spy': 0.0,
        'days': state.get('rebalance', {}).get('days_since_rebalance', 0)
    })

    perf_rolling = performance_metrics.get('rolling_30d', {
        'ml': 0.0,
        'canary': 0.0,
        'spy': 0.0
    })

    # Increment day counter
    state_manager.increment_day_counter()
    
    # Generate dashboard
    dashboard_gen = DashboardGenerator()
    
    # Get current holdings from state (if available)
    current_holdings = shadow_state.get('ml', {}).get('weights', {})
    
    # Get broker mode and kill switch
    broker_mode = os.environ.get('BROKER_MODE', 'paper')
    kill_switch_enabled = os.environ.get('KILL_SWITCH_ENABLED', 'false').lower() == 'true'
    
    dashboard_path = dashboard_gen.generate_daily_dashboard(
        as_of_date=as_of_date,
        active_model=active_model,
        ml_top10=[(sym, row['score']) for sym, row in signals_20d.head(10).iterrows()],
        canary_top10=[(sym, row['score']*100) for sym, row in canary_signals.head(10).iterrows()],
        current_holdings=current_holdings,
        shadow_state=shadow_state,
        performance_metrics=performance_metrics,
        broker_mode=broker_mode,
        kill_switch=kill_switch_enabled
    )

    # Send Discord summary (multi-horizon)
    discord = DiscordProductionNotifier()

    # Prepare top 10 for each horizon
    ml_top10_1d = [(sym, row['score']) for sym, row in signals_1d.head(10).iterrows()] if signals_1d is not None else []
    ml_top10_5d = [(sym, row['score']) for sym, row in signals_5d.head(10).iterrows()] if signals_5d is not None else []
    ml_top10_20d = [(sym, row['score']) for sym, row in signals_20d.head(10).iterrows()]
    canary_top10 = [(sym, row['score']*100) for sym, row in canary_signals.head(10).iterrows()]

    candidate_approved = state.get('models', {}).get('candidate_model', {}).get('approved_for_next_rebalance', False)

    # Send Discord message with dashboard image
    discord.send_multi_horizon_signals(
        as_of_date=as_of_date.isoformat(),
        active_model=active_model,
        candidate_approved=candidate_approved,
        universe_size=len(symbols),
        ml_top10_1d=ml_top10_1d,
        ml_top10_5d=ml_top10_5d,
        ml_top10_20d=ml_top10_20d,
        canary_top10=canary_top10,
        performance_since_rebal=perf_since_rebal,
        performance_rolling=perf_rolling,
        data_fresh=is_fresh,
        days_until_rebal=state.get('rebalance', {}).get('days_until_rebalance', 20),
        next_rebal_date=state.get('rebalance', {}).get('next_rebalance_date', 'TBD')
    )
    
    # Send dashboard image
    discord.send_image("📊 Daily Dashboard", dashboard_path)

    logger.info("=" * 60)
    logger.info("SIGNAL GENERATION COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
