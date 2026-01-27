#!/usr/bin/env python3
"""Generate daily signals from active ML model + canary baseline."""
import os
import sys
from datetime import timedelta
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config
from src.trading_calendar import TradingCalendar
from src.universe import Universe
from src.data_service import DataService
from src.strategy_simple import SimpleStrategy
from src.canary_tracker import CanaryTracker
from src.state_manager import StateManager
from src.file_storage import FileStorage
from src.discord_prod import DiscordProductionNotifier
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
    df = data_service.get_historical_data(
        symbols,
        start_date.strftime('%Y-%m-%d'),
        end_date.strftime('%Y-%m-%d')
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

    # Load state
    state_manager = StateManager(state_file_path="state-repo/latest_state.json")
    state = state_manager.load_state()

    # Load active model
    active_model = state.get('models', {}).get('active_model')
    if not active_model:
        logger.warning("No active model, using fallback")
        active_model = {'version': 'fallback'}

    # Generate multi-horizon ML signals
    logger.info(f"Generating multi-horizon signals (active: {active_model.get('version')})")
    strategy = SimpleStrategy(config, horizon='20d')

    # Compute features once
    df_features = strategy.compute_features(df)

    # Generate signals from all 3 horizons
    multi_signals = strategy.compute_multi_horizon_signals(df_features)

    signals_1d = multi_signals.get('1d')
    signals_5d = multi_signals.get('5d')
    signals_20d = multi_signals.get('20d')  # PRIMARY for rebalancing

    if signals_20d is None or signals_20d.empty:
        logger.error("20-day model (primary) missing or failed!")
        sys.exit(1)

    logger.info(f"✓ 1d signals: {len(signals_1d) if signals_1d is not None else 0} stocks")
    logger.info(f"✓ 5d signals: {len(signals_5d) if signals_5d is not None else 0} stocks")
    logger.info(f"✓ 20d signals: {len(signals_20d)} stocks (PRIMARY)")

    # Generate canary signals
    logger.info("Generating canary signals (pure momentum)")
    canary = CanaryTracker(config)
    canary_signals = canary.compute_momentum_signals(df, top_n=25)

    # Save signals (20d as primary, others for reference)
    storage = FileStorage()
    storage.save_signals(signals_20d, as_of_date)  # Primary for rebalancing

    # Get performance metrics (stub - would compute from state)
    perf_since_rebal = {
        'actual': 0.0,
        'canary': 0.0,
        'spy': 0.0,
        'btc': 0.0,
        'days': state.get('rebalance', {}).get('days_since_rebalance', 0)
    }

    perf_rolling = {
        'actual_30d': 0.0,
        'canary_30d': 0.0,
        'spy_30d': 0.0
    }

    # Increment day counter
    state_manager.increment_day_counter()

    # Send Discord summary (multi-horizon)
    discord = DiscordProductionNotifier()

    # Prepare top 10 for each horizon
    ml_top10_1d = [(sym, row['score']) for sym, row in signals_1d.head(10).iterrows()] if signals_1d is not None else []
    ml_top10_5d = [(sym, row['score']) for sym, row in signals_5d.head(10).iterrows()] if signals_5d is not None else []
    ml_top10_20d = [(sym, row['score']) for sym, row in signals_20d.head(10).iterrows()]
    canary_top10 = [(sym, row['score']*100) for sym, row in canary_signals.head(10).iterrows()]

    candidate_approved = state.get('models', {}).get('candidate_model', {}).get('approved_for_next_rebalance', False)

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

    logger.info("=" * 60)
    logger.info("SIGNAL GENERATION COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
