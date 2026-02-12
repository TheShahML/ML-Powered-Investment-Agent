#!/usr/bin/env python3
"""Generate daily signals from configured trading strategy + canary baseline."""
import os
import sys
import json
import re
from datetime import timedelta
from loguru import logger
from datetime import datetime, timezone

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


def _write_run_log(kind: str, payload: dict, as_of_date: str | None = None) -> str:
    logs_dir = os.path.join("reports", "run_logs")
    os.makedirs(logs_dir, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if as_of_date:
        safe_as_of = re.sub(r"[^0-9A-Za-z_-]", "_", str(as_of_date))
        name = f"{kind}_{safe_as_of}_{stamp}.json"
    else:
        name = f"{kind}_{stamp}.json"
    path = os.path.join(logs_dir, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    latest_path = os.path.join(logs_dir, f"latest_{kind}.json")
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    logger.info(f"Run log saved: {path}")
    return path


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
        _write_run_log("signals", {
            "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "status": "failed",
            "reason": "no_data",
            "as_of_date": str(as_of_date),
            "universe_size": len(symbols),
        }, as_of_date=str(as_of_date))
        sys.exit(1)

    # Verify freshness
    freshness_tolerance_days = int(config.get('freshness_tolerance_days', 0) or 0)
    max_stale_symbols = int(config.get('max_stale_symbols', 3) or 3)
    max_stale_pct = float(config.get('max_stale_pct', 0.002) or 0.002)  # 0.2% default
    is_fresh, freshness_details = calendar.verify_data_freshness(
        df,
        expected_date=as_of_date,
        symbols=symbols,
        tolerance_days=freshness_tolerance_days
    )

    if not is_fresh:
        stale_symbols = [x.get('symbol') for x in freshness_details.get('stale_symbols', []) if isinstance(x, dict) and x.get('symbol')]
        missing_symbols = list(freshness_details.get('missing_symbols', []))
        broken_symbols = sorted(set(stale_symbols + missing_symbols))
        broken_count = len(broken_symbols)
        total_count = max(1, len(symbols))
        stale_ratio = broken_count / total_count

        allow_degraded = (broken_count <= max_stale_symbols) or (stale_ratio <= max_stale_pct)

        if allow_degraded:
            logger.warning(
                f"Data freshness degraded but within tolerance; excluding {broken_count}/{total_count} symbols "
                f"(max_stale_symbols={max_stale_symbols}, max_stale_pct={max_stale_pct:.4f})"
            )
            # Drop stale/missing symbols from downstream processing
            symbols = [s for s in symbols if s not in set(broken_symbols)]
            if symbols:
                df = df[df.index.get_level_values(1).isin(symbols + ['SPY'])]

            discord = DiscordProductionNotifier()
            discord.send_health_alert([
                "Data freshness degraded (continuing with reduced universe)",
                f"Excluded symbols: {broken_count}",
                f"Remaining symbols: {len(symbols)}",
                f"Sample excluded: {', '.join(broken_symbols[:10]) if broken_symbols else 'N/A'}"
            ], severity="warning")
            is_fresh = True
        else:
            logger.error("Data is STALE - blocking signal generation")
            discord = DiscordProductionNotifier()
            discord.send_health_alert([
                "Stale data detected",
                f"Stale symbols: {len(freshness_details.get('stale_symbols', []))}",
                f"Missing symbols: {len(freshness_details.get('missing_symbols', []))}",
                f"Thresholds: max_stale_symbols={max_stale_symbols}, max_stale_pct={max_stale_pct:.4f}"
            ], severity="error")
            _write_run_log("signals", {
                "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "status": "failed",
                "reason": "stale_data_block",
                "as_of_date": str(as_of_date),
                "universe_size": len(symbols),
                "freshness_details": freshness_details,
            }, as_of_date=str(as_of_date))
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
        _write_run_log("signals", {
            "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "status": "failed",
            "reason": "no_primary_signals",
            "as_of_date": str(as_of_date),
            "strategy_name": strategy_name,
            "universe_size": len(symbols),
        }, as_of_date=str(as_of_date))
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

    # Send Discord summary
    discord = DiscordProductionNotifier()

    # Prepare top 10 for each horizon
    ml_top10_1d = [(sym, row['score']) for sym, row in signals_1d.head(10).iterrows()] if signals_1d is not None else []
    ml_top10_5d = [(sym, row['score']) for sym, row in signals_5d.head(10).iterrows()] if signals_5d is not None else []
    ml_top10_20d = [(sym, row['score']) for sym, row in signals_20d.head(10).iterrows()]
    canary_top10 = [(sym, row['score']*100) for sym, row in canary_signals.head(10).iterrows()]

    candidate_approved = state.get('models', {}).get('candidate_model', {}).get('approved_for_next_rebalance', False)

    if strategy_name == 'lc_reversal' and {'side', 'ret_1d', 'vol_z', 'impact_z'}.issubset(set(signals_20d.columns)):
        long_candidates = []
        short_candidates = []
        for sym, row in signals_20d.iterrows():
            item = (
                sym,
                float(row.get('ret_1d', 0.0)),
                float(row.get('vol_z', 0.0)),
                float(row.get('impact_z', 0.0)),
            )
            side = str(row.get('side', '')).lower()
            if side == 'buy':
                long_candidates.append(item)
            elif side == 'sell':
                short_candidates.append(item)

        discord.send_lc_reversal_signals(
            as_of_date=as_of_date.isoformat(),
            universe_size=len(symbols),
            long_candidates=long_candidates,
            short_candidates=short_candidates,
            canary_top10=canary_top10,
            performance_since_rebal=perf_since_rebal,
            performance_rolling=perf_rolling,
            data_fresh=is_fresh,
            days_until_rebal=state.get('rebalance', {}).get('days_until_rebalance', 20),
            next_rebal_date=state.get('rebalance', {}).get('next_rebalance_date', 'TBD')
        )
    else:
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

    _write_run_log("signals", {
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "success",
        "as_of_date": str(as_of_date),
        "strategy_name": strategy_name,
        "universe_size": len(symbols),
        "signals_count": {
            "1d": len(signals_1d) if signals_1d is not None else 0,
            "5d": len(signals_5d) if signals_5d is not None else 0,
            "20d": len(signals_20d) if signals_20d is not None else 0,
        },
        "canary_count": len(canary_signals) if canary_signals is not None else 0,
        "data_fresh": bool(is_fresh),
        "dashboard_path": dashboard_path,
    }, as_of_date=str(as_of_date))

    logger.info("=" * 60)
    logger.info("SIGNAL GENERATION COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
