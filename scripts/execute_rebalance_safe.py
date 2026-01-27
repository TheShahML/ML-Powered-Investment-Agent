#!/usr/bin/env python3
"""Safe monthly rebalance execution with all safety controls."""
import os
import sys
import argparse
from datetime import date
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config
from src.trading_calendar import TradingCalendar
from src.universe import Universe
from src.data_service import DataService
from src.state_manager import StateManager
from src.file_storage import FileStorage
from src.portfolio_constructor import PortfolioConstructor
from src.shadow_tracker import ShadowPortfolioTracker
from src.execution_safe import (
    check_market_open,
    check_kill_switch,
    execute_orders_safe,
    get_current_positions,
    get_account_equity
)
from src.discord_prod import DiscordProductionNotifier
from src.reporting.dashboard import DashboardGenerator
from datetime import timedelta
from typing import List
# Fix alpaca import issue
from src import alpaca_fix
import alpaca_trade_api as tradeapi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--force', action='store_true', help='Force execution even if rebalance is not due (testing only)')
    parser.add_argument('--ignore-market-closed', action='store_true', help='Continue even if market is closed (testing only)')
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("MONTHLY REBALANCE CHECK")
    logger.info(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE'}")
    logger.info("=" * 60)

    # Load config and API
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

    # Load state (from workspace, not state-repo - workflow copies it)
    state_manager = StateManager(state_file_path="./latest_state.json")
    state = state_manager.load_state()

    # Check if rebalance due
    if (not args.force) and (not state_manager.check_rebalance_due(threshold=20)):
        days_until = state.get('rebalance', {}).get('days_until_rebalance', 20)
        logger.info(f"Rebalance NOT due ({days_until} days remaining)")
        sys.exit(0)

    logger.info("Rebalance IS DUE - proceeding with checks" if not args.force else "FORCED REBALANCE (testing) - proceeding with checks")

    # Check idempotency
    if state_manager.check_already_rebalanced(as_of_date):
        logger.warning("Already rebalanced today - exiting")
        sys.exit(0)

    # Check market open
    if not check_market_open(api):
        if args.ignore_market_closed:
            logger.warning("Market is CLOSED, but --ignore-market-closed was set (testing) - continuing")
        else:
            logger.error("Market is CLOSED - cannot rebalance")
            discord = DiscordProductionNotifier()
            discord.send_monthly_rebalance_execution(
                broker_mode=os.environ.get('BROKER_MODE', 'paper'),
                as_of_date=as_of_date.isoformat(),
                model_promoted=None,
                strategy_type=(state.get('models', {}).get('active_model', {}) or {}).get('strategy_type')
                if 'state' in locals()
                else None,
                equity_allocation=0,
                btc_allocation=0,
                portfolio_value=0,
                orders=[],
                kill_switch=False,
                market_closed=True,
                dry_run=args.dry_run,
            )
            sys.exit(1)

    # Check kill switch
    kill_switch_enabled, kill_reason = check_kill_switch()
    if kill_switch_enabled:
        logger.error(f"KILL SWITCH ENABLED: {kill_reason}")
        discord = DiscordProductionNotifier()
        discord.send_monthly_rebalance_execution(
            broker_mode=os.environ.get('BROKER_MODE', 'paper'),
            as_of_date=as_of_date.isoformat(),
            model_promoted=None,
            strategy_type=(state.get('models', {}).get('active_model', {}) or {}).get('strategy_type') if 'state' in locals() else None,
            equity_allocation=0,
            btc_allocation=0,
            portfolio_value=0,
            orders=[],
            kill_switch=True,
            market_closed=False,
            dry_run=args.dry_run
        )
        sys.exit(1)

    # Get account info
    portfolio_value = get_account_equity(api)
    current_positions = get_current_positions(api)

    logger.info(f"Portfolio value: ${portfolio_value:,.0f}")
    logger.info(f"Current positions: {len(current_positions)}")

    # Load latest signals
    storage = FileStorage()
    signals = storage.get_latest_signals()

    if signals is None or signals.empty:
        logger.error("No signals available!")
        sys.exit(1)

    # Fetch data for portfolio construction (need historical data for inverse-vol)
    universe_builder = Universe(api, config)
    symbols, _ = universe_builder.build_universe()
    
    # Fetch recent data (need for inverse-vol calculation)
    end_date = as_of_date
    start_date = end_date - timedelta(days=365)
    
    data_service = DataService(config)
    crypto_tickers: List[str] = config.get('crypto_tickers', ['BTC/USD']) if config.get('trade_crypto', False) else []
    tickers_for_fetch = list(dict.fromkeys(symbols + crypto_tickers))
    df = data_service.get_historical_data(
        tickers_for_fetch,
        start_date.strftime('%Y-%m-%d'),
        end_date.strftime('%Y-%m-%d'),
        ensure_spy=True  # SPY required for regime filter
    )
    
    if df.empty:
        logger.error("No data available for portfolio construction!")
        sys.exit(1)
    
    # Get SPY data for regime filter
    spy_data = None
    try:
        spy_data = df.xs('SPY', level=1)
    except KeyError:
        logger.error("SPY not found in data - required for regime filter")
        sys.exit(1)
    
    # Compute features if needed
    from src.strategy_simple import SimpleStrategy
    strategy = SimpleStrategy(config, horizon='20d')
    df_features = strategy.compute_features(df)
    
    # Get current weights from positions
    current_weights = {}
    crypto_alpaca_symbols = {t.replace('/', '') for t in crypto_tickers}
    for symbol, qty in current_positions.items():
        # Skip crypto positions when building equity weights (handled separately)
        if symbol in crypto_alpaca_symbols:
            continue
        try:
            quote = api.get_latest_trade(symbol, feed='iex')
            price = float(quote.price)
            dollar_value = qty * price
            current_weights[symbol] = dollar_value / portfolio_value if portfolio_value > 0 else 0.0
        except:
            continue
    
    # Use PortfolioConstructor to compute target weights
    portfolio_constructor = PortfolioConstructor(
        top_n=25,
        max_weight=0.10,
        vol_window=60,
        turnover_buffer_pct=1.0,
        regime_filter=True,
        regime_scale=0.5
    )
    
    target_weights, metadata = portfolio_constructor.compute_target_weights(
        signals,
        df_features,
        as_of_date,
        current_weights=current_weights,
        spy_data=spy_data
    )
    
    logger.info(f"Target: {len(target_weights)} positions")
    logger.info(f"Regime: {metadata.get('regime', {}).get('regime', 'N/A')}")

    # BTC sleeve (dynamic 5–15% based on momentum)
    btc_allocation = 0.0
    crypto_target_allocations = {}
    crypto_latest_prices = {}
    if config.get('trade_crypto', False) and crypto_tickers:
        from src.crypto_strategy import BitcoinMomentumStrategy, CryptoStrategy, get_crypto_orders

        crypto_strategy_type = (config.get('crypto_allocation_strategy') or 'momentum').lower()

        if crypto_strategy_type == 'fixed':
            fixed = CryptoStrategy(config)
            # crypto_allocation is fixed (e.g., 10%)
            btc_allocation = float(config.get('crypto_allocation', 0.10))
            crypto_target_allocations = fixed.get_target_allocation(portfolio_value)
        else:
            # Default: momentum-based allocation (clipped via crypto_min_allocation/crypto_max_allocation)
            btc_prices = None
            try:
                if 'BTC/USD' in df_features.index.get_level_values(1):
                    btc_series = df_features.xs('BTC/USD', level=1)
                    if 'close' in btc_series.columns and len(btc_series) > 0:
                        btc_prices = btc_series['close'].astype(float).sort_index()
                        crypto_latest_prices['BTC/USD'] = float(btc_prices.iloc[-1])
            except Exception as e:
                logger.warning(f"Could not extract BTC price series for momentum: {e}")

            momentum = BitcoinMomentumStrategy(config)
            btc_allocation = float(momentum.get_dynamic_allocation(btc_prices)) if btc_prices is not None else float(config.get('crypto_allocation', 0.10))

            per_asset = (portfolio_value * btc_allocation) / len(crypto_tickers)
            crypto_target_allocations = {t: per_asset for t in crypto_tickers}

        logger.info(f"BTC sleeve target allocation: {btc_allocation*100:.1f}% ({crypto_strategy_type})")

    # Scale equity weights to leave room for BTC sleeve
    equity_scale = max(0.0, 1.0 - btc_allocation)
    if target_weights and equity_scale < 1.0:
        target_weights = {s: w * equity_scale for s, w in target_weights.items()}

    # Execute orders
    exec_config = {
        'max_orders_per_rebalance': 100,
        'max_daily_notional': 1_000_000,
        'min_trade_notional': 10,
        'turnover_buffer_pct': 1.0
    }

    orders = execute_orders_safe(
        api,
        target_weights,
        {s: q for s, q in current_positions.items() if s not in crypto_alpaca_symbols},
        portfolio_value,
        exec_config,
        dry_run=args.dry_run
    )

    # Execute crypto orders (separate endpoint)
    crypto_orders = []
    if crypto_target_allocations:
        from src.crypto_strategy import get_crypto_orders
        crypto_orders = get_crypto_orders(api, crypto_target_allocations, latest_prices=crypto_latest_prices, dry_run=args.dry_run)

    # Combine for reporting
    orders = (orders or []) + (crypto_orders or [])

    logger.info(f"Executed {len(orders)} orders")

    # Promote candidate if approved
    model_promoted = None
    candidate = state.get('models', {}).get('candidate_model', {})
    if candidate.get('approved_for_next_rebalance'):
        logger.info(f"Promoting candidate: {candidate.get('version')}")
        state_manager.promote_candidate_to_active()
        
        # Include strategy type in model_promoted string
        strategy_type = candidate.get('strategy_type') or state.get('strategies', {}).get('best') or 'simple'
        model_version = candidate.get('version', 'unknown')
        model_promoted = f"{model_version} ({strategy_type})"

    # Update shadow portfolios on rebalance
    shadow_tracker = ShadowPortfolioTracker(config)
    shadow_state = shadow_tracker.initialize_from_state(state)
    
    # Get canary signals
    from src.strategy_simple import PureMomentumStrategy
    canary_strategy = PureMomentumStrategy(config)
    canary_signals = canary_strategy.compute_signals(df_features)
    
    # Rebalance shadow portfolios
    shadow_state = shadow_tracker.rebalance_shadow(
        shadow_state,
        signals,
        canary_signals,
        df_features,
        as_of_date,
        spy_data=spy_data
    )
    
    # Update state with shadow portfolios
    state['shadow_portfolios'] = shadow_state
    state_manager.save_state(state)
    
    # Update rebalance schedule
    if not args.dry_run:
        state_manager.update_rebalance_schedule(as_of_date, rebalance_freq=20)
    
    # Compute allocations
    equity_allocation = sum(target_weights.values())
    # btc_allocation set above (0.0 if disabled)
    
    # Generate trades dashboard
    dashboard_gen = DashboardGenerator()
    trades_dashboard_path = dashboard_gen.generate_trades_dashboard(
        as_of_date=as_of_date,
        target_weights=target_weights,
        orders=orders
    )

    # Send Discord notification
    discord = DiscordProductionNotifier()
    # Determine strategy type for reporting (active or promoted candidate)
    strategy_type = None
    try:
        active = state.get('models', {}).get('active_model') or {}
        strategy_type = active.get('strategy_type') or state.get('strategies', {}).get('best')
        if candidate.get('approved_for_next_rebalance') and candidate.get('strategy_type'):
            # We are promoting this strategy on rebalance (or would in dry-run)
            strategy_type = candidate.get('strategy_type')
    except Exception:
        pass

    discord.send_monthly_rebalance_execution(
        broker_mode=os.environ.get('BROKER_MODE', 'paper'),
        as_of_date=as_of_date.isoformat(),
        model_promoted=model_promoted,
        strategy_type=strategy_type,
        equity_allocation=equity_allocation,
        btc_allocation=btc_allocation,
        portfolio_value=portfolio_value,
        orders=orders,
        kill_switch=False,
        market_closed=False,
        dry_run=args.dry_run
    )
    
    # Send trades dashboard image
    discord.send_image("📊 Rebalance Trades", trades_dashboard_path)

    logger.info("=" * 60)
    logger.info("REBALANCE COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
