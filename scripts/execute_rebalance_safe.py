#!/usr/bin/env python3
"""Safe strategy execution with explicit observability and idempotency."""
import os
import sys
import argparse
import traceback
import json
import re
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any
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
    get_open_orders,
    cancel_open_orders,
)
from src.discord_prod import DiscordProductionNotifier
from src.strategies.lc_reversal import LCReversalStrategy
# Fix alpaca import issue
from src import alpaca_fix
import alpaca_trade_api as tradeapi


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _validate_execution_env(config: Dict[str, Any], dry_run: bool) -> str:
    broker_mode = (os.getenv("BROKER_MODE") or config.get("BROKER_MODE") or "paper").strip().lower()
    api_key = config.get("ALPACA_API_KEY")
    secret_key = config.get("ALPACA_SECRET_KEY")
    base_url = (config.get("ALPACA_BASE_URL") or "").strip()

    missing = [k for k, v in {
        "ALPACA_API_KEY": api_key,
        "ALPACA_SECRET_KEY": secret_key,
        "ALPACA_BASE_URL": base_url,
    }.items() if not v]
    if missing:
        raise RuntimeError(
            "Missing required Alpaca credentials/config: "
            + ", ".join(missing)
            + ". Set these in GitHub secrets or local env."
        )

    if broker_mode not in {"paper", "live"}:
        raise RuntimeError(f"Invalid BROKER_MODE={broker_mode}. Expected 'paper' or 'live'.")

    if broker_mode == "live" and ("paper" in base_url.lower()):
        raise RuntimeError(
            f"BROKER_MODE=live but ALPACA_BASE_URL points to paper endpoint: {base_url}"
        )
    if broker_mode == "paper" and ("paper" not in base_url.lower()):
        raise RuntimeError(
            f"BROKER_MODE=paper but ALPACA_BASE_URL does not look like paper endpoint: {base_url}"
        )

    if broker_mode == "live" and not dry_run and not _env_bool("I_ACKNOWLEDGE_LIVE_TRADING", False):
        raise RuntimeError(
            "Live trading blocked: set I_ACKNOWLEDGE_LIVE_TRADING=true to allow BROKER_MODE=live order submission."
        )

    return broker_mode


def _stack_snippet() -> str:
    return traceback.format_exc(limit=8)[-800:]


def _get_account_snapshot(api: tradeapi.REST) -> Dict[str, float]:
    account = api.get_account()
    cash = float(getattr(account, "cash", 0.0) or 0.0)
    equity = float(getattr(account, "equity", 0.0) or 0.0)
    long_mv = float(getattr(account, "long_market_value", 0.0) or 0.0)
    short_mv = float(getattr(account, "short_market_value", 0.0) or 0.0)
    gross = long_mv + abs(short_mv)
    net = long_mv - abs(short_mv)
    return {
        "cash": cash,
        "equity": equity,
        "gross_exposure": round(gross, 2),
        "net_exposure": round(net, 2),
    }


def _get_positions_top10(api: tradeapi.REST) -> List[Dict[str, Any]]:
    items = []
    for p in api.list_positions():
        try:
            mv = float(getattr(p, "market_value", 0.0) or 0.0)
        except Exception:
            mv = 0.0
        items.append({
            "symbol": getattr(p, "symbol", "N/A"),
            "qty": float(getattr(p, "qty", 0.0) or 0.0),
            "notional": abs(mv),
        })
    return sorted(items, key=lambda x: x["notional"], reverse=True)[:10]


def _enrich_order_statuses(api: tradeapi.REST, orders: List[Dict[str, Any]], dry_run: bool) -> List[Dict[str, Any]]:
    if dry_run:
        return orders

    enriched: List[Dict[str, Any]] = []
    for order in orders:
        item = dict(order)
        if item.get("status") != "submitted" or not item.get("order_id"):
            enriched.append(item)
            continue
        try:
            latest = api.get_order(item["order_id"])
            latest_status = str(getattr(latest, "status", "submitted"))
            item["status"] = latest_status
            item["alpaca_status"] = latest_status
            if latest_status == "rejected":
                reason = getattr(latest, "reject_reason", None) or "rejected"
                item["error"] = str(reason)
        except Exception as e:
            item["status"] = "submitted"
            item["error"] = f"status_lookup_failed: {e}"
        enriched.append(item)
    return enriched


def _build_run_payload(
    broker_mode: str,
    dry_run: bool,
    smoke_test: bool,
    strategy_name: str,
    universe_size: int,
    account_snapshot: Dict[str, float],
    positions_top10: List[Dict[str, Any]],
    target_trades: List[Dict[str, Any]],
    execution_results: List[Dict[str, Any]],
    target_diff: Dict[str, Any] | None,
    status: str,
    errors: List[str],
    stack_snippet: str | None = None,
) -> Dict[str, Any]:
    return {
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "broker_mode": broker_mode,
        "dry_run": dry_run,
        "smoke_test": smoke_test,
        "strategy_name": strategy_name,
        "universe_size": universe_size,
        "portfolio": account_snapshot,
        "positions_top10": positions_top10,
        "target_trades": target_trades,
        "execution_results": execution_results,
        "target_diff": target_diff or {},
        "status": status,
        "errors": errors,
        "stack_snippet": stack_snippet,
    }


def _compute_target_diff(previous_weights: Dict[str, Any], current_weights: Dict[str, Any]) -> Dict[str, Any]:
    prev = {str(k): float(v) for k, v in (previous_weights or {}).items() if v is not None}
    curr = {str(k): float(v) for k, v in (current_weights or {}).items() if v is not None}
    tol = 1e-6

    prev_syms = {s for s, w in prev.items() if abs(w) > tol}
    curr_syms = {s for s, w in curr.items() if abs(w) > tol}
    all_syms = sorted(prev_syms | curr_syms)

    added = sorted(curr_syms - prev_syms)
    removed = sorted(prev_syms - curr_syms)
    increased: List[str] = []
    decreased: List[str] = []
    changed: List[Dict[str, Any]] = []

    for sym in all_syms:
        pw = prev.get(sym, 0.0)
        cw = curr.get(sym, 0.0)
        delta = cw - pw
        if abs(delta) <= tol:
            continue
        if abs(cw) > abs(pw):
            increased.append(sym)
        else:
            decreased.append(sym)
        changed.append({
            "symbol": sym,
            "prev_weight": pw,
            "new_weight": cw,
            "delta_weight": delta,
            "abs_delta_weight": abs(delta),
        })

    changed = sorted(changed, key=lambda x: x["abs_delta_weight"], reverse=True)
    return {
        "added_symbols": added,
        "removed_symbols": removed,
        "increased_symbols": sorted(increased),
        "decreased_symbols": sorted(decreased),
        "changed_weights_top": changed[:10],
        "num_added": len(added),
        "num_removed": len(removed),
        "num_increased": len(increased),
        "num_decreased": len(decreased),
    }


def _write_run_log(kind: str, payload: Dict[str, Any], as_of_date: str | None = None) -> str:
    logs_dir = os.path.join("reports", "run_logs")
    os.makedirs(logs_dir, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if as_of_date:
        safe_as_of = re.sub(r"[^0-9A-Za-z_-]", "_", str(as_of_date))
        filename = f"{kind}_{safe_as_of}_{stamp}.json"
    else:
        filename = f"{kind}_{stamp}.json"
    path = os.path.join(logs_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    latest_path = os.path.join(logs_dir, f"latest_{kind}.json")
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    logger.info(f"Run log saved: {path}")
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--force', action='store_true', help='Force execution even if rebalance is not due (testing only)')
    parser.add_argument('--ignore-market-closed', action='store_true', help='Continue even if market is closed (testing only)')
    args = parser.parse_args()

    dry_run = bool(args.dry_run or _env_bool("DRY_RUN", False))
    smoke_test = _env_bool("SMOKE_TEST", False)
    open_orders_action = (os.getenv("OPEN_ORDERS_ACTION", "leave") or "leave").strip().lower()
    if open_orders_action not in {"leave", "cancel"}:
        raise RuntimeError(f"Invalid OPEN_ORDERS_ACTION={open_orders_action}. Expected 'leave' or 'cancel'.")

    config = load_config()
    state_manager = StateManager(state_file_path="./latest_state.json")
    state = state_manager.load_state()
    discord = DiscordProductionNotifier()

    as_of_date = None
    strategy_name = config.get("strategy_name", "lc_reversal")
    is_lc_reversal = strategy_name == "lc_reversal"

    logger.info("=" * 60)
    logger.info("DAILY LC-REVERSAL EXECUTION CHECK" if is_lc_reversal else "REBALANCE EXECUTION CHECK")
    logger.info(f"Strategy: {strategy_name}")
    logger.info(f"Mode: {'DRY RUN' if dry_run else 'LIVE'}")
    logger.info(f"Smoke test: {smoke_test}")
    logger.info(
        "Runtime flags raw env: "
        f"DRY_RUN={os.getenv('DRY_RUN')}, "
        f"SMOKE_TEST={os.getenv('SMOKE_TEST')}, "
        f"OPEN_ORDERS_ACTION={os.getenv('OPEN_ORDERS_ACTION')}"
    )
    logger.info("=" * 60)

    try:
        broker_mode = _validate_execution_env(config, dry_run=dry_run)
        api = tradeapi.REST(
            config['ALPACA_API_KEY'],
            config['ALPACA_SECRET_KEY'],
            config['ALPACA_BASE_URL']
        )

        calendar = TradingCalendar(api)
        as_of_date = calendar.get_last_completed_trading_day()
        as_of_date_str = as_of_date.isoformat()
        logger.info(f"As-of date: {as_of_date_str}")

        execution_state = state.setdefault("execution", {})
        last_successful_trade_date = execution_state.get("last_successful_trade_date")
        last_run = execution_state.get("last_run") or {}

        if (not args.force) and last_successful_trade_date == as_of_date_str:
            logger.warning("Already executed successfully for this trade date. Exiting for idempotency.")
            account_snapshot = _get_account_snapshot(api)
            positions_top10 = _get_positions_top10(api)
            discord.send_rebalance_markdown_summary(
                _build_run_payload(
                    broker_mode=broker_mode,
                    dry_run=dry_run,
                    smoke_test=smoke_test,
                    strategy_name=strategy_name,
                    universe_size=0,
                    account_snapshot=account_snapshot,
                    positions_top10=positions_top10,
                    target_trades=[],
                    execution_results=[],
                    target_diff=None,
                    status="skipped_already_successful",
                    errors=[f"Already executed successfully on {as_of_date_str}"]
                )
            )
            return

        if (not args.force) and (not state_manager.check_rebalance_due(threshold=20)) and not smoke_test and not is_lc_reversal:
            days_until = state.get('rebalance', {}).get('days_until_rebalance', 20)
            logger.info(f"Rebalance NOT due ({days_until} days remaining)")
            account_snapshot = _get_account_snapshot(api)
            positions_top10 = _get_positions_top10(api)
            discord.send_rebalance_markdown_summary(
                _build_run_payload(
                    broker_mode=broker_mode,
                    dry_run=dry_run,
                    smoke_test=smoke_test,
                    strategy_name=strategy_name,
                    universe_size=0,
                    account_snapshot=account_snapshot,
                    positions_top10=positions_top10,
                    target_trades=[],
                    execution_results=[],
                    target_diff=None,
                    status="skipped_not_due",
                    errors=[f"Rebalance not due. days_until_rebalance={days_until}"]
                )
            )
            return
        # Existing date guard from rebalance schedule, unless previous run was partial failure.
        if (not is_lc_reversal) and state_manager.check_already_rebalanced(as_of_date):
            if last_run.get("trade_date") == as_of_date_str and last_run.get("status") == "partial_failure":
                logger.warning("Previous run on this date had partial_failure; continuing with reconciliation.")
            else:
                logger.warning("Already rebalanced today - exiting")
                account_snapshot = _get_account_snapshot(api)
                positions_top10 = _get_positions_top10(api)
                discord.send_rebalance_markdown_summary(
                    _build_run_payload(
                        broker_mode=broker_mode,
                        dry_run=dry_run,
                        smoke_test=smoke_test,
                        strategy_name=strategy_name,
                        universe_size=0,
                        account_snapshot=account_snapshot,
                        positions_top10=positions_top10,
                        target_trades=[],
                        execution_results=[],
                        target_diff=None,
                        status="skipped_already_rebalanced",
                        errors=[f"State indicates rebalance already executed on {as_of_date_str}"]
                    )
                )
                return

        if not check_market_open(api):
            if args.ignore_market_closed:
                logger.warning("Market is CLOSED, but --ignore-market-closed was set (testing) - continuing")
            else:
                raise RuntimeError("Market is closed; aborting rebalance execution.")

        kill_switch_enabled, kill_reason = check_kill_switch()
        if kill_switch_enabled:
            raise RuntimeError(f"Kill switch enabled: {kill_reason}")

        account_snapshot = _get_account_snapshot(api)
        portfolio_value = float(account_snapshot.get("equity", 0.0) or 0.0)
        if portfolio_value <= 0:
            raise RuntimeError(f"Invalid account equity for execution: {portfolio_value}")

        current_positions = get_current_positions(api)
        positions_top10 = _get_positions_top10(api)

        logger.info(f"Portfolio equity: ${portfolio_value:,.2f}")
        logger.info(f"Current positions: {len(current_positions)}")

        open_orders = get_open_orders(api)
        open_order_results: List[Dict[str, Any]] = []
        open_orders_by_symbol: Dict[str, List[Dict[str, Any]]] = {}
        if open_orders:
            logger.warning(f"Found {len(open_orders)} open order(s) before rebalance.")
            if open_orders_action == "cancel" and not dry_run:
                open_order_results = cancel_open_orders(api, open_orders)
                failed_cancels = [o for o in open_order_results if o.get("status") != "cancelled"]
                if failed_cancels:
                    logger.error(f"Failed to cancel {len(failed_cancels)} open order(s); these symbols may be skipped.")
                    for o in failed_cancels:
                        sym = o.get("symbol")
                        open_orders_by_symbol.setdefault(sym, []).append(o)
            else:
                for o in open_orders:
                    sym = o.get("symbol")
                    open_orders_by_symbol.setdefault(sym, []).append(o)

        target_weights: Dict[str, float] = {}
        metadata: Dict[str, Any] = {}
        btc_allocation = 0.0
        crypto_target_allocations: Dict[str, float] = {}
        crypto_latest_prices: Dict[str, float] = {}
        universe_size = 0

        if smoke_test:
            smoke_symbol = (os.getenv("SMOKE_TEST_SYMBOL", "SPY") or "SPY").strip().upper()
            smoke_notional = float(os.getenv("SMOKE_TEST_NOTIONAL", "50") or 50)
            if smoke_notional <= 0:
                raise RuntimeError(f"SMOKE_TEST_NOTIONAL must be > 0, got {smoke_notional}")
            logger.warning(f"SMOKE_TEST enabled: placing single tiny trade in {smoke_symbol} (${smoke_notional:.2f}).")

            quote = api.get_latest_trade(smoke_symbol, feed='iex')
            smoke_price = float(quote.price)
            smoke_qty = round(smoke_notional / smoke_price, 6)
            smoke_order = {
                "symbol": smoke_symbol,
                "side": "buy",
                "qty": smoke_qty,
                "notional": smoke_notional,
                "price": smoke_price,
            }

            if smoke_symbol in open_orders_by_symbol:
                smoke_order["status"] = "skipped_existing_open_order"
                smoke_order["error"] = "Open order already exists for smoke symbol"
                orders = [smoke_order]
            elif dry_run:
                smoke_order["status"] = "dry_run"
                orders = [smoke_order]
            else:
                client_order_id = f"smoke-{as_of_date_str.replace('-', '')}-{smoke_symbol}"[:48]
                try:
                    submitted = api.submit_order(
                        symbol=smoke_symbol,
                        notional=smoke_notional,
                        side="buy",
                        type="market",
                        time_in_force="day",
                        client_order_id=client_order_id,
                    )
                    smoke_order["status"] = "submitted"
                    smoke_order["order_id"] = getattr(submitted, "id", None)
                    smoke_order["alpaca_status"] = getattr(submitted, "status", "unknown")
                except Exception as e:
                    smoke_order["status"] = "failed"
                    smoke_order["error"] = str(e)
                orders = [smoke_order]

            strategy_name = "smoke_test"
            target_trades = [
                {
                    "symbol": smoke_order["symbol"],
                    "side": smoke_order.get("side"),
                    "qty": smoke_order.get("qty"),
                    "notional": smoke_order.get("notional"),
                    "limit_price": None,
                }
            ]

        else:
            universe_builder = Universe(api, config)
            symbols, _ = universe_builder.build_universe()
            universe_size = len(symbols)

            end_date = as_of_date
            start_date = end_date - timedelta(days=365)

            data_service = DataService(config)
            crypto_tickers: List[str] = config.get('crypto_tickers', ['BTC/USD']) if config.get('trade_crypto', False) else []
            tickers_for_fetch = list(dict.fromkeys(symbols + crypto_tickers))
            df = data_service.get_historical_data(
                tickers_for_fetch,
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d'),
                ensure_spy=True
            )
            if df.empty:
                raise RuntimeError("No historical data available for portfolio construction.")

            try:
                spy_data = df.xs('SPY', level=1)
            except KeyError:
                raise RuntimeError("SPY not found in historical data (required for regime filter).")

            current_weights = {}
            trade_crypto_enabled = bool(config.get('trade_crypto', False))
            crypto_alpaca_symbols = {t.replace('/', '') for t in crypto_tickers}
            if not trade_crypto_enabled:
                for sym in current_positions.keys():
                    if '/' in sym or (sym.endswith('USD') and sym.isalpha() and len(sym) > 3):
                        crypto_alpaca_symbols.add(sym)
            for symbol, qty in current_positions.items():
                if symbol in crypto_alpaca_symbols:
                    continue
                try:
                    quote = api.get_latest_trade(symbol, feed='iex')
                    price = float(quote.price)
                    dollar_value = qty * price
                    current_weights[symbol] = dollar_value / portfolio_value if portfolio_value > 0 else 0.0
                except Exception:
                    continue

            if is_lc_reversal:
                strategy = LCReversalStrategy(config)
                df_features = strategy.compute_features(df)
                lc_state = state.get("lc_reversal", {}) or {}
                target_weights, metadata, lc_next_state = strategy.build_targets(
                    df_features=df_features,
                    as_of_date=as_of_date,
                    portfolio_value=portfolio_value,
                    current_positions={s: q for s, q in current_positions.items() if s not in crypto_alpaca_symbols},
                    current_weights=current_weights,
                    lc_state=lc_state,
                )
                state["lc_reversal"] = lc_next_state
                logger.info(
                    f"LC-Reversal target weights: {len(target_weights)} "
                    f"(universe={metadata.get('universe_size')}, "
                    f"long={metadata.get('long_selected')}, short={metadata.get('short_selected')})"
                )
            else:
                storage = FileStorage()
                signals = storage.get_latest_signals()
                if signals is None or signals.empty:
                    raise RuntimeError("No signals available for portfolio construction.")

                from src.strategy_simple import SimpleStrategy
                strategy = SimpleStrategy(config, horizon='20d')
                df_features = strategy.compute_features(df)
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

            if config.get('trade_crypto', False) and crypto_tickers:
                from src.crypto_strategy import BitcoinMomentumStrategy, CryptoStrategy
                crypto_strategy_type = (config.get('crypto_allocation_strategy') or 'momentum').lower()

                if crypto_strategy_type == 'fixed':
                    fixed = CryptoStrategy(config)
                    btc_allocation = float(config.get('crypto_allocation', 0.10))
                    crypto_target_allocations = fixed.get_target_allocation(portfolio_value)
                else:
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

            equity_scale = max(0.0, 1.0 - btc_allocation)
            if target_weights and equity_scale < 1.0:
                target_weights = {s: w * equity_scale for s, w in target_weights.items()}

            exec_config = {
                'max_orders_per_rebalance': 100,
                'max_daily_notional': 1_000_000,
                'min_trade_notional': 10,
                'turnover_buffer_pct': 1.0,
                'order_type': config.get('order_type', 'market'),
                'limit_offset_bps': config.get('limit_offset_bps', 10),
            }

            crypto_alpaca_symbols = {t.replace('/', '') for t in (config.get('crypto_tickers', ['BTC/USD']) if config.get('trade_crypto', False) else [])}
            if not config.get('trade_crypto', False):
                for sym in current_positions.keys():
                    if '/' in sym or (sym.endswith('USD') and sym.isalpha() and len(sym) > 3):
                        crypto_alpaca_symbols.add(sym)
            order_prefix = f"rebal-{as_of_date_str.replace('-', '')}"

            orders = execute_orders_safe(
                api,
                target_weights,
                {s: q for s, q in current_positions.items() if s not in crypto_alpaca_symbols},
                portfolio_value,
                exec_config,
                dry_run=dry_run,
                open_orders_by_symbol=open_orders_by_symbol,
                client_order_prefix=order_prefix,
            )

            crypto_orders = []
            if crypto_target_allocations:
                from src.crypto_strategy import get_crypto_orders
                crypto_orders = get_crypto_orders(
                    api,
                    crypto_target_allocations,
                    latest_prices=crypto_latest_prices,
                    dry_run=dry_run
                )

            orders = (orders or []) + (crypto_orders or [])
            target_trades = [
                {
                    "symbol": o.get("symbol"),
                    "side": o.get("side"),
                    "qty": o.get("qty"),
                    "notional": o.get("notional"),
                    "limit_price": o.get("limit_price"),
                }
                for o in orders
                if o.get("symbol")
            ]

            # Keep existing promotion + shadow tracking behavior for ML momentum path.
            if not is_lc_reversal:
                model_promoted = None
                candidate = state.get('models', {}).get('candidate_model', {})
                if candidate.get('approved_for_next_rebalance') and not dry_run:
                    logger.info(f"Promoting candidate: {candidate.get('version')}")
                    state_manager.promote_candidate_to_active()
                    strategy_type = candidate.get('strategy_type') or state.get('strategies', {}).get('best') or 'simple'
                    model_version = candidate.get('version', 'unknown')
                    model_promoted = f"{model_version} ({strategy_type})"
                    logger.info(f"Model promoted: {model_promoted}")

                shadow_tracker = ShadowPortfolioTracker(config)
                shadow_state = shadow_tracker.initialize_from_state(state)

                from src.strategy_simple import PureMomentumStrategy
                canary_strategy = PureMomentumStrategy(config)
                canary_signals = canary_strategy.compute_signals(df_features)

                shadow_state = shadow_tracker.rebalance_shadow(
                    shadow_state,
                    signals,
                    canary_signals,
                    df_features,
                    as_of_date,
                    spy_data=spy_data
                )

                state['shadow_portfolios'] = shadow_state

        orders = _enrich_order_statuses(api, orders, dry_run=dry_run)
        orders = (open_order_results or []) + (orders or [])

        submitted_like_statuses = {
            "submitted",
            "new",
            "accepted",
            "partially_filled",
            "filled",
            "dry_run",
            "cancelled",
            "skipped_existing_open_order",
            "skipped_infeasible_short_qty",
        }
        failures = [o for o in orders if str(o.get("status", "")).lower() not in submitted_like_statuses]
        run_status = "success" if not failures else "partial_failure"

        execution_state = state.setdefault("execution", {})
        previous_target_weights = (
            (execution_state.get("last_run") or {})
            .get("target_portfolio_snapshot", {})
            .get("target_weights", {})
        )
        target_diff = _compute_target_diff(previous_target_weights, target_weights)

        execution_state["last_run"] = {
            "trade_date": as_of_date_str,
            "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "status": run_status,
            "dry_run": dry_run,
            "smoke_test": smoke_test,
            "open_orders_action": open_orders_action,
            "target_portfolio_snapshot": {
                "target_weights": target_weights,
                "btc_allocation": btc_allocation,
            },
            "orders_attempted": target_trades,
            "order_results": orders,
        }
        if run_status == "success" and not dry_run:
            execution_state["last_successful_trade_date"] = as_of_date_str
            if not is_lc_reversal:
                rebalance = state.setdefault("rebalance", {})
                rebalance["last_rebalance_date"] = as_of_date_str
                rebalance["days_since_rebalance"] = 0
                next_rebalance = as_of_date + timedelta(days=20)
                rebalance["next_rebalance_date"] = next_rebalance.isoformat()
                rebalance["days_until_rebalance"] = 20

        state_manager.save_state(state)

        run_payload = _build_run_payload(
            broker_mode=broker_mode,
            dry_run=dry_run,
            smoke_test=smoke_test,
            strategy_name=strategy_name,
            universe_size=universe_size,
            account_snapshot=account_snapshot,
            positions_top10=positions_top10,
            target_trades=target_trades,
            execution_results=orders,
            target_diff=target_diff,
            status=run_status,
            errors=[o.get("error") for o in failures if o.get("error")],
        )
        _write_run_log("rebalance", run_payload, as_of_date=as_of_date_str)
        discord.send_rebalance_markdown_summary(run_payload)

        if failures:
            raise RuntimeError(f"Order execution completed with {len(failures)} failure(s).")

        logger.info("=" * 60)
        logger.info("REBALANCE COMPLETE")
        logger.info("=" * 60)

    except Exception as e:
        logger.exception(f"Rebalance execution failed: {e}")
        err_date = as_of_date.isoformat() if as_of_date else datetime.now(timezone.utc).date().isoformat()
        stack = _stack_snippet()

        try:
            failure_payload = _build_run_payload(
                broker_mode=(os.getenv("BROKER_MODE", "paper") or "paper").lower(),
                dry_run=dry_run,
                smoke_test=smoke_test,
                strategy_name=strategy_name,
                universe_size=0,
                account_snapshot={"cash": 0.0, "equity": 0.0},
                positions_top10=[],
                target_trades=[],
                execution_results=[],
                target_diff=None,
                status="failed",
                errors=[f"{type(e).__name__}: {e}", f"trade_date={err_date}"],
                stack_snippet=stack,
            )
            _write_run_log("rebalance", failure_payload, as_of_date=err_date)
            discord.send_rebalance_markdown_summary(failure_payload)
        except Exception as notify_err:
            logger.error(f"Failed to send failure webhook: {notify_err}")

        execution_state = state.setdefault("execution", {})
        execution_state["last_run"] = {
            "trade_date": err_date,
            "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "status": "failed",
            "dry_run": dry_run,
            "smoke_test": smoke_test,
            "error": str(e),
            "stack_snippet": stack,
        }
        state_manager.save_state(state)
        raise


if __name__ == "__main__":
    main()
