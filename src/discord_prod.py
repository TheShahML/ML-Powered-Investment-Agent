"""Production Discord notifications - clean, operational messages."""
import requests
import os
from datetime import datetime
from typing import Dict, List, Optional
from loguru import logger


class DiscordProductionNotifier:
    """Clean operational Discord notifications."""

    def __init__(self, webhook_url: str = None):
        self.webhook_url = webhook_url or os.environ.get('DISCORD_WEBHOOK_URL')

    def _send(self, embed: Dict) -> bool:
        """Send embed to Discord."""
        if not self.webhook_url:
            logger.warning("Discord webhook not configured")
            return False

        try:
            response = requests.post(
                self.webhook_url,
                json={"embeds": [embed]},
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            logger.info("Discord notification sent")
            return True
        except Exception as e:
            logger.error(f"Discord send failed: {e}")
            return False

    def send_image(self, text: str, image_path: str) -> bool:
        """
        Send image to Discord via webhook.
        
        Args:
            text: Text content (can be empty string)
            image_path: Path to image file
        
        Returns:
            Success status
        """
        if not self.webhook_url:
            logger.warning("Discord webhook not configured")
            return False

        if not os.path.exists(image_path):
            logger.error(f"Image file not found: {image_path}")
            return False

        try:
            with open(image_path, 'rb') as f:
                files = {
                    'file': (os.path.basename(image_path), f, 'image/png')
                }
                data = {
                    'content': text
                }
                response = requests.post(
                    self.webhook_url,
                    files=files,
                    data=data
                )
                response.raise_for_status()
                logger.info(f"Discord image sent: {image_path}")
                return True
        except Exception as e:
            logger.error(f"Discord image send failed: {e}")
            return False

    def send_multiple_images(self, text: str, image_paths: List[str]) -> bool:
        """
        Send multiple images to Discord (as separate messages).
        
        Args:
            text: Text content for first message
            image_paths: List of image file paths
        
        Returns:
            Success status
        """
        if not image_paths:
            return False

        # Send first image with text
        success = self.send_image(text, image_paths[0])
        
        # Send remaining images without text
        for img_path in image_paths[1:]:
            self.send_image("", img_path)
        
        return success

    def send_multi_horizon_signals(
        self,
        as_of_date: str,
        active_model: Dict,
        candidate_approved: bool,
        universe_size: int,
        ml_top10_1d: List[tuple],
        ml_top10_5d: List[tuple],
        ml_top10_20d: List[tuple],
        canary_top10: List[tuple],
        performance_since_rebal: Dict,
        performance_rolling: Dict,
        data_fresh: bool,
        days_until_rebal: int,
        next_rebal_date: str
    ):
        """Multi-horizon daily signals summary."""
        # ML top 5 for each horizon (to fit in Discord)
        ml_list_1d = "\n".join([f"{i}. **{sym}** ({score:+.4f})" for i, (sym, score) in enumerate(ml_top10_1d[:5], 1)])
        ml_list_5d = "\n".join([f"{i}. **{sym}** ({score:+.4f})" for i, (sym, score) in enumerate(ml_top10_5d[:5], 1)])
        ml_list_20d = "\n".join([f"{i}. **{sym}** ({score:+.4f})" for i, (sym, score) in enumerate(ml_top10_20d[:5], 1)])

        # Benchmark momentum (canary) top 5
        canary_list = "\n".join([f"{i}. **{sym}** ({score:+.1f}%)" for i, (sym, score) in enumerate(canary_top10[:5], 1)])

        # Performance since last rebal
        perf_since = (
            f"**Actual:** {performance_since_rebal.get('actual', 0):+.1f}%\n"
            f"**Benchmark Momentum (Canary):** {performance_since_rebal.get('canary', 0):+.1f}%\n"
            f"**SPY:** {performance_since_rebal.get('spy', 0):+.1f}%"
        )

        # Rolling 30d
        perf_30d = (
            f"Actual: {performance_rolling.get('actual_30d', 0):+.1f}% | "
            f"Benchmark Momentum: {performance_rolling.get('canary_30d', 0):+.1f}% | "
            f"SPY: {performance_rolling.get('spy_30d', 0):+.1f}%"
        )

        # Determine strategy type from active model
        strategy_type = active_model.get('strategy_type', 'simple')
        strategy_names = {
            'simple': 'XGBoost Multi-Horizon',
            'lstm': 'LSTM Neural Network',
            'pure_momentum': 'Pure Momentum'
        }
        strategy_name = strategy_names.get(strategy_type, strategy_type)
        
        embed = {
            "title": f"📊 MULTI-HORIZON SIGNALS | {as_of_date}",
            "description": f"**Strategy:** {strategy_name} (1d/5d/20d horizons)",
            "color": 3447003,
            "fields": [
                {
                    "name": "📅 Info",
                    "value": (
                        f"**As-of:** {as_of_date}\n"
                        f"**Active Model:** {active_model.get('version', 'None')}\n"
                        f"**Strategy:** {strategy_name}\n"
                        f"**Candidate:** {'✅ Approved' if candidate_approved else '❌ Pending'}\n"
                        f"**Universe:** {universe_size} stocks"
                    ),
                    "inline": False
                },
                {
                    "name": "🔵 1-Day Horizon Top 5",
                    "value": ml_list_1d or "No signals",
                    "inline": True
                },
                {
                    "name": "🟢 5-Day Horizon Top 5",
                    "value": ml_list_5d or "No signals",
                    "inline": True
                },
                {
                    "name": "🎯 20-Day Horizon Top 5 (PRIMARY)",
                    "value": ml_list_20d or "No signals",
                    "inline": True
                },
                {
                    "name": "📈 Benchmark Momentum (Canary) Top 5",
                    "value": canary_list or "No signals",
                    "inline": True
                },
                {
                    "name": f"📊 Since Last Rebalance ({performance_since_rebal.get('days', 0)}d)",
                    "value": perf_since,
                    "inline": False
                },
                {
                    "name": "📆 Rolling 30d",
                    "value": perf_30d,
                    "inline": False
                },
                {
                    "name": "⏭️ Next Rebalance",
                    "value": f"**{next_rebal_date}** ({days_until_rebal} trading days)",
                    "inline": True
                },
                {
                    "name": "💾 Data Freshness",
                    "value": "✅ Fresh" if data_fresh else "❌ STALE",
                    "inline": True
                }
            ],
            "footer": {"text": "Investment Bot • Multi-Horizon Signals"},
            "timestamp": datetime.utcnow().isoformat()
        }

        return self._send(embed)

    def send_daily_signals_summary(
        self,
        as_of_date: str,
        active_model: Dict,
        candidate_approved: bool,
        universe_size: int,
        ml_top10: List[tuple],
        canary_top10: List[tuple],
        performance_since_rebal: Dict,
        performance_rolling: Dict,
        data_fresh: bool,
        days_until_rebal: int,
        next_rebal_date: str
    ):
        """Daily signals summary (single horizon - legacy)."""
        # ML top 10
        ml_list = "\n".join([f"{i}. **{sym}** ({score:+.4f})" for i, (sym, score) in enumerate(ml_top10[:10], 1)])

        # Benchmark momentum (canary) top 10
        canary_list = "\n".join([f"{i}. **{sym}** ({score:+.1f}%)" for i, (sym, score) in enumerate(canary_top10[:10], 1)])

        # Performance since last rebal
        perf_since = (
            f"**Actual:** {performance_since_rebal.get('actual', 0):+.1f}%\n"
            f"**Benchmark Momentum (Canary):** {performance_since_rebal.get('canary', 0):+.1f}%\n"
            f"**SPY:** {performance_since_rebal.get('spy', 0):+.1f}%\n"
            f"**BTC:** {performance_since_rebal.get('btc', 0):+.1f}%"
        )

        # Rolling 30d
        perf_30d = (
            f"Actual: {performance_rolling.get('actual_30d', 0):+.1f}% | "
            f"Benchmark Momentum: {performance_rolling.get('canary_30d', 0):+.1f}% | "
            f"SPY: {performance_rolling.get('spy_30d', 0):+.1f}%"
        )

        embed = {
            "title": f"📊 DAILY SIGNALS | {as_of_date}",
            "color": 3447003,
            "fields": [
                {
                    "name": "📅 Info",
                    "value": (
                        f"**As-of:** {as_of_date}\n"
                        f"**Active Model:** {active_model.get('version', 'None')}\n"
                        f"**Strategy:** {active_model.get('strategy_type', 'simple')}\n"
                        f"**Candidate:** {'✅ Approved' if candidate_approved else '❌ Pending'}\n"
                        f"**Universe:** {universe_size} stocks"
                    ),
                    "inline": False
                },
                {
                    "name": "🤖 ML Top 10",
                    "value": ml_list or "No signals",
                    "inline": True
                },
                {
                    "name": "📈 Benchmark Momentum (Canary) Top 10",
                    "value": canary_list or "No signals",
                    "inline": True
                },
                {
                    "name": f"📊 Since Last Rebalance ({performance_since_rebal.get('days', 0)}d)",
                    "value": perf_since,
                    "inline": False
                },
                {
                    "name": "📆 Rolling 30d",
                    "value": perf_30d,
                    "inline": False
                },
                {
                    "name": "⏭️ Next Rebalance",
                    "value": f"**{next_rebal_date}** ({days_until_rebal} trading days)",
                    "inline": True
                },
                {
                    "name": "💾 Data Freshness",
                    "value": "✅ Fresh" if data_fresh else "❌ STALE",
                    "inline": True
                }
            ],
            "footer": {"text": "Investment Bot • Daily Signals"},
            "timestamp": datetime.utcnow().isoformat()
        }

        return self._send(embed)

    def send_weekly_training_report(
        self,
        candidate_version: str,
        training_window: tuple,
        cv_metrics: Dict,
        backtest_candidate: Dict,
        backtest_baselines: Dict,
        gate_passed: bool,
        gate_details: Dict,
        strategy_type: str = "simple",
        multi_strategy_results: Dict = None
    ):
        """
        Weekly training completion report (multi-horizon, multi-strategy).
        
        Args:
            candidate_version: Model version string
            training_window: (start_date, end_date) tuple
            cv_metrics: Cross-validation metrics (can be multi-horizon dict or single)
            backtest_candidate: Backtest results for candidate
            backtest_baselines: Dict of baseline results
            gate_passed: Whether promotion gate passed
            gate_details: Gate check details
            strategy_type: Strategy type ('simple', 'lstm', etc.)
            multi_strategy_results: Optional dict mapping strategy_type -> results for comparison
        """
        # Strategy name mapping
        strategy_names = {
            'simple': 'XGBoost Multi-Horizon',
            'lstm': 'LSTM Neural Network',
            'pure_momentum': 'Pure Momentum'
        }
        strategy_name = strategy_names.get(strategy_type, strategy_type.upper())
        
        # CV metrics - check if multi-horizon or single
        if '1d' in cv_metrics and '5d' in cv_metrics and '20d' in cv_metrics:
            # Multi-horizon format
            cv_text = (
                f"**1d Model IC:** {cv_metrics['1d'].get('cv_mean_ic', 0):.4f} ± {cv_metrics['1d'].get('cv_std_ic', 0):.4f}\n"
                f"**5d Model IC:** {cv_metrics['5d'].get('cv_mean_ic', 0):.4f} ± {cv_metrics['5d'].get('cv_std_ic', 0):.4f}\n"
                f"**20d Model IC:** {cv_metrics['20d'].get('cv_mean_ic', 0):.4f} ± {cv_metrics['20d'].get('cv_std_ic', 0):.4f}\n"
                f"*(20d model used for rebalancing)*"
            )
        else:
            # Single horizon format (legacy)
            cv_text = (
                f"**Mean IC:** {cv_metrics.get('cv_mean_ic', 0):.4f} ± {cv_metrics.get('cv_std_ic', 0):.4f}\n"
                f"**Range:** [{cv_metrics.get('cv_min_ic', 0):.4f}, {cv_metrics.get('cv_max_ic', 0):.4f}]\n"
                f"**% Positive Days:** {cv_metrics.get('cv_mean_pct_positive', 0):.1f}%"
            )

        # Backtest (richer)
        bt_text = (
            f"**CAGR:** {backtest_candidate.get('cagr', 0)*100:.1f}%\n"
            f"**Sharpe:** {backtest_candidate.get('sharpe', 0):.2f}\n"
            f"**MaxDD:** {backtest_candidate.get('max_drawdown', 0)*100:.1f}%\n"
            f"**Total Return:** {backtest_candidate.get('total_return', 0)*100:.1f}%\n"
            f"**Worst Month:** {backtest_candidate.get('worst_month', 0)*100:.1f}%\n"
            f"**Trades:** {backtest_candidate.get('total_trades', 0)}\n"
            f"**Turnover:** {backtest_candidate.get('avg_turnover', 0)*100:.1f}%\n"
            f"**Cost Drag:** {backtest_candidate.get('cost_drag_cumulative', 0)*100:.2f}%"
        )

        # vs Baselines (include MaxDD)
        baseline_text = ""
        for name, metrics in backtest_baselines.items():
            baseline_text += (
                f"**{name.upper()}:** "
                f"{metrics.get('cagr', 0)*100:.1f}% CAGR, "
                f"{metrics.get('sharpe', 0):.2f} Sharpe, "
                f"{metrics.get('max_drawdown', 0)*100:.1f}% MaxDD\n"
            )

        # Gate
        gate_emoji = "✅ PASSED" if gate_passed else "❌ FAILED"
        gate_text = (
            f"**Status:** {gate_emoji}\n"
            f"Sharpe margin: {gate_details.get('sharpe_margin_achieved', 0):+.2f} "
            f"(req: {gate_details.get('sharpe_margin_required', 0):+.2f})\n"
            f"MaxDD diff: {gate_details.get('maxdd_diff', 0):+.2f} "
            f"(tol: {gate_details.get('maxdd_tolerance', 0):+.2f})"
        )

        fields = [
            {
                "name": "🆕 Candidate Model",
                "value": (
                    f"**Version:** {candidate_version}\n"
                    f"**Strategy:** {strategy_name}\n"
                    f"**Window:** {training_window[0]} to {training_window[1]}"
                ),
                "inline": False
            },
            {
                "name": "📊 Cross-Validation (Leakage-Safe)",
                "value": cv_text,
                "inline": False
            },
            {
                "name": "📈 Walk-Forward Backtest",
                "value": bt_text,
                "inline": True
            },
            {
                "name": "📊 vs Baselines",
                "value": baseline_text or "No baselines",
                "inline": True
            },
            {
                "name": "🚪 Promotion Gate",
                "value": gate_text,
                "inline": False
            }
        ]
        
        # Add multi-strategy comparison if available
        if multi_strategy_results:
            comparison_text = ""
            for strat_type, results in multi_strategy_results.items():
                strat_name = strategy_names.get(strat_type, strat_type.upper())
                sharpe = results.get('sharpe', 0)
                cagr = results.get('cagr', 0) * 100
                maxdd = results.get('max_drawdown', 0) * 100
                comparison_text += (
                    f"**{strat_name}:** "
                    f"{cagr:.1f}% CAGR, "
                    f"{sharpe:.2f} Sharpe, "
                    f"{maxdd:.1f}% MaxDD\n"
                )
            
            fields.append({
                "name": "🔄 Strategy Comparison",
                "value": comparison_text or "No comparison data",
                "inline": False
            })

        embed = {
            "title": "🧠 WEEKLY TRAINING COMPLETE",
            "description": f"**Strategy:** {strategy_name}",
            "color": 5763719 if gate_passed else 15158332,
            "fields": fields,
            "footer": {"text": "Investment Bot • Weekly Training"},
            "timestamp": datetime.utcnow().isoformat()
        }

        return self._send(embed)
    
    def send_multi_strategy_comparison(
        self,
        as_of_date: str,
        strategy_results: Dict[str, Dict],
        baselines: Dict[str, Dict],
        best_strategy: str,
        training_window: tuple,
        gate_passed: bool = False,
        gate_details: Optional[Dict] = None
    ):
        """
        Send multi-strategy comparison report.
        
        Args:
            as_of_date: Training date
            strategy_results: Dict mapping strategy_type -> {sharpe, cagr, max_drawdown, ...}
            baselines: Dict mapping baseline_name -> {sharpe, cagr, max_drawdown, ...}
            best_strategy: Name of best performing strategy
            training_window: (start_date, end_date) tuple
        """
        strategy_names = {
            'simple': '🤖 XGBoost',
            'pure_momentum': '📈 Pure Momentum'
        }
        
        # Build strategy comparison (include richer metrics)
        strategy_comparison = ""
        for strat_type, results in strategy_results.items():
            strat_name = strategy_names.get(strat_type, strat_type.upper())
            sharpe = results.get('sharpe', 0)
            cagr = results.get('cagr', 0) * 100
            maxdd = results.get('max_drawdown', 0) * 100
            total_ret = results.get('total_return', 0) * 100
            worst_month = results.get('worst_month', 0) * 100
            trades = results.get('total_trades', 0)
            cost_drag = results.get('cost_drag_cumulative', 0) * 100
            is_best = "**(best candidate)**" if strat_type == best_strategy else ""
            
            strategy_comparison += (
                f"{is_best} **{strat_name}**\n"
                f"  CAGR: {cagr:.1f}% | Sharpe: {sharpe:.2f} | MaxDD: {maxdd:.1f}%\n"
                f"  TotalRet: {total_ret:.1f}% | WorstMo: {worst_month:.1f}% | Trades: {trades} | Cost: {cost_drag:.2f}%\n\n"
            )
        
        # Baselines (include richer metrics)
        baseline_text = ""
        for name, metrics in baselines.items():
            baseline_text += (
                f"**{name.upper()}:** "
                f"{metrics.get('cagr', 0)*100:.1f}% CAGR, "
                f"{metrics.get('sharpe', 0):.2f} Sharpe, "
                f"{metrics.get('max_drawdown', 0)*100:.1f}% MaxDD, "
                f"{metrics.get('worst_month', 0)*100:.1f}% WorstMo\n"
            )
        
        best_strategy_name = strategy_names.get(best_strategy, best_strategy.upper())

        # Promotion gate summary
        gate_details = gate_details or {}
        gate_status = "✅ PASSED" if gate_passed else "❌ FAILED"
        gate_text = (
            f"**Status:** {gate_status}\n"
            f"Sharpe margin: {gate_details.get('sharpe_margin_achieved', 0):+.2f} (req {gate_details.get('sharpe_margin_required', 0):+.2f})\n"
            f"MaxDD diff: {gate_details.get('maxdd_diff', 0):+.2f} (tol {gate_details.get('maxdd_tolerance', 0):+.2f})"
        )
        
        embed = {
            "title": "🔄 MULTI-STRATEGY COMPARISON",
            "description": f"**Selected Strategy (among candidates):** {best_strategy_name}",
            "color": 5763719 if gate_passed else 15158332,
            "fields": [
                {
                    "name": "📅 Training Info",
                    "value": (
                        f"**Date:** {as_of_date}\n"
                        f"**Window:** {training_window[0]} to {training_window[1]}"
                    ),
                    "inline": False
                },
                {
                    "name": "🤖 Strategy Performance",
                    "value": strategy_comparison or "No results",
                    "inline": False
                },
                {
                    "name": "📊 Benchmarks (Buy & Hold)",
                    "value": baseline_text or "No benchmarks",
                    "inline": False
                },
                {
                    "name": "🚪 Promotion Gate",
                    "value": gate_text,
                    "inline": False
                },
                {
                    "name": "✅ Next Rebalance Action",
                    "value": (
                        f"**PROMOTE candidate** → use **{best_strategy_name}**"
                        if gate_passed else
                        "**DO NOT PROMOTE** → keep current active model (benchmarks outperformed)"
                    ),
                    "inline": False
                }
            ],
            "footer": {"text": "Investment Bot • Multi-Strategy Training"},
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return self._send(embed)

    def send_monthly_rebalance_execution(
        self,
        broker_mode: str,
        as_of_date: str,
        model_promoted: Optional[str],
        strategy_type: Optional[str],
        equity_allocation: float,
        btc_allocation: float,
        portfolio_value: float,
        orders: List[Dict],
        kill_switch: bool,
        market_closed: bool,
        dry_run: bool
    ):
        """Monthly rebalance execution report."""
        # Status
        if kill_switch:
            title = "🛑 REBALANCE BLOCKED - KILL SWITCH"
            color = 15158332
        elif market_closed:
            title = "⏸️ REBALANCE SKIPPED - MARKET CLOSED"
            color = 16776960
        elif dry_run:
            title = "🧪 REBALANCE DRY RUN"
            color = 16776960
        else:
            title = "✅ MONTHLY REBALANCE EXECUTED"
            color = 5763719

        # Separate equity and crypto orders
        equity_orders = [o for o in orders if '/' not in o.get('symbol', '')]
        crypto_orders = [o for o in orders if '/' in o.get('symbol', '')]

        equity_buys = [o for o in equity_orders if o.get('side') == 'buy']
        equity_sells = [o for o in equity_orders if o.get('side') == 'sell']
        crypto_buys = [o for o in crypto_orders if o.get('side') == 'buy']
        crypto_sells = [o for o in crypto_orders if o.get('side') == 'sell']

        # Top buys/sells
        top_buys = sorted(equity_buys, key=lambda x: x.get('notional', 0), reverse=True)[:5]
        top_sells = sorted(equity_sells, key=lambda x: x.get('notional', 0), reverse=True)[:5]

        buys_text = "\n".join([f"• **{o['symbol']}**: ${o['notional']:,.0f}" for o in top_buys]) or "None"
        sells_text = "\n".join([f"• **{o['symbol']}**: ${o['notional']:,.0f}" for o in top_sells]) or "None"

        # Crypto
        crypto_text = ""
        for o in crypto_buys:
            crypto_text += f"🟢 **{o['symbol']}**: ${o['notional']:,.0f}\n"
        for o in crypto_sells:
            crypto_text += f"🔴 **{o['symbol']}**: ${o['notional']:,.0f}\n"

        if not crypto_text:
            crypto_text = "None"

        fields = [
            {
                "name": "📅 Info",
                "value": (
                    f"**Date:** {as_of_date}\n"
                    f"**Broker:** {broker_mode.upper()}\n"
                    f"**Mode:** {'DRY RUN' if dry_run else 'LIVE'}\n"
                    f"**Strategy:** {strategy_type or 'N/A'}"
                ),
                "inline": False
            }
        ]

        if model_promoted:
            # Extract strategy type if available
            strategy_info = ""
            if 'lstm' in model_promoted.lower():
                strategy_info = " (LSTM Neural Network)"
            elif 'simple' in model_promoted.lower() or 'xgb' in model_promoted.lower() or 'multi_horizon' in model_promoted.lower():
                strategy_info = " (XGBoost Multi-Horizon)"
            
            fields.append({
                "name": "🔄 Model Promotion",
                "value": f"**{model_promoted}**{strategy_info} → ACTIVE",
                "inline": False
            })

        if not kill_switch and not market_closed:
            fields.extend([
                {
                    "name": "💼 Allocation",
                    "value": f"Equities: {equity_allocation*100:.0f}% | BTC: {btc_allocation*100:.0f}%",
                    "inline": False
                },
                {
                    "name": "💰 Portfolio Value",
                    "value": f"${portfolio_value:,.0f}",
                    "inline": True
                },
                {
                    "name": "📊 Orders",
                    "value": f"{len(equity_buys)} equity buys, {len(equity_sells)} equity sells",
                    "inline": True
                },
                {
                    "name": "🟢 Top Equity Buys",
                    "value": buys_text,
                    "inline": True
                },
                {
                    "name": "🔴 Top Equity Sells",
                    "value": sells_text,
                    "inline": True
                },
                {
                    "name": "₿ Bitcoin Orders",
                    "value": crypto_text,
                    "inline": False
                }
            ])

        embed = {
            "title": title,
            "color": color,
            "fields": fields,
            "footer": {"text": "Investment Bot • Monthly Rebalance"},
            "timestamp": datetime.utcnow().isoformat()
        }

        return self._send(embed)

    def send_health_alert(self, issues: List[str], severity: str = "warning"):
        """Health check alert (anomalies only)."""
        color = 15158332 if severity == "error" else 16776960

        issues_text = "\n".join([f"• {issue}" for issue in issues])

        embed = {
            "title": "⚠️ HEALTH CHECK ALERT",
            "description": issues_text,
            "color": color,
            "footer": {"text": "Investment Bot • Health Monitor"},
            "timestamp": datetime.utcnow().isoformat()
        }

        return self._send(embed)

    def send_generic_alert(self, title: str, message: str, level: str = "info"):
        """Generic alert."""
        color = {
            "info": 3447003,
            "warning": 16776960,
            "error": 15158332,
            "success": 5763719
        }.get(level, 3447003)

        embed = {
            "title": title,
            "description": message,
            "color": color,
            "footer": {"text": "Investment Bot"},
            "timestamp": datetime.utcnow().isoformat()
        }

        return self._send(embed)
