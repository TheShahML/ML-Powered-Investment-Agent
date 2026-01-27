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

        # Canary top 5
        canary_list = "\n".join([f"{i}. **{sym}** ({score:+.1f}%)" for i, (sym, score) in enumerate(canary_top10[:5], 1)])

        # Performance since last rebal
        perf_since = (
            f"**Actual:** {performance_since_rebal.get('actual', 0):+.1f}%\n"
            f"**Canary:** {performance_since_rebal.get('canary', 0):+.1f}%\n"
            f"**SPY:** {performance_since_rebal.get('spy', 0):+.1f}%"
        )

        # Rolling 30d
        perf_30d = (
            f"Actual: {performance_rolling.get('actual_30d', 0):+.1f}% | "
            f"Canary: {performance_rolling.get('canary_30d', 0):+.1f}% | "
            f"SPY: {performance_rolling.get('spy_30d', 0):+.1f}%"
        )

        embed = {
            "title": f"📊 MULTI-HORIZON SIGNALS | {as_of_date}",
            "description": "3 XGBoost models (1d/5d/20d horizons)",
            "color": 3447003,
            "fields": [
                {
                    "name": "📅 Info",
                    "value": (
                        f"**As-of:** {as_of_date}\n"
                        f"**Active Model:** {active_model.get('version', 'None')}\n"
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
                    "name": "📈 Canary Top 5 (Pure Momentum)",
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

        # Canary top 10
        canary_list = "\n".join([f"{i}. **{sym}** ({score:+.1f}%)" for i, (sym, score) in enumerate(canary_top10[:10], 1)])

        # Performance since last rebal
        perf_since = (
            f"**Actual:** {performance_since_rebal.get('actual', 0):+.1f}%\n"
            f"**Canary:** {performance_since_rebal.get('canary', 0):+.1f}%\n"
            f"**SPY:** {performance_since_rebal.get('spy', 0):+.1f}%\n"
            f"**BTC:** {performance_since_rebal.get('btc', 0):+.1f}%"
        )

        # Rolling 30d
        perf_30d = (
            f"Actual: {performance_rolling.get('actual_30d', 0):+.1f}% | "
            f"Canary: {performance_rolling.get('canary_30d', 0):+.1f}% | "
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
                    "name": "📈 Canary Top 10 (Momentum)",
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
        gate_details: Dict
    ):
        """Weekly training completion report (multi-horizon)."""
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

        # Backtest
        bt_text = (
            f"**CAGR:** {backtest_candidate.get('cagr', 0)*100:.1f}%\n"
            f"**Sharpe:** {backtest_candidate.get('sharpe', 0):.2f}\n"
            f"**MaxDD:** {backtest_candidate.get('max_drawdown', 0)*100:.1f}%\n"
            f"**Turnover:** {backtest_candidate.get('avg_turnover', 0)*100:.1f}%"
        )

        # vs Baselines
        baseline_text = ""
        for name, metrics in backtest_baselines.items():
            baseline_text += (
                f"**{name.upper()}:** "
                f"{metrics.get('cagr', 0)*100:.1f}% CAGR, "
                f"{metrics.get('sharpe', 0):.2f} Sharpe\n"
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

        embed = {
            "title": "🧠 WEEKLY TRAINING COMPLETE",
            "color": 5763719 if gate_passed else 15158332,
            "fields": [
                {
                    "name": "🆕 Candidate Model",
                    "value": f"**Version:** {candidate_version}\n**Window:** {training_window[0]} to {training_window[1]}",
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
            ],
            "footer": {"text": "Investment Bot • Weekly Training"},
            "timestamp": datetime.utcnow().isoformat()
        }

        return self._send(embed)

    def send_monthly_rebalance_execution(
        self,
        broker_mode: str,
        as_of_date: str,
        model_promoted: Optional[str],
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
                    f"**Mode:** {'DRY RUN' if dry_run else 'LIVE'}"
                ),
                "inline": False
            }
        ]

        if model_promoted:
            fields.append({
                "name": "🔄 Model Promotion",
                "value": f"**{model_promoted}** → ACTIVE",
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
