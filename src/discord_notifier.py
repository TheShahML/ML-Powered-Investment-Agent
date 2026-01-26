"""
Discord Webhook Notification Service.
Sends trading signals, rebalance updates, and portfolio performance to Discord.
"""

import requests
import json
import urllib.parse
from datetime import datetime
from typing import Dict, List, Optional
from loguru import logger
import os


class DiscordNotifier:
    """
    Sends formatted notifications to Discord via webhook.
    """

    def __init__(self, webhook_url: str = None):
        self.webhook_url = webhook_url or os.environ.get('DISCORD_WEBHOOK_URL')
        if not self.webhook_url:
            logger.warning("Discord webhook URL not configured")

    def _send_message(self, embed: Dict) -> bool:
        """Send a message to Discord."""
        if not self.webhook_url:
            logger.warning("Discord webhook URL not set, skipping notification")
            return False

        payload = {"embeds": [embed]}

        try:
            response = requests.post(
                self.webhook_url,
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            logger.info("Discord notification sent successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to send Discord notification: {e}")
            return False

    def send_daily_signals(self, signals_df, date: str = None):
        """
        Send daily signal generation notification.

        Args:
            signals_df: DataFrame with ticker, score, rank columns
            date: Date string for the signals
        """
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')

        # Get top 15 stocks
        top_stocks = signals_df.head(15)

        # Format stock list
        stock_list = ""
        for i, (ticker, row) in enumerate(top_stocks.iterrows(), 1):
            score = row['score'] if 'score' in row else 0
            stock_list += f"`{i:2d}.` **{ticker}** (score: {score:.3f})\n"

        embed = {
            "title": "📊 Daily Signal Generation Complete",
            "description": f"ML predictions generated for **{date}**",
            "color": 3447003,  # Blue
            "fields": [
                {
                    "name": "🏆 Top 15 Stocks",
                    "value": stock_list or "No signals generated",
                    "inline": False
                },
                {
                    "name": "📈 Total Stocks Analyzed",
                    "value": f"{len(signals_df)} stocks",
                    "inline": True
                },
                {
                    "name": "⏰ Next Rebalance",
                    "value": "Check monthly-rebalance workflow",
                    "inline": True
                }
            ],
            "footer": {"text": "Investment Bot • Daily Signals"},
            "timestamp": datetime.utcnow().isoformat()
        }

        return self._send_message(embed)

    def send_enhanced_signals(
        self,
        signals_df,
        days_until_rebalance: int = 0,
        next_rebalance_date: str = None,
        date: str = None
    ):
        """
        Send enhanced signal notification with top 5 for day, week, and month.

        Args:
            signals_df: DataFrame with ticker, score, rank columns
            days_until_rebalance: Days until next 20-day rebalance
            next_rebalance_date: Estimated date of next rebalance (YYYY-MM-DD)
            date: Date string for the signals
        """
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')

        # Calculate next rebalance date if not provided
        if next_rebalance_date is None:
            from datetime import timedelta
            next_date = datetime.now()
            days_added = 0
            while days_added < days_until_rebalance:
                next_date += timedelta(days=1)
                if next_date.weekday() < 5:  # Skip weekends
                    days_added += 1
            next_rebalance_date = next_date.strftime('%Y-%m-%d')

        # Get top 5 stocks for each timeframe
        # For now, we show the same top stocks but label them differently
        # In practice, you might have different models for different horizons
        top_5 = signals_df.head(5)

        def format_top_5(df, prefix=""):
            text = ""
            for i, (ticker, row) in enumerate(df.iterrows(), 1):
                score = row['score'] if 'score' in row else 0
                text += f"`{i}.` **{ticker}** ({score:+.3f})\n"
            return text or "No signals"

        top_5_text = format_top_5(top_5)

        # Rebalance status text
        if days_until_rebalance == 0:
            rebalance_text = "🔔 **TODAY!**"
        elif days_until_rebalance == 1:
            rebalance_text = f"⏰ **Tomorrow** ({next_rebalance_date})"
        else:
            rebalance_text = f"📅 **{days_until_rebalance} days** ({next_rebalance_date})"

        # Create embed with multiple sections
        embed = {
            "title": "📊 Daily Trading Signals",
            "description": f"ML predictions for **{date}**\n*Strategy: Simple XGBoost (5 features)*",
            "color": 3447003,  # Blue
            "fields": [
                {
                    "name": "🌅 Top 5 - Day",
                    "value": top_5_text,
                    "inline": True
                },
                {
                    "name": "📅 Top 5 - Week",
                    "value": top_5_text,
                    "inline": True
                },
                {
                    "name": "📆 Top 5 - Month (20-day)",
                    "value": top_5_text,
                    "inline": True
                },
                {
                    "name": "📈 Universe Stats",
                    "value": f"Analyzed: **{len(signals_df)}** stocks",
                    "inline": True
                },
                {
                    "name": "🔄 Next Rebalance",
                    "value": rebalance_text,
                    "inline": True
                }
            ],
            "footer": {"text": "Investment Bot • Simple Momentum Strategy"},
            "timestamp": datetime.utcnow().isoformat()
        }

        return self._send_message(embed)

    def send_multi_horizon_signals(
        self,
        signals_1d: Optional,
        signals_5d: Optional,
        signals_20d: Optional,
        days_until_rebalance: int = 0,
        next_rebalance_date: str = None,
        date: str = None
    ):
        """
        Send signal notification with separate Top 5 for each time horizon.

        Args:
            signals_1d: 1-day model predictions
            signals_5d: 5-day model predictions
            signals_20d: 20-day model predictions (primary)
            days_until_rebalance: Days until next 20-day rebalance
            next_rebalance_date: Estimated date of next rebalance
            date: Date string for the signals
        """
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')

        # Format Top 5 for each horizon
        def format_top_5(df, horizon_label):
            if df is None or df.empty:
                return f"*No {horizon_label} signals*"

            text = ""
            top_5 = df.head(5)
            for i, (ticker, row) in enumerate(top_5.iterrows(), 1):
                score = row['score'] if 'score' in row else 0
                text += f"`{i}.` **{ticker}** ({score:+.3f})\n"
            return text

        top_5_1d = format_top_5(signals_1d, "1-day")
        top_5_5d = format_top_5(signals_5d, "5-day")
        top_5_20d = format_top_5(signals_20d, "20-day")

        # Rebalance status
        if days_until_rebalance == 0:
            rebalance_text = "🔔 **TODAY!**"
        elif days_until_rebalance == 1:
            rebalance_text = f"⏰ **Tomorrow** ({next_rebalance_date})"
        else:
            rebalance_text = f"📅 **{days_until_rebalance} days** ({next_rebalance_date})"

        # Create enhanced embed
        embed = {
            "title": "📊 Multi-Horizon Trading Signals",
            "description": (
                f"ML predictions for **{date}**\n"
                f"*Strategy: 3 XGBoost models (1d/5d/20d horizons)*\n"
                f"*Rebalancing uses 20-day model*"
            ),
            "color": 3447003,  # Blue
            "fields": [
                {
                    "name": "🌅 Top 5 - Daily (1-day)",
                    "value": top_5_1d,
                    "inline": True
                },
                {
                    "name": "📅 Top 5 - Weekly (5-day)",
                    "value": top_5_5d,
                    "inline": True
                },
                {
                    "name": "📆 Top 5 - Monthly (20-day) 🎯",
                    "value": top_5_20d,
                    "inline": True
                },
                {
                    "name": "📈 Universe Stats",
                    "value": f"Analyzed: **{len(signals_20d) if signals_20d is not None else 0}** stocks",
                    "inline": True
                },
                {
                    "name": "🔄 Next Rebalance",
                    "value": rebalance_text,
                    "inline": True
                }
            ],
            "footer": {"text": "Investment Bot • Multi-Horizon Strategy"},
            "timestamp": datetime.utcnow().isoformat()
        }

        return self._send_message(embed)

    def send_multi_horizon_training_complete(self, all_metrics: Dict):
        """
        Send notification when all 3 models finish training.

        Args:
            all_metrics: Dict with keys '1d', '5d', '20d' mapping to metrics
        """
        metrics_text = ""

        for horizon in ['1d', '5d', '20d']:
            if horizon in all_metrics:
                metrics = all_metrics[horizon]
                metrics_text += f"\n**{horizon.upper()} Model:**\n"
                metrics_text += f"• CV Mean Corr: {metrics.get('cv_mean_corr', 0):.4f}\n"
                metrics_text += f"• CV Std: {metrics.get('cv_std_corr', 0):.4f}\n"
                metrics_text += f"• Samples: {metrics.get('n_samples', 0):,}\n"

        embed = {
            "title": "🧠 Multi-Horizon Model Training Complete",
            "description": "3 XGBoost models retrained (1d/5d/20d forward returns)",
            "color": 10181046,  # Purple
            "fields": [
                {
                    "name": "📊 Model Performance",
                    "value": metrics_text or "No metrics available",
                    "inline": False
                },
                {
                    "name": "🎯 Primary Model",
                    "value": "**20-day** model drives portfolio rebalancing",
                    "inline": False
                }
            ],
            "footer": {"text": "Investment Bot • Model Training"},
            "timestamp": datetime.utcnow().isoformat()
        }

        return self._send_message(embed)

    def _generate_chart_url(
        self,
        portfolio_returns: List[float],
        spy_returns: List[float],
        qqq_returns: List[float],
        vti_returns: List[float],
        labels: List[str]
    ) -> str:
        """
        Generate a QuickChart.io URL for performance comparison chart.

        Args:
            portfolio_returns: List of cumulative portfolio returns (%)
            spy_returns: List of SPY cumulative returns (%)
            qqq_returns: List of QQQ cumulative returns (%)
            vti_returns: List of VTI cumulative returns (%)
            labels: Date labels for x-axis

        Returns:
            URL string for the chart image
        """
        chart_config = {
            "type": "line",
            "data": {
                "labels": labels,
                "datasets": [
                    {
                        "label": "Portfolio",
                        "data": portfolio_returns,
                        "borderColor": "#00ff00",
                        "backgroundColor": "rgba(0, 255, 0, 0.1)",
                        "fill": False,
                        "tension": 0.1
                    },
                    {
                        "label": "SPY",
                        "data": spy_returns,
                        "borderColor": "#ff6384",
                        "fill": False,
                        "tension": 0.1
                    },
                    {
                        "label": "QQQ",
                        "data": qqq_returns,
                        "borderColor": "#36a2eb",
                        "fill": False,
                        "tension": 0.1
                    },
                    {
                        "label": "VTI",
                        "data": vti_returns,
                        "borderColor": "#ffce56",
                        "fill": False,
                        "tension": 0.1
                    }
                ]
            },
            "options": {
                "responsive": True,
                "plugins": {
                    "title": {
                        "display": True,
                        "text": "Portfolio vs Benchmarks (%)"
                    },
                    "legend": {
                        "position": "bottom"
                    }
                },
                "scales": {
                    "y": {
                        "title": {
                            "display": True,
                            "text": "Return (%)"
                        }
                    }
                }
            }
        }

        # Encode chart config for URL
        chart_json = json.dumps(chart_config)
        encoded_chart = urllib.parse.quote(chart_json)

        return f"https://quickchart.io/chart?c={encoded_chart}&w=600&h=300&bkg=white"

    def send_portfolio_with_chart(
        self,
        account_info: Dict,
        positions: List[Dict],
        performance: Dict = None,
        benchmark_data: Dict = None,
        crypto_positions: List[Dict] = None
    ):
        """
        Send portfolio update with performance chart vs benchmarks.

        Args:
            account_info: Account equity/cash info
            positions: List of all current positions (equities + crypto)
            performance: Performance metrics dict with history
            benchmark_data: Dict with SPY, QQQ, VTI return histories
            crypto_positions: Optional list of crypto positions (for explicit display)
        """
        # Format positions (top 10 by value)
        positions_text = ""
        sorted_positions = sorted(positions, key=lambda x: x.get('market_value', 0), reverse=True)

        # Separate equities and crypto
        equities = [p for p in sorted_positions if not p.get('symbol', '').endswith('USD')]
        crypto = [p for p in sorted_positions if p.get('symbol', '').endswith('USD')]

        for p in equities[:8]:  # Top 8 equities
            symbol = p.get('symbol', 'N/A')
            value = p.get('market_value', 0)
            pnl_pct = p.get('unrealized_plpc', 0) * 100 if p.get('unrealized_plpc') else 0
            emoji = "🟢" if pnl_pct >= 0 else "🔴"
            positions_text += f"{emoji} **{symbol}**: ${value:,.0f} ({pnl_pct:+.1f}%)\n"

        if len(equities) > 8:
            positions_text += f"*...+{len(equities) - 8} more equities*\n"

        # Crypto section
        crypto_text = ""
        for p in crypto:
            symbol = p.get('symbol', 'N/A')
            value = p.get('market_value', 0)
            pnl_pct = p.get('unrealized_plpc', 0) * 100 if p.get('unrealized_plpc') else 0
            emoji = "🟢" if pnl_pct >= 0 else "🔴"
            crypto_text += f"{emoji} **{symbol}**: ${value:,.0f} ({pnl_pct:+.1f}%)\n"

        fields = [
            {
                "name": "💰 Total Equity",
                "value": f"${account_info.get('equity', 0):,.2f}",
                "inline": True
            },
            {
                "name": "💵 Cash",
                "value": f"${account_info.get('cash', 0):,.2f}",
                "inline": True
            },
            {
                "name": "📊 Holdings",
                "value": f"{len(equities)} stocks + {len(crypto)} crypto",
                "inline": True
            }
        ]

        # Performance comparison vs benchmarks
        if performance:
            portfolio_ret = performance.get('total_return_pct', 0)
            spy_ret = performance.get('spy_return_pct', 0)
            qqq_ret = performance.get('qqq_return_pct', 0)
            vti_ret = performance.get('vti_return_pct', 0)

            # Determine if beating benchmarks
            beating_spy = portfolio_ret > spy_ret
            beating_qqq = portfolio_ret > qqq_ret
            beating_vti = portfolio_ret > vti_ret

            perf_text = (
                f"**Portfolio:** {portfolio_ret:+.2f}%\n"
                f"**vs SPY:** {portfolio_ret - spy_ret:+.2f}% {'✅' if beating_spy else '❌'}\n"
                f"**vs QQQ:** {portfolio_ret - qqq_ret:+.2f}% {'✅' if beating_qqq else '❌'}\n"
                f"**vs VTI:** {portfolio_ret - vti_ret:+.2f}% {'✅' if beating_vti else '❌'}\n"
                f"**Max DD:** {performance.get('max_drawdown_pct', 0):.2f}%"
            )
            fields.append({
                "name": "📈 Performance vs Benchmarks",
                "value": perf_text,
                "inline": False
            })

        if positions_text:
            fields.append({
                "name": "🏦 Stock Holdings",
                "value": positions_text,
                "inline": True
            })

        if crypto_text:
            fields.append({
                "name": "₿ Crypto Holdings",
                "value": crypto_text,
                "inline": True
            })

        embed = {
            "title": "📋 Portfolio Status Update",
            "description": f"Weekly summary as of {datetime.now().strftime('%Y-%m-%d')}",
            "color": 10181046,  # Purple
            "fields": fields,
            "footer": {"text": "Investment Bot • Portfolio Update"},
            "timestamp": datetime.utcnow().isoformat()
        }

        # Add chart image if benchmark data available
        if benchmark_data and performance:
            try:
                chart_url = self._generate_chart_url(
                    portfolio_returns=benchmark_data.get('portfolio', [0]),
                    spy_returns=benchmark_data.get('spy', [0]),
                    qqq_returns=benchmark_data.get('qqq', [0]),
                    vti_returns=benchmark_data.get('vti', [0]),
                    labels=benchmark_data.get('labels', ['Start'])
                )
                embed["image"] = {"url": chart_url}
            except Exception as e:
                logger.warning(f"Could not generate chart: {e}")

        return self._send_message(embed)

    def send_rebalance_notification(
        self,
        orders: List[Dict],
        account_info: Dict,
        performance: Dict = None,
        dry_run: bool = False
    ):
        """
        Send rebalance execution notification.

        Args:
            orders: List of order dictionaries
            account_info: Account equity/cash info
            performance: Performance metrics
            dry_run: Whether this was a dry run
        """
        # Separate equity and crypto orders
        equity_orders = [o for o in orders if '/' not in o.get('symbol', '')]
        crypto_orders = [o for o in orders if '/' in o.get('symbol', '')]

        buy_orders = [o for o in equity_orders if o.get('side') == 'buy']
        sell_orders = [o for o in equity_orders if o.get('side') == 'sell']
        crypto_buys = [o for o in crypto_orders if o.get('side') == 'buy']
        crypto_sells = [o for o in crypto_orders if o.get('side') == 'sell']

        # Format equity buy orders
        buys_text = ""
        for o in buy_orders[:10]:  # Limit to 10
            buys_text += f"• **{o['symbol']}**: {o['qty']:.2f} shares (${o.get('notional', 0):.0f})\n"
        if len(buy_orders) > 10:
            buys_text += f"*...and {len(buy_orders) - 10} more*\n"

        # Format equity sell orders
        sells_text = ""
        for o in sell_orders[:10]:
            sells_text += f"• **{o['symbol']}**: {o['qty']:.2f} shares (${o.get('notional', 0):.0f})\n"
        if len(sell_orders) > 10:
            sells_text += f"*...and {len(sell_orders) - 10} more*\n"

        # Format crypto orders
        crypto_text = ""
        for o in crypto_buys:
            crypto_text += f"🟢 **{o['symbol']}**: ${o.get('notional', 0):,.0f}\n"
        for o in crypto_sells:
            crypto_text += f"🔴 **{o['symbol']}**: ${o.get('notional', 0):,.0f}\n"

        # Status color
        if dry_run:
            color = 16776960  # Yellow
            title = "🧪 Rebalance DRY RUN Complete"
        else:
            color = 3066993  # Green
            title = "✅ Monthly Rebalance Executed"

        fields = [
            {
                "name": "💰 Portfolio Value",
                "value": f"${account_info.get('equity', 0):,.2f}",
                "inline": True
            },
            {
                "name": "💵 Cash",
                "value": f"${account_info.get('cash', 0):,.2f}",
                "inline": True
            },
            {
                "name": "📊 Equity Orders",
                "value": f"{len(buy_orders)} buys, {len(sell_orders)} sells",
                "inline": True
            }
        ]

        if crypto_orders:
            fields.append({
                "name": "₿ Crypto Orders",
                "value": f"{len(crypto_buys)} buys, {len(crypto_sells)} sells",
                "inline": True
            })

        if buys_text:
            fields.append({
                "name": "🟢 Equity Buys",
                "value": buys_text or "None",
                "inline": False
            })

        if sells_text:
            fields.append({
                "name": "🔴 Equity Sells",
                "value": sells_text or "None",
                "inline": False
            })

        if crypto_text:
            fields.append({
                "name": "₿ Bitcoin Rebalance",
                "value": crypto_text,
                "inline": False
            })

        if performance:
            fields.append({
                "name": "📈 Performance",
                "value": (
                    f"Total Return: **{performance.get('total_return_pct', 0):.2f}%**\n"
                    f"Max Drawdown: **{performance.get('max_drawdown_pct', 0):.2f}%**"
                ),
                "inline": False
            })

        embed = {
            "title": title,
            "description": f"20-day rebalance {'simulation' if dry_run else 'execution'} completed",
            "color": color,
            "fields": fields,
            "footer": {"text": "Investment Bot • Monthly Rebalance"},
            "timestamp": datetime.utcnow().isoformat()
        }

        return self._send_message(embed)

    def send_portfolio_update(self, account_info: Dict, positions: List[Dict], performance: Dict = None):
        """
        Send portfolio status update (can be used for weekly summaries).

        Args:
            account_info: Account equity/cash info
            positions: List of current positions
            performance: Performance metrics
        """
        # Format positions
        positions_text = ""
        sorted_positions = sorted(positions, key=lambda x: x.get('market_value', 0), reverse=True)
        for p in sorted_positions[:10]:
            symbol = p.get('symbol', 'N/A')
            qty = p.get('qty', 0)
            value = p.get('market_value', 0)
            pnl_pct = p.get('unrealized_plpc', 0) * 100 if p.get('unrealized_plpc') else 0
            emoji = "🟢" if pnl_pct >= 0 else "🔴"
            positions_text += f"{emoji} **{symbol}**: {qty:.2f} shares (${value:,.0f}, {pnl_pct:+.1f}%)\n"

        if len(positions) > 10:
            positions_text += f"*...and {len(positions) - 10} more positions*"

        fields = [
            {
                "name": "💰 Total Equity",
                "value": f"${account_info.get('equity', 0):,.2f}",
                "inline": True
            },
            {
                "name": "💵 Cash",
                "value": f"${account_info.get('cash', 0):,.2f}",
                "inline": True
            },
            {
                "name": "📊 Positions",
                "value": f"{len(positions)} holdings",
                "inline": True
            }
        ]

        if performance:
            fields.append({
                "name": "📈 Performance",
                "value": (
                    f"**Total Return:** {performance.get('total_return_pct', 0):+.2f}%\n"
                    f"**Max Drawdown:** {performance.get('max_drawdown_pct', 0):.2f}%\n"
                    f"**Days Tracked:** {performance.get('days_tracked', 0)}"
                ),
                "inline": False
            })

        if positions_text:
            fields.append({
                "name": "🏦 Current Holdings",
                "value": positions_text,
                "inline": False
            })

        embed = {
            "title": "📋 Portfolio Status Update",
            "description": f"Weekly summary as of {datetime.now().strftime('%Y-%m-%d')}",
            "color": 10181046,  # Purple
            "fields": fields,
            "footer": {"text": "Investment Bot • Portfolio Update"},
            "timestamp": datetime.utcnow().isoformat()
        }

        return self._send_message(embed)

    def send_alert(self, title: str, message: str, level: str = "info"):
        """
        Send a generic alert message.

        Args:
            title: Alert title
            message: Alert message
            level: "info", "warning", or "error"
        """
        colors = {
            "info": 3447003,     # Blue
            "warning": 16776960, # Yellow
            "error": 15158332    # Red
        }
        emojis = {
            "info": "ℹ️",
            "warning": "⚠️",
            "error": "🚨"
        }

        embed = {
            "title": f"{emojis.get(level, 'ℹ️')} {title}",
            "description": message,
            "color": colors.get(level, 3447003),
            "footer": {"text": "Investment Bot • Alert"},
            "timestamp": datetime.utcnow().isoformat()
        }

        return self._send_message(embed)

    def send_model_training_complete(self, metrics: Dict):
        """
        Send notification when model training completes.

        Args:
            metrics: Training metrics dictionary
        """
        metrics_text = ""
        for name, value in metrics.items():
            if isinstance(value, float):
                metrics_text += f"• **{name}:** {value:.4f}\n"
            else:
                metrics_text += f"• **{name}:** {value}\n"

        embed = {
            "title": "🧠 Model Training Complete",
            "description": "Weekly stacking ensemble retraining finished",
            "color": 10181046,  # Purple
            "fields": [
                {
                    "name": "📊 Model Performance",
                    "value": metrics_text or "No metrics available",
                    "inline": False
                }
            ],
            "footer": {"text": "Investment Bot • Model Training"},
            "timestamp": datetime.utcnow().isoformat()
        }

        return self._send_message(embed)
