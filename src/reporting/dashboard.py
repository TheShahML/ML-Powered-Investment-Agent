"""
Matplotlib + Seaborn dashboard generation for Discord reporting.
Generates professional dashboard images showing portfolio performance and signals.
"""
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple
import os
from loguru import logger

# Set seaborn style for professional look
sns.set_style("whitegrid")
sns.set_palette("husl")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'


class DashboardGenerator:
    """Generate dashboard images for Discord reporting."""

    def __init__(self, output_dir: str = 'reports'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def generate_daily_dashboard(
        self,
        as_of_date: date,
        active_model: Dict,
        ml_top10: List[Tuple[str, float]],
        canary_top10: List[Tuple[str, float]],
        current_holdings: Dict[str, float],
        shadow_state: Dict,
        performance_metrics: Dict,
        broker_mode: str,
        kill_switch: bool,
        filename: Optional[str] = None
    ) -> str:
        """
        Generate daily dashboard image with seaborn styling.
        
        Args:
            as_of_date: As-of date
            ml_top10: List of (symbol, score) tuples for ML top 10
            canary_top10: List of (symbol, score) tuples for canary top 10
            current_holdings: {symbol: weight} for current portfolio
            shadow_state: Shadow portfolio state
            performance_metrics: Performance metrics dict
            broker_mode: 'paper' or 'live'
            kill_switch: Kill switch status
            filename: Optional filename (defaults to dashboard_YYYYMMDD.png)
        
        Returns:
            Path to generated image
        """
        if filename is None:
            filename = f"dashboard_{as_of_date.strftime('%Y%m%d')}.png"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Create figure with subplots using seaborn style
        fig = plt.figure(figsize=(16, 10), facecolor='white')
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)
        
        # 1. Equity curve (top, spans 2 columns)
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_equity_curve(ax1, shadow_state, as_of_date)
        
        # 2. Drawdown curve (top right)
        ax2 = fig.add_subplot(gs[0, 2])
        self._plot_drawdown(ax2, shadow_state)
        
        # 3. Top 10 ML picks (middle left)
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_top_picks(ax3, ml_top10, "ML Top 10", color='#2E86AB')
        
        # 4. Top 10 Canary picks (middle center)
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_top_picks(ax4, canary_top10, "Canary Top 10", color='#06A77D')
        
        # 5. Current holdings weights (middle right)
        ax5 = fig.add_subplot(gs[1, 2])
        self._plot_holdings(ax5, current_holdings)
        
        # 6. Info table (bottom, spans all columns)
        ax6 = fig.add_subplot(gs[2, :])
        ax6.axis('off')
        self._plot_info_table(ax6, as_of_date, active_model, performance_metrics, broker_mode, kill_switch)
        
        plt.suptitle(f"Investment Bot Dashboard | {as_of_date.strftime('%Y-%m-%d')}", 
                     fontsize=18, fontweight='bold', y=0.98, color='#1a1a1a')
        
        plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        
        logger.info(f"Dashboard saved: {filepath}")
        return filepath

    def generate_trades_dashboard(
        self,
        as_of_date: date,
        target_weights: Dict[str, float],
        orders: List[Dict],
        filename: Optional[str] = None
    ) -> str:
        """
        Generate trades dashboard for rebalance day with seaborn styling.
        
        Args:
            as_of_date: Rebalance date
            target_weights: {symbol: weight} target weights
            orders: List of order dicts
            filename: Optional filename
        
        Returns:
            Path to generated image
        """
        if filename is None:
            filename = f"trades_{as_of_date.strftime('%Y%m%d')}.png"
        
        filepath = os.path.join(self.output_dir, filename)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 8), facecolor='white')
        
        # Left: Target weights bar chart
        if target_weights:
            symbols = list(target_weights.keys())[:20]  # Top 20
            weights = [target_weights[s] * 100 for s in symbols]
            
            # Use gradient palette
            colors = sns.color_palette("viridis", n_colors=len(symbols))
            
            bars = axes[0].barh(range(len(symbols)), weights, color=colors, alpha=0.85,
                               edgecolor='white', linewidth=1.5)
            
            # Add value labels
            for i, (bar, weight) in enumerate(zip(bars, weights)):
                axes[0].text(weight + 0.3, i, f'{weight:.1f}%', va='center', fontsize=9, fontweight='bold')
            
            axes[0].set_yticks(range(len(symbols)))
            axes[0].set_yticklabels(symbols, fontsize=10, fontweight='bold')
            axes[0].set_xlabel('Weight (%)', fontsize=11, fontweight='bold')
            axes[0].set_title('Target Portfolio Weights (Top 20)', fontsize=12, fontweight='bold', pad=10)
            axes[0].grid(axis='x', alpha=0.3, linestyle='--')
            sns.despine(ax=axes[0], left=True, bottom=False)
        
        # Right: Orders list
        axes[1].axis('off')
        if orders:
            buys = [o for o in orders if o.get('side') == 'buy']
            sells = [o for o in orders if o.get('side') == 'sell']
            
            text_lines = [
                f"📊 REBALANCE TRADES | {as_of_date.strftime('%Y-%m-%d')}\n",
                f"Total Orders: {len(orders)}\n",
                f"\n🟢 BUYS ({len(buys)}):\n"
            ]
            
            for o in sorted(buys, key=lambda x: x.get('notional', 0), reverse=True)[:10]:
                text_lines.append(f"  {o['symbol']}: ${o.get('notional', 0):,.0f}")
            
            text_lines.append(f"\n🔴 SELLS ({len(sells)}):\n")
            for o in sorted(sells, key=lambda x: x.get('notional', 0), reverse=True)[:10]:
                text_lines.append(f"  {o['symbol']}: ${o.get('notional', 0):,.0f}")
            
            axes[1].text(0.1, 0.9, '\n'.join(text_lines), 
                        transform=axes[1].transAxes,
                        fontsize=10, family='monospace',
                        verticalalignment='top')
        
        plt.suptitle(f"Rebalance Trades | {as_of_date.strftime('%Y-%m-%d')}", 
                     fontsize=16, fontweight='bold', y=0.98, color='#1a1a1a')
        
        plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        
        logger.info(f"Trades dashboard saved: {filepath}")
        return filepath

    def generate_weekly_training_dashboard(
        self,
        as_of_date: date,
        candidate_version: str,
        strategy_name: str,
        training_window: Tuple[str, str],
        cv_metrics: Dict,
        candidate_backtest: Dict,
        baselines: Dict[str, Dict],
        gate_passed: bool,
        gate_details: Dict,
        filename: Optional[str] = None
    ) -> str:
        """
        Weekly training dashboard: CV summary + backtest curves + metrics + gate.
        """
        if filename is None:
            filename = f"weekly_training_{as_of_date.strftime('%Y%m%d')}.png"

        filepath = os.path.join(self.output_dir, filename)

        fig = plt.figure(figsize=(18, 11), facecolor='white')
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)

        # Top: equity curves
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_backtest_equity_curves(
            ax1,
            series={
                f"{strategy_name} (Candidate)": candidate_backtest,
                "PURE_MOMENTUM": baselines.get('pure_momentum', {}),
                "SPY_BUY_HOLD": baselines.get('spy_buy_hold', {})
            }
        )

        # Top right: drawdown (candidate only)
        ax2 = fig.add_subplot(gs[0, 2])
        self._plot_backtest_drawdown(ax2, candidate_backtest, title="Candidate Drawdown")

        # Middle left: CV IC by horizon (if present)
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_cv_ic(ax3, cv_metrics)

        # Middle center: metrics comparison bar chart
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_metric_comparison(
            ax4,
            candidate_label=strategy_name,
            candidate=candidate_backtest,
            baselines=baselines
        )

        # Middle right: promotion gate summary
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.axis('off')
        self._plot_gate_summary(ax5, gate_passed, gate_details)

        # Bottom: header/info
        ax6 = fig.add_subplot(gs[2, :])
        ax6.axis('off')
        self._plot_weekly_info_table(
            ax6,
            as_of_date=as_of_date,
            candidate_version=candidate_version,
            strategy_name=strategy_name,
            training_window=training_window,
            candidate_backtest=candidate_backtest,
            gate_passed=gate_passed
        )

        status = "PASSED" if gate_passed else "FAILED"
        plt.suptitle(
            f"Weekly Training Report | {as_of_date.strftime('%Y-%m-%d')} | Gate: {status}",
            fontsize=18, fontweight='bold', y=0.98, color='#1a1a1a'
        )

        plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()

        logger.info(f"Weekly training dashboard saved: {filepath}")
        return filepath

    def generate_strategy_comparison_dashboard(
        self,
        as_of_date: date,
        strategy_results: Dict[str, Dict],
        baselines: Dict[str, Dict],
        best_strategy: str,
        training_window: Tuple[str, str],
        filename: Optional[str] = None
    ) -> str:
        """
        Strategy comparison dashboard (works for 1+ strategies).
        """
        if filename is None:
            filename = f"strategy_comparison_{as_of_date.strftime('%Y%m%d')}.png"

        filepath = os.path.join(self.output_dir, filename)

        fig = plt.figure(figsize=(18, 10), facecolor='white')
        gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.35)

        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_strategy_table(ax1, strategy_results, baselines, best_strategy)

        ax2 = fig.add_subplot(gs[0, 2])
        self._plot_metric_radar_like(ax2, strategy_results, baselines, best_strategy)

        ax3 = fig.add_subplot(gs[1, :])
        ax3.axis('off')
        self._plot_strategy_comparison_info(ax3, as_of_date, training_window, best_strategy)

        plt.suptitle(
            f"Strategy Comparison | {as_of_date.strftime('%Y-%m-%d')} | Selected: {best_strategy}",
            fontsize=18, fontweight='bold', y=0.98, color='#1a1a1a'
        )

        plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()

        logger.info(f"Strategy comparison dashboard saved: {filepath}")
        return filepath

    def _plot_equity_curve(self, ax, shadow_state: Dict, as_of_date: date):
        """Plot equity curves for ML, canary, and SPY with seaborn styling."""
        ml_curve = shadow_state.get('ml', {}).get('equity_curve', [])
        canary_curve = shadow_state.get('canary', {}).get('equity_curve', [])
        spy_curve = shadow_state.get('benchmarks', {}).get('spy', {}).get('equity_curve', [])
        
        ml_dates = shadow_state.get('ml', {}).get('dates', [])
        canary_dates = shadow_state.get('canary', {}).get('dates', [])
        spy_dates = shadow_state.get('benchmarks', {}).get('spy', {}).get('dates', [])
        
        # Normalize to start at 1.0
        if ml_curve:
            ml_normalized = np.array(ml_curve) / ml_curve[0] if ml_curve[0] > 0 else np.array(ml_curve)
            if ml_dates:
                dates_ml = [datetime.fromisoformat(d).date() for d in ml_dates]
                ax.plot(dates_ml, ml_normalized, label='ML Shadow', linewidth=2.5, 
                       color='#2E86AB', marker='o', markersize=3, alpha=0.9)
        
        if canary_curve:
            canary_normalized = np.array(canary_curve) / canary_curve[0] if canary_curve[0] > 0 else np.array(canary_curve)
            if canary_dates:
                dates_canary = [datetime.fromisoformat(d).date() for d in canary_dates]
                ax.plot(dates_canary, canary_normalized, label='Benchmark Momentum (Canary)', linewidth=2.5, 
                       color='#06A77D', linestyle='--', marker='s', markersize=3, alpha=0.9)
        
        if spy_curve:
            spy_normalized = np.array(spy_curve) / spy_curve[0] if spy_curve[0] > 0 else np.array(spy_curve)
            if spy_dates:
                dates_spy = [datetime.fromisoformat(d).date() for d in spy_dates]
                ax.plot(dates_spy, spy_normalized, label='SPY', linewidth=2, 
                       color='#6C757D', alpha=0.7, linestyle=':')
        
        ax.set_xlabel('Date', fontsize=11, fontweight='bold')
        ax.set_ylabel('Normalized Value', fontsize=11, fontweight='bold')
        ax.set_title('Equity Curves (Normalized)', fontsize=12, fontweight='bold', pad=10)
        ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        sns.despine(ax=ax, left=False, bottom=False)

    def _plot_drawdown(self, ax, shadow_state: Dict):
        """Plot drawdown curve with seaborn styling."""
        ml_curve = shadow_state.get('ml', {}).get('equity_curve', [])
        ml_dates = shadow_state.get('ml', {}).get('dates', [])
        
        if ml_curve and len(ml_curve) > 1:
            equity_array = np.array(ml_curve)
            cummax = np.maximum.accumulate(equity_array)
            drawdown = (equity_array - cummax) / cummax * 100
            
            if ml_dates:
                dates = [datetime.fromisoformat(d).date() for d in ml_dates]
                # Use gradient fill for drawdown
                ax.fill_between(dates, drawdown, 0, color='#DC3545', alpha=0.4, interpolate=True)
                ax.plot(dates, drawdown, color='#C82333', linewidth=2, alpha=0.8)
        
        ax.set_xlabel('Date', fontsize=11, fontweight='bold')
        ax.set_ylabel('Drawdown (%)', fontsize=11, fontweight='bold')
        ax.set_title('Drawdown', fontsize=12, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        sns.despine(ax=ax, left=False, bottom=False)

    def _plot_top_picks(self, ax, top_picks: List[Tuple[str, float]], title: str, color: str = '#2E86AB'):
        """Plot top picks bar chart with seaborn styling."""
        if not top_picks:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes,
                   fontsize=12, style='italic', color='gray')
            ax.set_title(title, fontsize=12, fontweight='bold')
            return
        
        symbols = [s for s, _ in top_picks[:10]]
        scores = [sc for _, sc in top_picks[:10]]
        
        # Normalize scores for display
        if max(scores) > 0:
            scores_normalized = [s / max(scores) * 100 for s in scores]
        else:
            scores_normalized = scores
        
        # Create gradient colors
        colors = sns.light_palette(color, n_colors=len(symbols), reverse=True)
        
        bars = ax.barh(range(len(symbols)), scores_normalized, color=colors, alpha=0.85, 
                      edgecolor='white', linewidth=1.5)
        
        # Add value labels on bars
        for i, (bar, score) in enumerate(zip(bars, scores_normalized)):
            ax.text(score + 1, i, f'{score:.1f}', va='center', fontsize=9, fontweight='bold')
        
        ax.set_yticks(range(len(symbols)))
        ax.set_yticklabels(symbols, fontsize=10, fontweight='bold')
        ax.set_xlabel('Score (normalized)', fontsize=10, fontweight='bold')
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        sns.despine(ax=ax, left=True, bottom=False)

    def _plot_holdings(self, ax, holdings: Dict[str, float]):
        """Plot current holdings weights with seaborn styling."""
        if not holdings:
            ax.text(0.5, 0.5, 'No holdings', ha='center', va='center', transform=ax.transAxes,
                   fontsize=12, style='italic', color='gray')
            ax.set_title('Current Holdings', fontsize=12, fontweight='bold')
            return
        
        # Top 10 holdings
        sorted_holdings = sorted(holdings.items(), key=lambda x: x[1], reverse=True)[:10]
        symbols = [s for s, _ in sorted_holdings]
        weights = [w * 100 for _, w in sorted_holdings]
        
        # Use gradient palette
        colors = sns.color_palette("Blues", n_colors=len(symbols))
        
        bars = ax.barh(range(len(symbols)), weights, color=colors, alpha=0.85,
                      edgecolor='white', linewidth=1.5)
        
        # Add value labels
        for i, (bar, weight) in enumerate(zip(bars, weights)):
            ax.text(weight + 0.5, i, f'{weight:.1f}%', va='center', fontsize=9, fontweight='bold')
        
        ax.set_yticks(range(len(symbols)))
        ax.set_yticklabels(symbols, fontsize=10, fontweight='bold')
        ax.set_xlabel('Weight (%)', fontsize=10, fontweight='bold')
        ax.set_title('Current Holdings (Top 10)', fontsize=12, fontweight='bold', pad=10)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        sns.despine(ax=ax, left=True, bottom=False)

    def _plot_info_table(self, ax, as_of_date: date, active_model: Dict, 
                        performance_metrics: Dict, broker_mode: str, kill_switch: bool):
        """Plot info table."""
        table_data = [
            ['As-of Date', as_of_date.strftime('%Y-%m-%d')],
            ['Active Model', active_model.get('version', 'N/A')],
            ['Strategy', active_model.get('strategy_type', 'N/A')],
            ['Broker Mode', broker_mode.upper()],
            ['Kill Switch', 'ENABLED' if kill_switch else 'Disabled'],
        ]
        
        # Add performance metrics
        since_rebal = performance_metrics.get('since_rebalance', {})
        if since_rebal:
            table_data.extend([
                ['Since Rebalance', f"{since_rebal.get('days', 0)}d"],
                ['ML Return', f"{since_rebal.get('ml', 0):+.1f}%"],
                ['Canary Return', f"{since_rebal.get('canary', 0):+.1f}%"],
                ['SPY Return', f"{since_rebal.get('spy', 0):+.1f}%"],
            ])
        
        rolling_30d = performance_metrics.get('rolling_30d', {})
        if rolling_30d:
            table_data.extend([
                ['Rolling 30d ML', f"{rolling_30d.get('ml', 0):+.1f}%"],
                ['Rolling 30d Canary', f"{rolling_30d.get('canary', 0):+.1f}%"],
                ['Rolling 30d SPY', f"{rolling_30d.get('spy', 0):+.1f}%"],
            ])
        
        # Create table with better styling
        table = ax.table(cellText=table_data,
                        cellLoc='left',
                        loc='center',
                        colWidths=[0.3, 0.7],
                        bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2.2)
        
        # Style table cells
        for i in range(len(table_data)):
            # Header column
            table[(i, 0)].set_facecolor('#E8F4F8')
            table[(i, 0)].set_text_props(weight='bold', color='#1a1a1a')
            table[(i, 0)].set_edgecolor('white')
            table[(i, 0)].set_linewidth(1.5)
            
            # Value column
            table[(i, 1)].set_facecolor('#F8F9FA')
            table[(i, 1)].set_text_props(color='#495057')
            table[(i, 1)].set_edgecolor('white')
            table[(i, 1)].set_linewidth(1.5)
        
        # Highlight important rows
        for i, row in enumerate(table_data):
            if 'Return' in row[0] or 'Sharpe' in row[0]:
                table[(i, 1)].set_facecolor('#FFF3CD')
        
        ax.set_title('Portfolio Information', fontweight='bold', fontsize=12, pad=20, color='#1a1a1a')

    # -----------------------
    # Weekly / Comparison helpers
    # -----------------------

    def _parse_backtest_series(self, metrics: Dict):
        equity = metrics.get('equity_curve') or []
        dates = metrics.get('dates') or []
        if not equity or not dates or len(equity) != len(dates):
            return None, None
        try:
            dts = [datetime.fromisoformat(str(d)).date() for d in dates]
        except Exception:
            dts = [pd.to_datetime(d).date() for d in dates]
        return np.array(equity, dtype=float), dts

    def _plot_backtest_equity_curves(self, ax, series: Dict[str, Dict]):
        plotted = 0
        for label, metrics in series.items():
            if not metrics:
                continue
            eq, dts = self._parse_backtest_series(metrics)
            if eq is None:
                continue
            if eq[0] != 0:
                eq = eq / eq[0]
            style = '-' if 'Candidate' in label else '--'
            alpha = 0.9 if 'Candidate' in label else 0.6
            lw = 2.8 if 'Candidate' in label else 2.0
            ax.plot(dts, eq, label=label, linewidth=lw, linestyle=style, alpha=alpha)
            plotted += 1

        ax.set_title("Equity Curves (Normalized)", fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel("Date", fontsize=11, fontweight='bold')
        ax.set_ylabel("Normalized Value", fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        if plotted:
            ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
        sns.despine(ax=ax, left=False, bottom=False)

    def _plot_backtest_drawdown(self, ax, metrics: Dict, title: str = "Drawdown"):
        eq, dts = self._parse_backtest_series(metrics)
        if eq is None or len(eq) < 2:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes,
                    fontsize=12, style='italic', color='gray')
            ax.set_title(title, fontsize=12, fontweight='bold')
            return
        cummax = np.maximum.accumulate(eq)
        dd = (eq - cummax) / cummax * 100
        ax.fill_between(dts, dd, 0, color='#DC3545', alpha=0.35, interpolate=True)
        ax.plot(dts, dd, color='#C82333', linewidth=2.0, alpha=0.8)
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel("Date", fontsize=11, fontweight='bold')
        ax.set_ylabel("Drawdown (%)", fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        sns.despine(ax=ax, left=False, bottom=False)

    def _plot_cv_ic(self, ax, cv_metrics: Dict):
        # Expect cv_metrics like {'1d': {...}, '5d': {...}, '20d': {...}} or a flat dict
        horizons = ['1d', '5d', '20d']
        if all(h in cv_metrics for h in horizons):
            means = [cv_metrics[h].get('cv_mean_ic', 0) for h in horizons]
            stds = [cv_metrics[h].get('cv_std_ic', 0) for h in horizons]
            ax.bar(horizons, means, yerr=stds, color=sns.color_palette("Blues", n_colors=3), alpha=0.85,
                   edgecolor='white', linewidth=1.5, capsize=4)
            ax.axhline(0, color='#6C757D', linewidth=1)
            ax.set_title("CV Information Coefficient (IC)", fontsize=12, fontweight='bold', pad=10)
            ax.set_ylabel("IC", fontsize=11, fontweight='bold')
            ax.grid(axis='y', alpha=0.3, linestyle='--')
        else:
            ax.text(0.5, 0.5, 'CV not available', ha='center', va='center', transform=ax.transAxes,
                    fontsize=12, style='italic', color='gray')
            ax.set_title("CV Information Coefficient (IC)", fontsize=12, fontweight='bold', pad=10)
        sns.despine(ax=ax, left=False, bottom=False)

    def _plot_metric_comparison(self, ax, candidate_label: str, candidate: Dict, baselines: Dict[str, Dict]):
        rows = []
        def _row(name, m):
            return {
                'name': name,
                'Sharpe': float(m.get('sharpe', 0)),
                'CAGR%': float(m.get('cagr', 0) * 100),
                'MaxDD%': float(m.get('max_drawdown', 0) * 100),
            }
        rows.append(_row(candidate_label, candidate))
        if 'pure_momentum' in baselines:
            rows.append(_row('PURE_MOMENTUM', baselines['pure_momentum']))
        if 'spy_buy_hold' in baselines:
            rows.append(_row('SPY', baselines['spy_buy_hold']))

        dfm = pd.DataFrame(rows)
        if dfm.empty:
            ax.text(0.5, 0.5, 'No metrics', ha='center', va='center', transform=ax.transAxes,
                    fontsize=12, style='italic', color='gray')
            ax.set_title("Metrics Comparison", fontsize=12, fontweight='bold')
            return

        # Plot Sharpe only (clean). Put CAGR/MaxDD into annotations.
        colors = ['#2E86AB'] + ['#06A77D'] * (len(dfm) - 1)
        ax.bar(dfm['name'], dfm['Sharpe'], color=colors, alpha=0.85, edgecolor='white', linewidth=1.5)
        ax.axhline(0, color='#6C757D', linewidth=1)
        ax.set_title("Sharpe (higher is better)", fontsize=12, fontweight='bold', pad=10)
        ax.set_ylabel("Sharpe", fontsize=11, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_xticklabels(dfm['name'], rotation=15, ha='right', fontsize=10, fontweight='bold')

        # Annotate with CAGR/MaxDD
        for i, r in dfm.iterrows():
            ax.text(
                i, r['Sharpe'],
                f"CAGR {r['CAGR%']:.0f}%\nMaxDD {r['MaxDD%']:.0f}%",
                ha='center', va='bottom', fontsize=9, fontweight='bold'
            )
        sns.despine(ax=ax, left=False, bottom=False)

    def _plot_gate_summary(self, ax, gate_passed: bool, gate_details: Dict):
        status = "PASSED ✅" if gate_passed else "FAILED ❌"
        sharpe_req = gate_details.get('sharpe_margin_required', 0)
        sharpe_got = gate_details.get('sharpe_margin_achieved', 0)
        dd_tol = gate_details.get('maxdd_tolerance', 0)
        dd_diff = gate_details.get('maxdd_diff', 0)
        table_data = [
            ['Gate Status', status],
            ['Sharpe Margin', f"{sharpe_got:+.3f} (req {sharpe_req:+.3f})"],
            ['MaxDD Diff', f"{dd_diff:+.3f} (tol {dd_tol:+.3f})"],
        ]
        table = ax.table(cellText=table_data, cellLoc='left', loc='center',
                         colWidths=[0.35, 0.65], bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2.0)
        for i in range(len(table_data)):
            table[(i, 0)].set_facecolor('#E8F4F8')
            table[(i, 0)].set_text_props(weight='bold', color='#1a1a1a')
            table[(i, 0)].set_edgecolor('white')
            table[(i, 0)].set_linewidth(1.5)
            table[(i, 1)].set_facecolor('#F8F9FA')
            table[(i, 1)].set_text_props(color='#495057', weight='bold' if i == 0 else 'normal')
            table[(i, 1)].set_edgecolor('white')
            table[(i, 1)].set_linewidth(1.5)
        if gate_passed:
            table[(0, 1)].set_facecolor('#D4EDDA')
        else:
            table[(0, 1)].set_facecolor('#F8D7DA')
        ax.set_title('Promotion Gate', fontweight='bold', fontsize=12, pad=20, color='#1a1a1a')

    def _plot_weekly_info_table(
        self,
        ax,
        as_of_date: date,
        candidate_version: str,
        strategy_name: str,
        training_window: Tuple[str, str],
        candidate_backtest: Dict,
        gate_passed: bool
    ):
        table_data = [
            ['As-of Date', as_of_date.strftime('%Y-%m-%d')],
            ['Candidate Version', candidate_version],
            ['Strategy', strategy_name],
            ['Training Window', f"{training_window[0]} → {training_window[1]}"],
            ['Gate', 'PASSED ✅' if gate_passed else 'FAILED ❌'],
            ['CAGR', f"{candidate_backtest.get('cagr', 0) * 100:.1f}%"],
            ['Sharpe', f"{candidate_backtest.get('sharpe', 0):.2f}"],
            ['MaxDD', f"{candidate_backtest.get('max_drawdown', 0) * 100:.1f}%"],
            ['Avg Turnover', f"{candidate_backtest.get('avg_turnover', 0) * 100:.1f}%"],
        ]
        table = ax.table(cellText=table_data, cellLoc='left', loc='center',
                         colWidths=[0.25, 0.75], bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2.2)
        for i in range(len(table_data)):
            table[(i, 0)].set_facecolor('#E8F4F8')
            table[(i, 0)].set_text_props(weight='bold', color='#1a1a1a')
            table[(i, 0)].set_edgecolor('white')
            table[(i, 0)].set_linewidth(1.5)
            table[(i, 1)].set_facecolor('#F8F9FA')
            table[(i, 1)].set_text_props(color='#495057')
            table[(i, 1)].set_edgecolor('white')
            table[(i, 1)].set_linewidth(1.5)
        # Highlight gate row
        gate_row = 4
        table[(gate_row, 1)].set_facecolor('#D4EDDA' if gate_passed else '#F8D7DA')
        ax.set_title('Weekly Summary', fontweight='bold', fontsize=12, pad=20, color='#1a1a1a')

    def _plot_strategy_table(self, ax, strategy_results: Dict[str, Dict], baselines: Dict[str, Dict], best: str):
        ax.axis('off')
        rows = []
        for k, v in strategy_results.items():
            # Avoid emoji/star glyph issues in matplotlib fonts on some systems
            rows.append([k, f"{v.get('cagr', 0)*100:.1f}%", f"{v.get('sharpe', 0):.2f}", f"{v.get('max_drawdown', 0)*100:.1f}%", "BEST" if k == best else ""])
        for b, v in baselines.items():
            rows.append([b, f"{v.get('cagr', 0)*100:.1f}%", f"{v.get('sharpe', 0):.2f}", f"{v.get('max_drawdown', 0)*100:.1f}%", ""])
        col_labels = ['Name', 'CAGR', 'Sharpe', 'MaxDD', 'Selected']
        table = ax.table(cellText=rows, colLabels=col_labels, cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.8)
        # header style
        for j in range(len(col_labels)):
            table[(0, j)].set_facecolor('#1F77B4')
            table[(0, j)].set_text_props(color='white', weight='bold')
            table[(0, j)].set_edgecolor('white')
        ax.set_title("Performance Table", fontsize=12, fontweight='bold', pad=10)

    def _plot_metric_radar_like(self, ax, strategy_results: Dict[str, Dict], baselines: Dict[str, Dict], best: str):
        # Simple horizontal comparison of Sharpe (primary) for readability
        labels = []
        values = []
        colors = []
        for k, v in strategy_results.items():
            labels.append(k)
            values.append(v.get('sharpe', 0))
            colors.append('#2E86AB' if k == best else '#6C757D')
        for b, v in baselines.items():
            labels.append(b)
            values.append(v.get('sharpe', 0))
            colors.append('#06A77D')
        if not labels:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes,
                    fontsize=12, style='italic', color='gray')
            ax.set_title("Sharpe Comparison", fontsize=12, fontweight='bold')
            return
        y = np.arange(len(labels))
        ax.barh(y, values, color=colors, alpha=0.85, edgecolor='white', linewidth=1.5)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=10, fontweight='bold')
        ax.set_title("Sharpe Comparison", fontsize=12, fontweight='bold', pad=10)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        sns.despine(ax=ax, left=True, bottom=False)

    def _plot_strategy_comparison_info(self, ax, as_of_date: date, training_window: Tuple[str, str], best: str):
        text = (
            f"Date: {as_of_date.strftime('%Y-%m-%d')}\n"
            f"Training Window: {training_window[0]} → {training_window[1]}\n"
            f"Selected Strategy: {best}\n"
            f"\nThis report is used by the monthly rebalance workflow. "
            f"The candidate is only promoted if it passes the promotion gate."
        )
        ax.text(0.02, 0.95, text, transform=ax.transAxes, fontsize=11, family='monospace',
                verticalalignment='top')

