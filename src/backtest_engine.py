"""Walk-forward portfolio backtest with realistic costs and promotion gate."""
import pandas as pd
import numpy as np
from loguru import logger
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import json
import os


class WalkForwardBacktest:
    """Realistic portfolio simulation for model validation."""

    def __init__(self, config: Dict):
        self.config = config
        self.top_n = config.get('top_n', 20)
        self.max_weight = config.get('max_weight', 0.08)
        self.weight_method = config.get('weight_method', 'inverse_vol')
        self.cost_bps = config.get('cost_bps', 15.0)
        self.slippage_bps = config.get('slippage_bps', 5.0)
        self.turnover_buffer_pct = config.get('turnover_buffer_pct', 1.0)
        self.rebalance_freq = config.get('rebalance_freq_days', 20)
        self.regime_filter = config.get('regime_filter', True)

    def run_backtest(
        self,
        strategy,
        data: pd.DataFrame,
        start_date: str,
        end_date: str,
        initial_capital: float = 100000.0
    ) -> Dict:
        """
        Run walk-forward backtest.

        Args:
            strategy: Strategy instance with compute_signals() method
            data: Historical data with MultiIndex (timestamp, symbol)
            start_date: Backtest start
            end_date: Backtest end
            initial_capital: Starting portfolio value

        Returns:
            Dict with performance metrics
        """
        logger.info(f"Running walk-forward backtest: {start_date} to {end_date}")

        # Get rebalance dates
        dates = pd.to_datetime(data.index.get_level_values(0)).unique().sort_values()
        dates = dates[(dates >= start_date) & (dates <= end_date)]

        rebalance_dates = [dates[i] for i in range(0, len(dates), self.rebalance_freq)]

        portfolio = {
            'cash': initial_capital,
            'holdings': {},
            'equity_curve': [],
            'dates': [],
            'trades': []
        }

        logger.info(f"Rebalancing {len(rebalance_dates)} times over {len(dates)} days")

        for i, rebal_date in enumerate(rebalance_dates):
            # Get data up to rebalance date
            data_slice = data[data.index.get_level_values(0) <= rebal_date]

            if len(data_slice) < 500:
                continue

            # Generate signals
            try:
                signals = strategy.compute_signals(data_slice)
            except Exception as e:
                logger.warning(f"Signal generation failed at {rebal_date}: {e}")
                continue

            # Compute target weights
            targets = self._compute_target_weights(signals, data_slice, rebal_date)

            # Apply regime filter
            if self.regime_filter:
                spy_regime = self._get_spy_regime(data_slice, rebal_date)
                if spy_regime == 'bearish':
                    # Scale down equity exposure
                    targets = {k: v * 0.5 for k, v in targets.items()}
                    logger.info(f"{rebal_date}: Bearish regime - scaling exposure to 50%")

            # Execute rebalance
            portfolio = self._execute_rebalance(portfolio, targets, data_slice, rebal_date)

            # Mark to market until next rebalance
            next_rebal_idx = min(i + 1, len(rebalance_dates) - 1)
            next_rebal_date = rebalance_dates[next_rebal_idx]

            mtm_dates = dates[(dates > rebal_date) & (dates <= next_rebal_date)]
            for mtm_date in mtm_dates:
                portfolio_value = self._mark_to_market(portfolio, data, mtm_date)
                portfolio['equity_curve'].append(portfolio_value)
                portfolio['dates'].append(mtm_date)

        # Compute metrics
        return self._compute_metrics(portfolio)

    def _compute_target_weights(
        self,
        signals: pd.DataFrame,
        data: pd.DataFrame,
        as_of_date
    ) -> Dict[str, float]:
        """Compute target portfolio weights."""
        top_stocks = signals.head(self.top_n).index.tolist()

        if self.weight_method == 'equal_weight':
            weight = 1.0 / len(top_stocks)
            return {s: weight for s in top_stocks}

        elif self.weight_method == 'inverse_vol':
            # Compute 60d realized volatility
            vols = {}
            for symbol in top_stocks:
                try:
                    symbol_data = data.xs(symbol, level=1)
                    symbol_data = symbol_data[symbol_data.index <= as_of_date]
                    if len(symbol_data) < 60:
                        vols[symbol] = 0.20  # Default vol
                        continue

                    returns = symbol_data['close'].pct_change()
                    vol = returns.tail(60).std() * np.sqrt(252)
                    vols[symbol] = max(vol, 0.01)  # Floor at 1%
                except:
                    vols[symbol] = 0.20

            # Inverse vol weights
            inv_vols = {s: 1.0 / v for s, v in vols.items()}
            total_inv_vol = sum(inv_vols.values())
            weights = {s: iv / total_inv_vol for s, iv in inv_vols.items()}

            # Cap at max_weight
            weights = {s: min(w, self.max_weight) for s, w in weights.items()}

            # Renormalize
            total = sum(weights.values())
            return {s: w / total for s, w in weights.items()}

        return {}

    def _get_spy_regime(self, data: pd.DataFrame, as_of_date) -> str:
        """Determine market regime from SPY 200d SMA."""
        try:
            spy_data = data.xs('SPY', level=1)
            spy_data = spy_data[spy_data.index <= as_of_date]

            if len(spy_data) < 200:
                return 'bullish'  # Default

            sma_200 = spy_data['close'].tail(200).mean()
            current_price = spy_data['close'].iloc[-1]

            return 'bearish' if current_price < sma_200 else 'bullish'
        except:
            return 'bullish'

    def _execute_rebalance(
        self,
        portfolio: Dict,
        target_weights: Dict[str, float],
        data: pd.DataFrame,
        rebal_date
    ) -> Dict:
        """Execute rebalance with transaction costs."""
        # Get current prices
        prices = {}
        for symbol in set(list(portfolio['holdings'].keys()) + list(target_weights.keys())):
            try:
                symbol_data = data.xs(symbol, level=1)
                symbol_data = symbol_data[symbol_data.index <= rebal_date]
                prices[symbol] = symbol_data['close'].iloc[-1]
            except:
                continue

        # Current portfolio value
        current_value = portfolio['cash'] + sum(
            portfolio['holdings'].get(s, 0) * prices.get(s, 0)
            for s in portfolio['holdings']
        )

        # Compute target dollar amounts
        target_dollars = {s: w * current_value for s, w in target_weights.items()}

        # Current dollar amounts
        current_dollars = {
            s: portfolio['holdings'].get(s, 0) * prices.get(s, 0)
            for s in set(list(portfolio['holdings'].keys()) + list(target_dollars.keys()))
        }

        # Compute trades (with turnover buffer)
        trades = {}
        total_turnover = 0.0

        for symbol in set(list(current_dollars.keys()) + list(target_dollars.keys())):
            current = current_dollars.get(symbol, 0)
            target = target_dollars.get(symbol, 0)
            diff = target - current

            # Apply turnover buffer
            if abs(diff / current_value) < (self.turnover_buffer_pct / 100.0):
                continue

            trades[symbol] = diff
            total_turnover += abs(diff)

        # Apply transaction costs
        total_cost_bps = self.cost_bps + self.slippage_bps
        transaction_cost = total_turnover * (total_cost_bps / 10000.0)

        # Update portfolio
        new_holdings = portfolio['holdings'].copy()
        new_cash = portfolio['cash'] - transaction_cost

        for symbol, trade_dollars in trades.items():
            price = prices.get(symbol, 0)
            if price == 0:
                continue

            shares = trade_dollars / price
            new_holdings[symbol] = new_holdings.get(symbol, 0) + shares
            new_cash -= trade_dollars

        # Remove zero positions
        new_holdings = {s: q for s, q in new_holdings.items() if abs(q) > 0.001}

        portfolio['holdings'] = new_holdings
        portfolio['cash'] = new_cash
        portfolio['trades'].append({
            'date': rebal_date,
            'num_trades': len(trades),
            'turnover': total_turnover / current_value if current_value > 0 else 0,
            'cost': transaction_cost
        })

        logger.info(
            f"Rebalanced: {len(trades)} trades, "
            f"turnover {total_turnover/current_value*100:.1f}%, "
            f"cost ${transaction_cost:.2f}"
        )

        return portfolio

    def _mark_to_market(self, portfolio: Dict, data: pd.DataFrame, mtm_date) -> float:
        """Mark portfolio to market."""
        total_value = portfolio['cash']

        for symbol, qty in portfolio['holdings'].items():
            try:
                symbol_data = data.xs(symbol, level=1)
                symbol_data = symbol_data[symbol_data.index <= mtm_date]
                price = symbol_data['close'].iloc[-1]
                total_value += qty * price
            except:
                continue

        return total_value

    def _compute_metrics(self, portfolio: Dict) -> Dict:
        """Compute performance metrics."""
        equity_curve = np.array(portfolio['equity_curve'])
        dates = portfolio['dates']

        if len(equity_curve) < 2:
            return {'error': 'Insufficient data'}

        # Returns
        returns = np.diff(equity_curve) / equity_curve[:-1]

        # CAGR
        total_return = (equity_curve[-1] / equity_curve[0]) - 1
        years = len(equity_curve) / 252
        cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

        # Sharpe
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0

        # Max drawdown
        cummax = np.maximum.accumulate(equity_curve)
        drawdowns = (equity_curve - cummax) / cummax
        max_dd = np.min(drawdowns)

        # Worst month
        monthly_returns = []
        for i in range(0, len(returns), 21):
            chunk = returns[i:i+21]
            if len(chunk) > 0:
                monthly_returns.append(np.prod(1 + chunk) - 1)
        worst_month = np.min(monthly_returns) if monthly_returns else 0

        # Turnover stats
        turnovers = [t['turnover'] for t in portfolio['trades']]
        avg_turnover = np.mean(turnovers) if turnovers else 0

        # Cost drag
        total_cost = sum(t['cost'] for t in portfolio['trades'])
        cost_drag = total_cost / equity_curve[0] if equity_curve[0] > 0 else 0

        return {
            'cagr': cagr,
            'sharpe': sharpe,
            'max_drawdown': max_dd,
            'worst_month': worst_month,
            'total_return': total_return,
            'avg_turnover': avg_turnover,
            'total_trades': len(portfolio['trades']),
            'cost_drag_cumulative': cost_drag,
            'final_value': equity_curve[-1],
            'equity_curve': equity_curve.tolist(),
            'dates': [str(d) for d in dates]
        }


def check_promotion_gate(
    candidate_metrics: Dict,
    baseline_metrics: Dict,
    sharpe_margin: float = 0.2,
    maxdd_tolerance: float = 0.05
) -> Tuple[bool, Dict]:
    """
    Check if candidate passes promotion gate.

    Args:
        candidate_metrics: Backtest results for candidate
        baseline_metrics: Dict of baseline results (momentum, spy, etc.)
        sharpe_margin: Required Sharpe improvement
        maxdd_tolerance: Max acceptable MaxDD increase

    Returns:
        (passed: bool, details: dict)
    """
    details = {
        'sharpe_margin_required': sharpe_margin,
        'maxdd_tolerance': maxdd_tolerance
    }

    # Get best baseline Sharpe and best baseline MaxDD
    baseline_sharpes = [m['sharpe'] for m in baseline_metrics.values() if 'sharpe' in m]
    baseline_maxdds = [m['max_drawdown'] for m in baseline_metrics.values() if 'max_drawdown' in m]

    if not baseline_sharpes:
        logger.warning("No baseline Sharpe ratios available, skipping gate")
        return True, {'reason': 'no_baselines'}

    best_baseline_sharpe = max(baseline_sharpes)
    best_baseline_maxdd = max(baseline_maxdds)  # Least negative

    candidate_sharpe = candidate_metrics.get('sharpe', 0)
    candidate_maxdd = candidate_metrics.get('max_drawdown', 0)

    # Check Sharpe margin
    sharpe_achieved = candidate_sharpe - best_baseline_sharpe
    sharpe_pass = sharpe_achieved >= sharpe_margin

    # Check MaxDD tolerance
    maxdd_diff = candidate_maxdd - best_baseline_maxdd
    maxdd_pass = maxdd_diff <= maxdd_tolerance

    passed = sharpe_pass and maxdd_pass

    details.update({
        'best_baseline_sharpe': best_baseline_sharpe,
        'candidate_sharpe': candidate_sharpe,
        'sharpe_margin_achieved': sharpe_achieved,
        'sharpe_pass': sharpe_pass,
        'best_baseline_maxdd': best_baseline_maxdd,
        'candidate_maxdd': candidate_maxdd,
        'maxdd_diff': maxdd_diff,
        'maxdd_pass': maxdd_pass,
        'overall_pass': passed
    })

    logger.info(f"Promotion Gate: {'PASS' if passed else 'FAIL'}")
    logger.info(f"  Sharpe: {candidate_sharpe:.3f} vs {best_baseline_sharpe:.3f} (margin: {sharpe_achieved:+.3f}, req: {sharpe_margin:+.3f})")
    logger.info(f"  MaxDD: {candidate_maxdd:.3f} vs {best_baseline_maxdd:.3f} (diff: {maxdd_diff:+.3f}, tol: {maxdd_tolerance:+.3f})")

    return passed, details


def save_backtest_report(
    candidate_metrics: Dict,
    baseline_metrics: Dict,
    gate_result: Tuple[bool, Dict],
    output_dir: str = 'backtests'
) -> str:
    """Save backtest report to JSON."""
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'backtest_{timestamp}.json'
    filepath = os.path.join(output_dir, filename)

    report = {
        'timestamp': datetime.now().isoformat(),
        'candidate': candidate_metrics,
        'baselines': baseline_metrics,
        'gate': {
            'passed': gate_result[0],
            'details': gate_result[1]
        }
    }

    with open(filepath, 'w') as f:
        json.dump(report, f, indent=2)

    logger.info(f"Backtest report saved: {filepath}")
    return filepath
