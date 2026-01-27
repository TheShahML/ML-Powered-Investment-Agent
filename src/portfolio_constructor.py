"""
Shared portfolio construction logic for backtest and live execution.
Ensures identical logic between simulation and real trading.
"""
import pandas as pd
import numpy as np
from loguru import logger
from typing import Dict, List, Tuple, Optional
from datetime import date


class PortfolioConstructor:
    """
    Shared portfolio construction logic.
    
    Policy:
    - Universe: S&P 500 ∪ NASDAQ-100 filtered to tradeable/liquid
    - Rank by primary horizon score (20d)
    - Select top N (default 25)
    - Weight by inverse-vol (default 60d realized vol), then cap max_weight (default 10%), renormalize
    - Apply turnover buffer (skip trades if |target_w - current_w| < 1%)
    - Apply regime filter: if SPY close < SPY 200d SMA => equity exposure scaled down
    """

    def __init__(
        self,
        top_n: int = 25,
        max_weight: float = 0.10,
        vol_window: int = 60,
        turnover_buffer_pct: float = 1.0,
        regime_filter: bool = True,
        regime_scale: float = 0.5
    ):
        self.top_n = top_n
        self.max_weight = max_weight
        self.vol_window = vol_window
        self.turnover_buffer_pct = turnover_buffer_pct / 100.0
        self.regime_filter = regime_filter
        self.regime_scale = regime_scale  # Scale factor when bearish

    def compute_target_weights(
        self,
        signals: pd.DataFrame,
        data: pd.DataFrame,
        as_of_date,
        current_weights: Optional[Dict[str, float]] = None,
        spy_data: Optional[pd.DataFrame] = None
    ) -> Tuple[Dict[str, float], Dict]:
        """
        Compute target portfolio weights.

        Args:
            signals: DataFrame with 'score' column, indexed by symbol
            data: Historical data (MultiIndex: timestamp, symbol)
            as_of_date: As-of date for data
            current_weights: Current portfolio weights {symbol: weight}
            spy_data: SPY data (if None, will try to extract from data)

        Returns:
            (target_weights: Dict[str, float], metadata: Dict)
        """
        # Select top N stocks
        top_stocks = signals.head(self.top_n).index.tolist()

        if not top_stocks:
            logger.warning("No stocks selected")
            return {}, {'error': 'no_stocks'}

        # Compute inverse-volatility weights
        weights = self._compute_inverse_vol_weights(
            top_stocks, data, as_of_date
        )

        # Apply regime filter if enabled
        regime_info = {}
        if self.regime_filter:
            regime, spy_price, spy_sma = self._get_regime(
                data, as_of_date, spy_data
            )
            regime_info = {
                'regime': regime,
                'spy_price': spy_price,
                'spy_sma_200': spy_sma
            }

            if regime == 'bearish':
                logger.info(f"Bearish regime detected - scaling equity exposure to {self.regime_scale*100:.0f}%")
                weights = {s: w * self.regime_scale for s, w in weights.items()}

        # Apply turnover buffer if current weights provided
        if current_weights:
            weights = self._apply_turnover_buffer(weights, current_weights)

        metadata = {
            'top_n': len(weights),
            'regime': regime_info,
            'total_weight': sum(weights.values())
        }

        return weights, metadata

    def _compute_inverse_vol_weights(
        self,
        symbols: List[str],
        data: pd.DataFrame,
        as_of_date
    ) -> Dict[str, float]:
        """Compute inverse-volatility weights."""
        as_of_ts = pd.Timestamp(as_of_date)
        vols = {}
        for symbol in symbols:
            try:
                symbol_data = data.xs(symbol, level=1)
                idx_tz = getattr(symbol_data.index, "tz", None)
                if idx_tz is not None and as_of_ts.tzinfo is None:
                    as_of_ts_local = as_of_ts.tz_localize(idx_tz)
                else:
                    as_of_ts_local = as_of_ts
                symbol_data = symbol_data[symbol_data.index <= as_of_ts_local]

                if len(symbol_data) < self.vol_window:
                    vols[symbol] = 0.20  # Default 20% annualized
                    continue

                returns = symbol_data['close'].pct_change()
                vol = returns.tail(self.vol_window).std() * np.sqrt(252)
                vols[symbol] = max(vol, 0.01)  # Floor at 1%

            except Exception as e:
                logger.warning(f"Could not compute vol for {symbol}: {e}")
                vols[symbol] = 0.20

        # Inverse vol weights
        inv_vols = {s: 1.0 / v for s, v in vols.items()}
        total_inv_vol = sum(inv_vols.values())

        if total_inv_vol == 0:
            # Fallback to equal weight
            weight = 1.0 / len(symbols)
            return {s: weight for s in symbols}

        weights = {s: iv / total_inv_vol for s, iv in inv_vols.items()}

        # Cap at max_weight
        weights = {s: min(w, self.max_weight) for s, w in weights.items()}

        # Renormalize
        total = sum(weights.values())
        if total > 0:
            weights = {s: w / total for s, w in weights.items()}

        return weights

    def _get_regime(
        self,
        data: pd.DataFrame,
        as_of_date,
        spy_data: Optional[pd.DataFrame] = None
    ) -> Tuple[str, float, float]:
        """
        Determine market regime from SPY 200d SMA.

        Returns:
            (regime: 'bullish'|'bearish', spy_price: float, spy_sma_200: float)
        """
        # Try to get SPY data
        if spy_data is not None:
            spy_series = spy_data
        else:
            try:
                spy_series = data.xs('SPY', level=1)
            except KeyError:
                logger.error("SPY not found in data - cannot determine regime")
                raise ValueError("SPY data required for regime filter")

        as_of_ts = pd.Timestamp(as_of_date)
        idx_tz = getattr(spy_series.index, "tz", None)
        if idx_tz is not None and as_of_ts.tzinfo is None:
            as_of_ts = as_of_ts.tz_localize(idx_tz)
        spy_series = spy_series[spy_series.index <= as_of_ts]

        if len(spy_series) < 200:
            # Use shorter SMA if available, otherwise default to conservative (bearish)
            if len(spy_series) >= 50:
                # Use 50-day SMA as fallback
                sma_50 = spy_series['close'].tail(50).mean()
                current_price = spy_series['close'].iloc[-1]
                regime = 'bearish' if current_price < sma_50 else 'bullish'
                logger.warning(f"SPY data insufficient ({len(spy_series)} days < 200) - using 50-day SMA fallback: {regime}")
                return regime, float(current_price), float(sma_50)
            elif len(spy_series) >= 20:
                # Use 20-day SMA as last resort
                sma_20 = spy_series['close'].tail(20).mean()
                current_price = spy_series['close'].iloc[-1]
                regime = 'bearish' if current_price < sma_20 else 'bullish'
                logger.warning(f"SPY data insufficient ({len(spy_series)} days < 200) - using 20-day SMA fallback: {regime}")
                return regime, float(current_price), float(sma_20)
            else:
                # Very little data - default to conservative (bearish) to reduce risk
                logger.warning(f"SPY data very insufficient ({len(spy_series)} days < 20) - defaulting to BEARISH (conservative)")
                current_price = spy_series['close'].iloc[-1] if len(spy_series) > 0 else 0.0
                return 'bearish', float(current_price), float(current_price)

        sma_200 = spy_series['close'].tail(200).mean()
        current_price = spy_series['close'].iloc[-1]

        regime = 'bearish' if current_price < sma_200 else 'bullish'
        return regime, float(current_price), float(sma_200)

    def _apply_turnover_buffer(
        self,
        target_weights: Dict[str, float],
        current_weights: Dict[str, float]
    ) -> Dict[str, float]:
        """Apply turnover buffer to avoid small trades."""
        # If no current weights (first rebalance), return targets as-is
        if not current_weights:
            return target_weights
        
        all_symbols = set(target_weights.keys()) | set(current_weights.keys())
        filtered = {}

        for symbol in all_symbols:
            target_w = target_weights.get(symbol, 0.0)
            current_w = current_weights.get(symbol, 0.0)
            diff = abs(target_w - current_w)

            if diff >= self.turnover_buffer_pct:
                filtered[symbol] = target_w
            else:
                # Keep current weight if change is too small
                if current_w > 0:
                    filtered[symbol] = current_w

        # Renormalize
        total = sum(filtered.values())
        if total > 0:
            filtered = {s: w / total for s, w in filtered.items()}

        return filtered

    def compute_turnover(
        self,
        target_weights: Dict[str, float],
        current_weights: Dict[str, float],
        portfolio_value: float
    ) -> Dict:
        """
        Compute turnover metrics.

        Returns:
            Dict with 'turnover', 'cost', 'num_trades', etc.
        """
        all_symbols = set(target_weights.keys()) | set(current_weights.keys())
        total_turnover = 0.0
        num_trades = 0

        for symbol in all_symbols:
            target_dollars = target_weights.get(symbol, 0.0) * portfolio_value
            current_dollars = current_weights.get(symbol, 0.0) * portfolio_value
            diff = abs(target_dollars - current_dollars)

            if diff > 0:
                total_turnover += diff
                if diff / portfolio_value >= self.turnover_buffer_pct:
                    num_trades += 1

        turnover_pct = total_turnover / (2 * portfolio_value) if portfolio_value > 0 else 0.0

        return {
            'turnover_pct': turnover_pct,
            'turnover_dollars': total_turnover,
            'num_trades': num_trades
        }

