"""Liquidity-Conditioned Reversal (LC-Reversal) strategy."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
try:
    from loguru import logger
except Exception:  # pragma: no cover - fallback for minimal local environments
    import logging
    logger = logging.getLogger(__name__)

from src.strategy_base import BaseStrategy


@dataclass
class LCParams:
    n_universe: int = 500
    pct_tail: float = 0.10
    vol_z_min: float = 1.0
    impact_z_min: float = 1.0
    n_long: int = 10
    n_short: int = 10
    enable_shorts: bool = True
    gross_exposure: float = 1.0
    max_position_weight: float = 0.03
    weight_method: str = "equal_weight"  # equal_weight | inverse_vol
    adv_pct: float = 0.005
    hold_days: int = 2
    enable_stop_loss: bool = False
    stop_atr_mult: float = 1.5
    bear_gross_mult: float = 0.5
    earnings_filter_enabled: bool = False


class LCReversalStrategy(BaseStrategy):
    """Daily cross-sectional mean-reversion with liquidity/impact conditioning."""

    def __init__(self, config: Dict, model_dir: str = "models", horizon: str = "1d"):
        super().__init__(config, model_dir, horizon)
        self.feature_cols = [
            "ret_1d",
            "dollar_vol_1d",
            "vol_z",
            "impact",
            "impact_z",
            "atr_14",
            "avg_dollar_vol_20d",
            "volatility_20d",
        ]

    def _params(self) -> LCParams:
        cfg = self.config.get("lc_reversal", {}) or {}
        if not isinstance(cfg, dict):
            cfg = {}
        def _get(name: str, default: Any) -> Any:
            if name in cfg:
                return cfg.get(name)
            return self.config.get(name, default)

        return LCParams(
            n_universe=int(_get("n_universe", 500)),
            pct_tail=float(_get("pct_tail", 0.10)),
            vol_z_min=float(_get("vol_z_min", 1.0)),
            impact_z_min=float(_get("impact_z_min", 1.0)),
            n_long=int(_get("n_long", 10)),
            n_short=int(_get("n_short", 10)),
            enable_shorts=bool(_get("enable_shorts", True)),
            gross_exposure=float(_get("gross_exposure", 1.0)),
            max_position_weight=float(_get("max_position_weight", 0.03)),
            weight_method=str(_get("weight_method", "equal_weight")),
            adv_pct=float(_get("adv_pct", 0.005)),
            hold_days=int(_get("hold_days", 2)),
            enable_stop_loss=bool(_get("enable_stop_loss", False)),
            stop_atr_mult=float(_get("stop_atr_mult", 1.5)),
            bear_gross_mult=float(_get("bear_gross_mult", 0.5)),
            earnings_filter_enabled=bool(_get("earnings_filter_enabled", False)),
        )

    @staticmethod
    def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
        mean = series.rolling(window).mean()
        std = series.rolling(window).std(ddof=0)
        return (series - mean) / std.replace(0, np.nan)

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute LC-Reversal features from daily OHLCV bars."""
        if df.empty:
            return df

        data = df.copy().sort_index()
        grouped = data.groupby(level=1)

        data["ret_1d"] = grouped["close"].pct_change(1)
        data["dollar_vol_1d"] = data["close"] * data["volume"]

        data["vol_z"] = grouped["dollar_vol_1d"].transform(lambda x: self._rolling_zscore(x, 20))

        data["impact"] = (data["ret_1d"].abs() / data["dollar_vol_1d"]).replace([np.inf, -np.inf], np.nan)
        data["impact_z"] = grouped["impact"].transform(lambda x: self._rolling_zscore(x, 60))

        prev_close = grouped["close"].shift(1)
        tr1 = data["high"] - data["low"]
        tr2 = (data["high"] - prev_close).abs()
        tr3 = (data["low"] - prev_close).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        data["atr_14"] = true_range.groupby(level=1).transform(lambda x: x.rolling(14).mean())

        data["avg_dollar_vol_20d"] = grouped["dollar_vol_1d"].transform(lambda x: x.rolling(20).mean())
        data["volatility_20d"] = grouped["ret_1d"].transform(lambda x: x.rolling(20).std())

        return data

    def train(self, df: pd.DataFrame, target_col: str = "target", embargo_days: int = None) -> Dict:
        """Rule-based strategy: no model training required."""
        logger.info("LC-Reversal is rule-based; skipping training")
        return {
            "cv_mean_ic": 0.0,
            "cv_std_ic": 0.0,
            "n_samples": len(df) if df is not None else 0,
            "n_features": len(self.feature_cols),
            "embargo_days": 0,
        }

    def load_model(self, horizon: str = None) -> bool:
        """Rule-based strategy has no model artifact."""
        if horizon:
            self.horizon = horizon
        return True

    def _latest_snapshot(self, df_features: pd.DataFrame, as_of_date: Optional[date] = None) -> pd.DataFrame:
        as_of_ts = pd.Timestamp(as_of_date) if as_of_date is not None else None
        snap = df_features
        if as_of_ts is not None:
            idx = snap.index.get_level_values(0)
            idx_tz = getattr(idx, "tz", None)
            if idx_tz is not None and as_of_ts.tzinfo is None:
                as_of_ts = as_of_ts.tz_localize(idx_tz)
            snap = snap[snap.index.get_level_values(0) <= as_of_ts]
        latest = snap.groupby(level=1).tail(1).copy()
        latest.index = latest.index.get_level_values(1)
        return latest

    def _earnings_blocklist(self, as_of_date: date) -> set[str]:
        params = self._params()
        if not params.earnings_filter_enabled:
            return set()
        # Placeholder interface: no earnings provider wired yet.
        logger.warning("LC-Reversal earnings filter enabled, but no earnings calendar provider is configured. Skipping filter.")
        return set()

    def _filter_liquid_universe(self, latest: pd.DataFrame) -> pd.DataFrame:
        params = self._params()
        required = ["ret_1d", "vol_z", "impact_z", "avg_dollar_vol_20d", "dollar_vol_1d"]
        latest = latest.dropna(subset=required)
        latest = latest[latest["avg_dollar_vol_20d"] > 0]
        if latest.empty:
            return latest
        latest = latest.sort_values("avg_dollar_vol_20d", ascending=False).head(params.n_universe)
        return latest

    def _select_candidates(self, latest: pd.DataFrame, as_of_date: date) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        params = self._params()
        if latest.empty:
            return latest.iloc[0:0], latest.iloc[0:0], {"universe_size": 0}

        blocklist = self._earnings_blocklist(as_of_date)
        if blocklist:
            latest = latest[~latest.index.isin(blocklist)]

        if latest.empty:
            return latest.iloc[0:0], latest.iloc[0:0], {"universe_size": 0}

        low_cut = latest["ret_1d"].quantile(params.pct_tail)
        high_cut = latest["ret_1d"].quantile(1.0 - params.pct_tail)

        long_pool = latest[
            (latest["ret_1d"] <= low_cut)
            & (latest["vol_z"] > params.vol_z_min)
            & (latest["impact_z"] > params.impact_z_min)
        ].copy()

        short_pool = latest[
            (latest["ret_1d"] >= high_cut)
            & (latest["vol_z"] > params.vol_z_min)
            & (latest["impact_z"] > params.impact_z_min)
        ].copy()

        long_sel = long_pool.sort_values("ret_1d", ascending=True).head(params.n_long)
        short_sel = short_pool.sort_values("ret_1d", ascending=False).head(params.n_short)

        meta = {
            "universe_size": int(len(latest)),
            "long_pool": int(len(long_pool)),
            "short_pool": int(len(short_pool)),
            "long_selected": int(len(long_sel)),
            "short_selected": int(len(short_sel)),
            "ret_tail_low_cut": float(low_cut),
            "ret_tail_high_cut": float(high_cut),
        }
        return long_sel, short_sel, meta

    def _weights_for_side(self, frame: pd.DataFrame, side_gross: float, method: str) -> Dict[str, float]:
        if frame.empty or side_gross <= 0:
            return {}
        if method == "inverse_vol":
            vol = frame["volatility_20d"].replace(0, np.nan).fillna(frame["volatility_20d"].median())
            vol = vol.fillna(0.02)
            inv = 1.0 / np.clip(vol.values, 1e-4, None)
            raw = inv / inv.sum()
        else:
            raw = np.full(len(frame), 1.0 / len(frame))
        return {sym: float(w * side_gross) for sym, w in zip(frame.index, raw)}

    @staticmethod
    def _is_bearish_regime(df: pd.DataFrame, as_of_date: date) -> bool:
        try:
            spy = df.xs("SPY", level=1)
            spy = spy[spy.index <= pd.Timestamp(as_of_date)]
            if len(spy) < 50:
                return False
            window = 200 if len(spy) >= 200 else 50
            sma = spy["close"].tail(window).mean()
            return float(spy["close"].iloc[-1]) < float(sma)
        except Exception:
            return False

    def compute_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build ranked LC-Reversal candidate signals from the latest completed bar."""
        df_features = df if "ret_1d" in df.columns else self.compute_features(df)
        latest = self._latest_snapshot(df_features)
        latest = self._filter_liquid_universe(latest)

        as_of = pd.Timestamp(df_features.index.get_level_values(0).max()).date()
        long_sel, short_sel, _ = self._select_candidates(latest, as_of)

        rows: List[Dict[str, Any]] = []
        for sym, row in long_sel.iterrows():
            rows.append({
                "ticker": sym,
                "score": float(-row["ret_1d"]),
                "side": "buy",
                "ret_1d": float(row["ret_1d"]),
                "vol_z": float(row["vol_z"]),
                "impact_z": float(row["impact_z"]),
                "avg_dollar_vol_20d": float(row["avg_dollar_vol_20d"]),
                "atr_14": float(row.get("atr_14", np.nan)),
            })
        for sym, row in short_sel.iterrows():
            rows.append({
                "ticker": sym,
                "score": float(row["ret_1d"]),
                "side": "sell",
                "ret_1d": float(row["ret_1d"]),
                "vol_z": float(row["vol_z"]),
                "impact_z": float(row["impact_z"]),
                "avg_dollar_vol_20d": float(row["avg_dollar_vol_20d"]),
                "atr_14": float(row.get("atr_14", np.nan)),
            })

        if not rows:
            return pd.DataFrame(columns=["score", "rank", "side", "ret_1d", "vol_z", "impact_z", "avg_dollar_vol_20d", "atr_14"])

        out = pd.DataFrame(rows).set_index("ticker")
        out["rank"] = out["score"].rank(ascending=False, method="first")
        return out.sort_values("rank")

    def build_targets(
        self,
        df_features: pd.DataFrame,
        as_of_date: date,
        portfolio_value: float,
        current_positions: Dict[str, float],
        current_weights: Dict[str, float],
        lc_state: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, float], Dict[str, Any], Dict[str, Any]]:
        """Build target weights and updated LC state for execution."""
        params = self._params()
        latest = self._latest_snapshot(df_features, as_of_date=as_of_date)
        latest = self._filter_liquid_universe(latest)
        long_sel, short_sel, meta = self._select_candidates(latest, as_of_date)

        bearish = self._is_bearish_regime(df_features, as_of_date)
        gross = params.gross_exposure * (params.bear_gross_mult if bearish else 1.0)
        long_gross = gross * (0.5 if params.enable_shorts else 1.0)
        short_gross = gross * 0.5 if params.enable_shorts else 0.0

        long_w = self._weights_for_side(long_sel, long_gross, params.weight_method)
        short_w_abs = self._weights_for_side(short_sel, short_gross, params.weight_method)

        target_weights: Dict[str, float] = {}
        for sym, w in long_w.items():
            target_weights[sym] = float(w)
        if params.enable_shorts:
            for sym, w in short_w_abs.items():
                target_weights[sym] = float(-w)

        # ADV capacity cap
        for sym in list(target_weights.keys()):
            if sym not in latest.index:
                continue
            adv20 = float(latest.loc[sym, "avg_dollar_vol_20d"])
            if adv20 <= 0 or portfolio_value <= 0:
                target_weights[sym] = 0.0
                continue
            max_notional = params.adv_pct * adv20
            max_weight = max_notional / portfolio_value
            target_weights[sym] = float(
                np.sign(target_weights[sym]) * min(abs(target_weights[sym]), max_weight, params.max_position_weight)
            )

        # Holding period + stop-loss enforcement for open positions
        lc_state = lc_state or {}
        open_positions = (lc_state.get("open_positions") or {}).copy()
        updated_open_positions = open_positions.copy()

        latest_close = latest["close"].to_dict() if "close" in latest.columns else {}
        latest_atr = latest["atr_14"].to_dict() if "atr_14" in latest.columns else {}

        for sym, pos in open_positions.items():
            entry_date = pos.get("entry_date")
            if not entry_date:
                continue
            try:
                held_days = max(0, (as_of_date - pd.Timestamp(entry_date).date()).days)
            except Exception:
                held_days = 0
            must_exit = held_days >= params.hold_days

            if params.enable_stop_loss and not must_exit and sym in latest_close:
                entry_price = float(pos.get("entry_price", latest_close[sym]))
                atr = float(pos.get("entry_atr", latest_atr.get(sym, 0.0)) or 0.0)
                stop_move = params.stop_atr_mult * atr
                side = pos.get("side", "long")
                px = float(latest_close[sym])
                if side == "long" and px <= entry_price - stop_move:
                    must_exit = True
                if side == "short" and px >= entry_price + stop_move:
                    must_exit = True

            if must_exit:
                target_weights[sym] = 0.0
                updated_open_positions.pop(sym, None)
            else:
                if sym in current_weights and sym not in target_weights:
                    target_weights[sym] = float(current_weights[sym])

        # Register newly targeted positions in state
        for sym, w in target_weights.items():
            if abs(w) < 1e-8:
                continue
            if sym not in updated_open_positions:
                updated_open_positions[sym] = {
                    "entry_date": as_of_date.isoformat(),
                    "entry_price": float(latest_close.get(sym, 0.0)),
                    "entry_atr": float(latest_atr.get(sym, 0.0) or 0.0),
                    "side": "long" if w > 0 else "short",
                }

        # Drop tiny weights
        target_weights = {s: float(w) for s, w in target_weights.items() if abs(float(w)) > 1e-6}

        metadata = {
            **meta,
            "bearish_regime": bearish,
            "gross_exposure_target": float(gross),
            "strategy": "lc_reversal",
            "long_symbols": list(long_sel.index),
            "short_symbols": list(short_sel.index),
        }

        next_state = {
            "open_positions": updated_open_positions,
            "last_trade_date": as_of_date.isoformat(),
            "last_metadata": metadata,
        }

        return target_weights, metadata, next_state

    def run_backtest(
        self,
        df_features: pd.DataFrame,
        start_date: date,
        end_date: date,
        initial_capital: float = 100000.0,
        tx_cost_bps: float = 20.0,
        impact_slippage_mult: float = 0.0,
    ) -> Dict[str, Any]:
        """Lightweight no-lookahead daily rebalancing backtest for LC-Reversal."""
        data = df_features.sort_index().copy()
        start_date = pd.Timestamp(start_date).date()
        end_date = pd.Timestamp(end_date).date()

        def _to_date(value: Any) -> date:
            return pd.Timestamp(value).date()

        all_dates = sorted({_to_date(ts) for ts in data.index.get_level_values(0)})
        trade_dates = [d for d in all_dates if start_date <= d <= end_date]
        if len(trade_dates) < 10:
            return {}

        current_weights: Dict[str, float] = {}
        current_positions: Dict[str, float] = {}
        lc_state: Dict[str, Any] = {"open_positions": {}}

        equity = float(initial_capital)
        equity_curve: List[float] = [equity]
        turnover_series: List[float] = []
        daily_rets: List[float] = []
        wins = 0
        losses = 0

        for i in range(len(trade_dates) - 1):
            d = trade_dates[i]
            d_next = trade_dates[i + 1]
            if i % 25 == 0:
                logger.info(f"LC backtest progress: {i + 1}/{len(trade_dates) - 1} days ({d} -> {d_next})")

            history = data[data.index.get_level_values(0).date <= d]
            tw, meta, lc_state = self.build_targets(
                history,
                d,
                equity,
                current_positions=current_positions,
                current_weights=current_weights,
                lc_state=lc_state,
            )

            # Transaction costs from turnover
            all_syms = set(current_weights.keys()) | set(tw.keys())
            turnover = sum(abs(tw.get(s, 0.0) - current_weights.get(s, 0.0)) for s in all_syms)
            turnover_series.append(turnover)
            cost = turnover * (tx_cost_bps / 10000.0)

            # Next-day realized return
            pnl_ret = 0.0
            for s, w in tw.items():
                try:
                    sym = data.xs(s, level=1)
                    p_t = float(sym[sym.index.date == d]["close"].iloc[-1])
                    p_n = float(sym[sym.index.date == d_next]["close"].iloc[-1])
                    r = (p_n / p_t) - 1.0
                    slip = 0.0
                    if impact_slippage_mult > 0:
                        try:
                            imp = float(history.xs(s, level=1).tail(1)["impact_z"].iloc[-1])
                            slip = impact_slippage_mult * max(0.0, imp) / 10000.0
                        except Exception:
                            slip = 0.0
                    pnl_ret += w * (r - slip)
                except Exception:
                    continue

            net_ret = pnl_ret - cost
            if net_ret >= 0:
                wins += 1
            else:
                losses += 1
            daily_rets.append(net_ret)

            equity *= (1.0 + net_ret)
            equity_curve.append(equity)
            current_weights = tw

        if not daily_rets:
            return {}

        rets = np.array(daily_rets, dtype=float)
        years = max(len(rets) / 252.0, 1e-6)
        total_ret = float(equity_curve[-1] / equity_curve[0] - 1.0)
        cagr = float((1.0 + total_ret) ** (1.0 / years) - 1.0)
        vol = float(np.std(rets))
        sharpe = float((np.mean(rets) / vol) * np.sqrt(252)) if vol > 0 else 0.0

        eq_np = np.array(equity_curve, dtype=float)
        peak = np.maximum.accumulate(eq_np)
        drawdown = (eq_np - peak) / peak
        max_dd = float(np.min(drawdown)) if len(drawdown) else 0.0

        hit_rate = float(wins / max(1, wins + losses))
        avg_turnover = float(np.mean(turnover_series)) if turnover_series else 0.0

        return {
            "cagr": cagr,
            "sharpe": sharpe,
            "max_drawdown": max_dd,
            "turnover": avg_turnover,
            "hit_rate": hit_rate,
            "total_return": total_ret,
            "equity_curve": equity_curve,
            "dates": [str(d) for d in trade_dates],
        }
