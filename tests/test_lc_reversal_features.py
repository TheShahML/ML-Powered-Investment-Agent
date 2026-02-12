from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategies.lc_reversal import LCReversalStrategy


def _make_ohlcv(n_days: int = 120, symbols: list[str] | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    syms = symbols or ["AAA", "BBB", "CCC", "SPY"]
    dates = pd.bdate_range("2025-01-01", periods=n_days)
    rows: list[dict] = []

    for sym in syms:
        px = 100.0 + rng.normal(0, 1.0)
        for d in dates:
            ret = rng.normal(0.0005, 0.01)
            px = max(1.0, px * (1.0 + ret))
            high = px * (1.0 + abs(rng.normal(0, 0.01)))
            low = px * (1.0 - abs(rng.normal(0, 0.01)))
            rows.append(
                {
                    "timestamp": d,
                    "symbol": sym,
                    "open": px * (1.0 + rng.normal(0, 0.002)),
                    "high": high,
                    "low": low,
                    "close": px,
                    "volume": float(rng.integers(1_000_000, 8_000_000)),
                }
            )
    return pd.DataFrame(rows).set_index(["timestamp", "symbol"]).sort_index()


def test_lc_feature_columns_and_impact_safety() -> None:
    strategy = LCReversalStrategy(config={"lc_reversal": {}})
    raw = _make_ohlcv()
    feats = strategy.compute_features(raw)

    expected_cols = {
        "ret_1d",
        "dollar_vol_1d",
        "vol_z",
        "impact",
        "impact_z",
        "atr_14",
        "avg_dollar_vol_20d",
        "volatility_20d",
    }
    assert expected_cols.issubset(set(feats.columns))
    assert np.isfinite(feats["impact"].dropna().to_numpy()).all()


def test_ret_1d_formula_matches_pct_change() -> None:
    strategy = LCReversalStrategy(config={"lc_reversal": {}})
    raw = _make_ohlcv(symbols=["AAA", "SPY"])
    feats = strategy.compute_features(raw)

    aaa_raw = raw.xs("AAA", level=1)
    aaa_feats = feats.xs("AAA", level=1)
    t = aaa_feats.index[10]
    expected = aaa_raw["close"].loc[t] / aaa_raw["close"].shift(1).loc[t] - 1.0
    actual = aaa_feats["ret_1d"].loc[t]

    assert np.isclose(actual, expected, atol=1e-12)

