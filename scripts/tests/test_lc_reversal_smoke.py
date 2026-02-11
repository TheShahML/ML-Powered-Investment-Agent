#!/usr/bin/env python3
"""Smoke test for LC-Reversal feature generation and target selection."""

import os
import sys
from datetime import date

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.strategies.lc_reversal import LCReversalStrategy


def make_synthetic_data(n_days: int = 120, n_symbols: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.bdate_range(end=pd.Timestamp("2026-01-30"), periods=n_days)
    symbols = [f"S{i:03d}" for i in range(n_symbols)] + ["SPY"]

    rows = []
    for sym in symbols:
        base = 100.0 + rng.normal(0, 5)
        prices = [base]
        for _ in range(1, len(dates)):
            prices.append(prices[-1] * (1.0 + rng.normal(0, 0.015)))
        prices = np.array(prices)
        volume = rng.integers(2_000_000, 10_000_000, size=len(dates))

        for d, p, v in zip(dates, prices, volume):
            rows.append(
                {
                    "timestamp": d,
                    "symbol": sym,
                    "open": p * (1 + rng.normal(0, 0.002)),
                    "high": p * (1 + abs(rng.normal(0, 0.01))),
                    "low": p * (1 - abs(rng.normal(0, 0.01))),
                    "close": p,
                    "volume": float(v),
                }
            )

    df = pd.DataFrame(rows).set_index(["timestamp", "symbol"]).sort_index()
    return df


def main() -> None:
    config = {
        "lc_reversal": {
            "n_universe": 25,
            "pct_tail": 0.2,
            "vol_z_min": -10.0,
            "impact_z_min": -10.0,
            "n_long": 5,
            "n_short": 5,
            "enable_shorts": True,
            "gross_exposure": 1.0,
            "weight_method": "equal_weight",
            "adv_pct": 0.02,
            "hold_days": 2,
            "enable_stop_loss": False,
            "bear_gross_mult": 0.5,
        }
    }

    strategy = LCReversalStrategy(config)
    raw = make_synthetic_data()
    feats = strategy.compute_features(raw)

    for col in ["ret_1d", "dollar_vol_1d", "vol_z", "impact", "impact_z", "atr_14", "avg_dollar_vol_20d"]:
        assert col in feats.columns, f"Missing feature column: {col}"

    signals = strategy.compute_signals(feats)
    assert isinstance(signals, pd.DataFrame)

    as_of = date(2026, 1, 30)
    tw, meta, state = strategy.build_targets(
        df_features=feats,
        as_of_date=as_of,
        portfolio_value=100000.0,
        current_positions={},
        current_weights={},
        lc_state={},
    )

    assert isinstance(tw, dict)
    assert "universe_size" in meta
    assert isinstance(state.get("open_positions", {}), dict)

    print("LC-Reversal smoke test passed")


if __name__ == "__main__":
    main()
