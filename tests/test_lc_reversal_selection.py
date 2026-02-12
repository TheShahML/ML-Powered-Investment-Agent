from __future__ import annotations

import pandas as pd

from src.strategies.lc_reversal import LCReversalStrategy


def _strategy(config_overrides: dict | None = None) -> LCReversalStrategy:
    cfg = {
        "lc_reversal": {
            "n_universe": 6,
            "pct_tail": 0.5,
            "vol_z_min": 1.0,
            "impact_z_min": 1.0,
            "n_long": 2,
            "n_short": 2,
            "enable_shorts": True,
            "weight_method": "equal_weight",
            "gross_exposure": 1.0,
            "max_position_weight": 0.03,
        }
    }
    if config_overrides:
        cfg["lc_reversal"].update(config_overrides)
    return LCReversalStrategy(cfg)


def _latest_frame() -> pd.DataFrame:
    idx = ["A", "B", "C", "D", "E", "F"]
    return pd.DataFrame(
        {
            "ret_1d": [-0.09, -0.03, -0.01, 0.01, 0.04, 0.08],
            "vol_z": [1.5, 1.2, 1.3, 1.4, 1.8, 1.1],
            "impact_z": [1.6, 1.4, 1.2, 1.1, 1.7, 1.5],
            "avg_dollar_vol_20d": [10_000_000] * 6,
            "dollar_vol_1d": [9_000_000] * 6,
            "volatility_20d": [0.02, 0.03, 0.04, 0.05, 0.03, 0.02],
            "close": [100.0] * 6,
            "atr_14": [2.0] * 6,
        },
        index=idx,
    )


def test_tail_selection_and_ordering() -> None:
    s = _strategy()
    latest = _latest_frame()
    long_sel, short_sel, meta = s._select_candidates(latest, as_of_date=pd.Timestamp("2026-01-30").date())

    assert list(long_sel.index) == ["A", "B"]
    assert list(short_sel.index) == ["F", "E"]
    assert meta["long_selected"] == 2
    assert meta["short_selected"] == 2


def test_insufficient_candidates_is_handled() -> None:
    s = _strategy({"vol_z_min": 99.0, "impact_z_min": 99.0})
    latest = _latest_frame()
    long_sel, short_sel, meta = s._select_candidates(latest, as_of_date=pd.Timestamp("2026-01-30").date())

    assert long_sel.empty
    assert short_sel.empty
    assert meta["long_selected"] == 0
    assert meta["short_selected"] == 0

