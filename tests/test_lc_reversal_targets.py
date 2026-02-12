from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategies.lc_reversal import LCReversalStrategy


def _make_features_frame() -> pd.DataFrame:
    dates = pd.bdate_range("2025-01-01", periods=70)
    symbols = ["L1", "L2", "S1", "S2", "SPY"]
    rows: list[dict] = []
    for d in dates:
        for s in symbols:
            rows.append(
                {
                    "timestamp": d,
                    "symbol": s,
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.0 if s != "SPY" else (100.0 + (d.day % 5)),
                    "volume": 4_000_000.0,
                }
            )
    raw = pd.DataFrame(rows).set_index(["timestamp", "symbol"]).sort_index()
    st = LCReversalStrategy({"lc_reversal": {}})
    feats = st.compute_features(raw)

    last_ts = feats.index.get_level_values(0).max()
    latest = feats.loc[(last_ts, slice(None)), :].copy()
    latest.loc[(last_ts, "L1"), ["ret_1d", "vol_z", "impact_z"]] = [-0.10, 2.0, 2.0]
    latest.loc[(last_ts, "L2"), ["ret_1d", "vol_z", "impact_z"]] = [-0.08, 1.8, 1.7]
    latest.loc[(last_ts, "S1"), ["ret_1d", "vol_z", "impact_z"]] = [0.09, 2.1, 2.1]
    latest.loc[(last_ts, "S2"), ["ret_1d", "vol_z", "impact_z"]] = [0.07, 1.9, 1.9]
    latest.loc[(last_ts, "L1"), "avg_dollar_vol_20d"] = 9_000_000
    latest.loc[(last_ts, "L2"), "avg_dollar_vol_20d"] = 9_000_000
    latest.loc[(last_ts, "S1"), "avg_dollar_vol_20d"] = 9_000_000
    latest.loc[(last_ts, "S2"), "avg_dollar_vol_20d"] = 9_000_000
    latest.loc[(last_ts, "L1"), "volatility_20d"] = 0.02
    latest.loc[(last_ts, "L2"), "volatility_20d"] = 0.03
    latest.loc[(last_ts, "S1"), "volatility_20d"] = 0.02
    latest.loc[(last_ts, "S2"), "volatility_20d"] = 0.03
    feats.update(latest)
    return feats


def test_target_weights_obey_position_cap_and_neutrality() -> None:
    cfg = {
        "lc_reversal": {
            "n_universe": 5,
            "pct_tail": 0.4,
            "vol_z_min": 1.0,
            "impact_z_min": 1.0,
            "n_long": 2,
            "n_short": 2,
            "enable_shorts": True,
            "gross_exposure": 1.0,
            "weight_method": "equal_weight",
            "adv_pct": 1.0,
            "max_position_weight": 0.03,
            "hold_days": 2,
        }
    }
    strategy = LCReversalStrategy(cfg)
    feats = _make_features_frame()
    as_of = pd.Timestamp(feats.index.get_level_values(0).max()).date()

    tw, _meta, _state = strategy.build_targets(
        df_features=feats,
        as_of_date=as_of,
        portfolio_value=100_000.0,
        current_positions={},
        current_weights={},
        lc_state={},
    )

    assert tw, "Expected non-empty target weights"
    assert all(abs(w) <= 0.03 + 1e-12 for w in tw.values())
    assert abs(sum(tw.values())) < 1e-6


def test_hold_days_forces_exit() -> None:
    cfg = {
        "lc_reversal": {
            "n_universe": 5,
            "pct_tail": 0.4,
            "vol_z_min": 1.0,
            "impact_z_min": 1.0,
            "n_long": 2,
            "n_short": 2,
            "enable_shorts": True,
            "gross_exposure": 1.0,
            "max_position_weight": 0.03,
            "hold_days": 2,
        }
    }
    strategy = LCReversalStrategy(cfg)
    feats = _make_features_frame()
    as_of = pd.Timestamp(feats.index.get_level_values(0).max()).date()
    old_entry = (pd.Timestamp(as_of) - pd.Timedelta(days=4)).date().isoformat()

    tw, _meta, next_state = strategy.build_targets(
        df_features=feats,
        as_of_date=as_of,
        portfolio_value=100_000.0,
        current_positions={"OLD": 10.0},
        current_weights={"OLD": 0.02},
        lc_state={"open_positions": {"OLD": {"entry_date": old_entry, "entry_price": 100.0, "entry_atr": 1.0, "side": "long"}}},
    )

    assert tw.get("OLD", 0.0) == 0.0
    assert "OLD" not in (next_state.get("open_positions") or {})

