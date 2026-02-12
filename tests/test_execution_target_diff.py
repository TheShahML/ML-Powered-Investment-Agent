from __future__ import annotations

from scripts.execute_rebalance_safe import _compute_target_diff


def test_compute_target_diff_counts_and_top_changes() -> None:
    prev = {"A": 0.02, "B": -0.02, "C": 0.01}
    curr = {"A": 0.03, "B": -0.01, "D": 0.02}

    d = _compute_target_diff(prev, curr)

    assert d["num_added"] == 1
    assert d["num_removed"] == 1
    assert "D" in d["added_symbols"]
    assert "C" in d["removed_symbols"]
    assert d["changed_weights_top"][0]["symbol"] in {"C", "D", "A", "B"}

