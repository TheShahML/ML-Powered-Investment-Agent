from src.execution_safe import execute_orders_safe


class _FakeTrade:
    def __init__(self, price: float):
        self.price = price


class _FakeApi:
    def __init__(self, price: float = 100.0):
        self._price = price

    def get_latest_trade(self, symbol, feed="iex"):
        return _FakeTrade(self._price)


def _base_config():
    return {
        "max_orders_per_rebalance": 50,
        "max_daily_notional": 1_000_000.0,
        "min_trade_notional": 1.0,
        "turnover_buffer_pct": 0.0,
        "order_type": "market",
    }


def test_fractional_short_is_skipped_not_dry_run():
    api = _FakeApi(price=100.0)
    orders = execute_orders_safe(
        api=api,
        target_weights={"XYZ": -0.015},
        current_positions={},
        portfolio_value=1_000.0,
        config=_base_config(),
        dry_run=True,
    )
    assert len(orders) == 1
    assert orders[0]["status"] == "skipped_infeasible_short_qty"
    assert "fractional short" in str(orders[0].get("error", "")).lower()


def test_short_qty_for_new_short_is_whole_share():
    api = _FakeApi(price=10.0)
    orders = execute_orders_safe(
        api=api,
        target_weights={"XYZ": -0.25},
        current_positions={},
        portfolio_value=1_000.0,
        config=_base_config(),
        dry_run=True,
    )
    assert len(orders) == 1
    assert orders[0]["status"] == "dry_run"
    assert float(orders[0]["qty"]).is_integer()
