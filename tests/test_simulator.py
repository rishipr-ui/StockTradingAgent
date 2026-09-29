"""Tests for the date-bounded market simulator."""

import pandas as pd
import pytest

from src.simulator import MarketSim, load_market_data


@pytest.fixture
def market_data():
    dates = pd.date_range("2024-01-01", periods=8, freq="D")
    close = [100, 101, 102, 103, 104, 95, 94, 93]
    return {
        "TEST": pd.DataFrame(
            {
                "Open": close,
                "High": [price + 1 for price in close],
                "Low": [price - 1 for price in close],
                "Close": close,
                "Volume": [1000] * len(close),
            },
            index=dates,
        )
    }


def test_future_data_cannot_leak(market_data):
    sim = MarketSim(market_data, cash=10_000)
    visible = sim.get_visible_data("TEST")

    assert visible.index.max() == sim.current_date
    assert len(visible) == 1
    with pytest.raises(KeyError):
        visible.loc[pd.Timestamp("2024-01-02")]

    sim.step_to_next_day()
    assert sim.get_visible_data("TEST").index.max() == sim.current_date


def test_trade_size_is_capped_at_ten_percent_of_equity(market_data):
    sim = MarketSim(market_data, cash=10_000)
    trade = sim.place_order("TEST", "BUY", 1.0)

    assert trade.shares * trade.entry_price <= 1_000


def test_resolve_trade_uses_only_post_decision_data(market_data):
    sim = MarketSim(market_data, cash=10_000)
    trade = sim.place_order("TEST", "BUY", 0.5)

    assert sim.resolve_trade(trade, horizon_days=2) == pytest.approx((102 / 100 - 1) * 100)


def test_stop_loss_closes_position(market_data):
    sim = MarketSim(market_data, cash=10_000)
    sim.place_order("TEST", "BUY", 1.0)
    for _ in range(5):
        sim.step_to_next_day()

    assert "TEST" not in sim.positions
    assert any(trade.action == "SELL" and trade.status == "CLOSED" for trade in sim.trades)


def test_drawdown_cap_halts_trading(market_data):
    sim = MarketSim(market_data, cash=10_000, max_drawdown=0.0005)
    sim.place_order("TEST", "BUY", 1.0)
    for _ in range(5):
        sim.step_to_next_day()

    assert sim.halted
    with pytest.raises(RuntimeError):
        sim.place_order("TEST", "BUY", 0.1)


def test_market_data_cache_round_trip_parses_datetime_index(tmp_path, monkeypatch):
    dates = pd.date_range("2024-01-01", periods=2, freq="D")
    frame = pd.DataFrame(
        {
            "Open": [100, 101],
            "High": [101, 102],
            "Low": [99, 100],
            "Close": [100, 101],
            "Volume": [1000, 1100],
        },
        index=dates,
    )
    monkeypatch.setattr("src.simulator.yf.download", lambda *args, **kwargs: frame)

    load_market_data(["TEST"], data_dir=tmp_path)
    cached = load_market_data(["TEST"], data_dir=tmp_path)["TEST"]

    assert isinstance(cached.index, pd.DatetimeIndex)
    assert str(cached.index.dtype).startswith("datetime64[")


def test_market_data_handles_multiindex_columns(tmp_path, monkeypatch):
    dates = pd.date_range("2024-01-01", periods=2, freq="D")
    fields = ["Open", "High", "Low", "Close", "Volume"]
    columns = pd.MultiIndex.from_product([fields, ["TEST"]])
    frame = pd.DataFrame(
        [[100, 101, 99, 100, 1000], [101, 102, 100, 101, 1100]],
        index=dates,
        columns=columns,
    )
    monkeypatch.setattr("src.simulator.yf.download", lambda *args, **kwargs: frame)

    result = load_market_data(["TEST"], data_dir=tmp_path)
    cached_lines = (tmp_path / "TEST.csv").read_text(encoding="utf-8").splitlines()

    assert list(result["TEST"].columns) == fields
    assert isinstance(result["TEST"].index, pd.DatetimeIndex)
    assert cached_lines[0] == "Date,Open,High,Low,Close,Volume"
    assert not cached_lines[1].startswith("Ticker,")
