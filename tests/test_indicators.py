"""Tests for technical indicator calculations and summaries."""

import pandas as pd
import pytest

from src.indicators import calculate_indicators


def test_calculates_latest_indicators():
    close = pd.Series(range(1, 61), dtype=float)
    data = pd.DataFrame({"Close": close, "Volume": [100.0] * 59 + [250.0]})

    result = calculate_indicators(data)

    assert result["rsi"] == pytest.approx(100.0)
    assert result["sma20"] == pytest.approx(50.5)
    assert result["sma50"] == pytest.approx(35.5)
    assert result["volume_spike_ratio"] == pytest.approx(250 / 107.5)
    assert result["macd"] is not None
    assert result["macd_signal"] is not None
    expected_volatility = close.pct_change().rolling(20).std().iloc[-1]
    assert result["volatility20"] == pytest.approx(expected_volatility)


def test_summary_describes_oversold_price_and_volume():
    close = [100.0] * 50 + [80.0] * 10
    volume = [100.0] * 59 + [210.0]
    result = calculate_indicators(pd.DataFrame({"Close": close, "Volume": volume}))

    assert "RSI" in result["summary"]
    assert "oversold" in result["summary"]
    assert "price below SMA50" in result["summary"]
    assert "volume" in result["summary"]


def test_insufficient_history_returns_none_for_long_windows():
    data = pd.DataFrame(
        {"Close": [100.0, 101.0, 99.0], "Volume": [100.0, 110.0, 90.0]}
    )

    result = calculate_indicators(data)

    assert result["rsi"] is None
    assert result["sma20"] is None
    assert result["sma50"] is None
    assert result["volume_spike_ratio"] is None
    assert result["volatility20"] is None
    assert result["summary"].startswith("RSI unavailable")
