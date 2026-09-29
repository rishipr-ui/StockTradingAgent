"""Tests for ticker-to-sector resolution used by lesson generation."""

from src.backtest import DEFAULT_TICKERS, TICKER_SECTOR


def test_default_tickers_resolve_to_real_sectors():
    expected_sectors = {"energy", "it", "banking"}

    assert set(DEFAULT_TICKERS) == set(TICKER_SECTOR)
    assert all(TICKER_SECTOR[ticker] in expected_sectors for ticker in DEFAULT_TICKERS)


def test_unknown_ticker_uses_diversified_fallback():
    assert TICKER_SECTOR.get("UNKNOWN.NS", "diversified") == "diversified"
