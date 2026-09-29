"""Market data loading and a constrained, date-stepped trading simulator."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping

import pandas as pd
import yfinance as yf

Action = Literal["BUY", "SELL", "HOLD"]


@dataclass
class Position:
    """A long position held by the simulator."""

    ticker: str
    shares: float
    entry_price: float
    entry_date: pd.Timestamp
    cost_basis: float


@dataclass
class Trade:
    """A decision and its executed trade details."""

    ticker: str
    action: Action
    decision_date: pd.Timestamp
    entry_price: float | None = None
    exit_price: float | None = None
    shares: float = 0.0
    realized_return_pct: float | None = None
    status: str = "OPEN"


def load_market_data(
    tickers: list[str] | None = None,
    data_dir: str | Path = "data",
    start: str | None = None,
    end: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Download daily OHLCV data and cache each ticker as CSV."""
    tickers = tickers or [
        "RELIANCE.NS",
        "TCS.NS",
        "INFY.NS",
        "HDFCBANK.NS",
        "ICICIBANK.NS",
    ]
    cache_dir = Path(data_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, pd.DataFrame] = {}

    for ticker in tickers:
        cache_path = cache_dir / f"{ticker.replace('/', '_')}.csv"
        if cache_path.exists():
            frame = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            if not isinstance(frame.index, pd.DatetimeIndex):
                cache_path.unlink()
                frame = _download_and_normalise(ticker, start, end)
                frame.to_csv(cache_path, index_label="Date", header=True)
        else:
            frame = _download_and_normalise(ticker, start, end)
            frame.to_csv(cache_path, index_label="Date", header=True)
        result[ticker] = _normalise_ohlcv(frame)
    return result


def _download_and_normalise(
    ticker: str,
    start: str | None,
    end: str | None,
) -> pd.DataFrame:
    """Download one ticker and flatten its columns before caching."""
    frame = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
    )
    if frame.empty:
        raise ValueError(f"No market data returned for {ticker}")
    return _normalise_ohlcv(frame)


def _normalise_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    """Return sorted OHLCV data with a timezone-naive DatetimeIndex."""
    data = frame.copy()
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data.columns = [str(column).title() for column in data.columns]
    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"OHLCV data is missing columns: {sorted(missing)}")
    index = pd.to_datetime(data.index, errors="coerce")
    if index.isna().any():
        raise ValueError("OHLCV data index could not be parsed as dates")
    if getattr(index, "tz", None) is not None:
        index = index.tz_localize(None)
    data.index = index.normalize()
    return data.sort_index()[["Open", "High", "Low", "Close", "Volume"]]


class MarketSim:
    """A long-only daily market simulator with bounded risk."""

    def __init__(
        self,
        data: Mapping[str, pd.DataFrame] | None = None,
        *,
        tickers: list[str] | None = None,
        cash: float = 100_000.0,
        transaction_cost: float = 0.001,
        data_dir: str | Path = "data",
        start: str | None = None,
        end: str | None = None,
        max_drawdown: float = 0.20,
    ) -> None:
        """Create a simulator, optionally using supplied data for testing."""
        self.data = {
            ticker: _normalise_ohlcv(frame)
            for ticker, frame in (data or load_market_data(tickers, data_dir, start, end)).items()
        }
        if not self.data:
            raise ValueError("At least one ticker with OHLCV data is required")
        self.cash = float(cash)
        self.initial_cash = float(cash)
        self.transaction_cost = float(transaction_cost)
        self.max_drawdown = float(max_drawdown)
        self.positions: dict[str, Position] = {}
        self.trades: list[Trade] = []
        self.equity_curve: list[dict[str, object]] = []
        self.halted = False
        self._peak_equity = float(cash)
        self._dates = sorted(set().union(*(set(frame.index) for frame in self.data.values())))
        self.current_date = self._dates[0]
        self._record_equity()

    def get_visible_data(self, ticker: str) -> pd.DataFrame:
        """Return only data available through the current simulation date."""
        if ticker not in self.data:
            raise KeyError(f"Unknown ticker: {ticker}")
        visible = self.data[ticker].loc[self.data[ticker].index <= self.current_date].copy()
        assert visible.index.max() <= self.current_date
        return visible

    def step_to_next_day(self) -> pd.Timestamp:
        """Advance one trading day, applying risk controls and recording equity."""
        position_date_index = self._dates.index(self.current_date)
        if position_date_index + 1 >= len(self._dates):
            raise StopIteration("The simulation has reached its final date")
        self.current_date = self._dates[position_date_index + 1]
        self._apply_stop_losses()
        self._record_equity()
        return self.current_date

    def place_order(self, ticker: str, action: Action, size_pct_of_cash: float) -> Trade:
        """Place a bounded order using only data visible on the current date."""
        if action not in {"BUY", "SELL", "HOLD"}:
            raise ValueError("action must be BUY, SELL, or HOLD")
        if not 0 <= size_pct_of_cash <= 1:
            raise ValueError("size_pct_of_cash must be between 0 and 1")
        if self.halted and action != "HOLD":
            raise RuntimeError("Trading is halted by the drawdown cap")
        row = self.get_visible_data(ticker).iloc[-1]
        price = float(row["Close"])
        trade = Trade(ticker=ticker, action=action, decision_date=self.current_date)
        if action == "BUY":
            equity = self._equity_at_current_prices()
            allocation = min(self.cash * size_pct_of_cash, equity * 0.10)
            shares = allocation / (price * (1 + self.transaction_cost))
            total_cost = shares * price * (1 + self.transaction_cost)
            if shares > 0 and total_cost <= self.cash:
                self.cash -= total_cost
                self.positions[ticker] = Position(
                    ticker, shares, price, self.current_date, total_cost
                )
                trade.entry_price, trade.shares = price, shares
        elif action == "SELL" and ticker in self.positions:
            position = self.positions.pop(ticker)
            proceeds = position.shares * price * (1 - self.transaction_cost)
            self.cash += proceeds
            trade.entry_price = position.entry_price
            trade.exit_price = price
            trade.shares = position.shares
            trade.realized_return_pct = (proceeds - position.cost_basis) / position.cost_basis * 100
            trade.status = "CLOSED"
        self.trades.append(trade)
        self._record_equity()
        return trade

    def resolve_trade(self, trade: Trade, horizon_days: int = 5) -> float:
        """Compute a trade's close-to-close return after its decision date."""
        if horizon_days < 1:
            raise ValueError("horizon_days must be positive")
        if trade.ticker not in self.data:
            raise KeyError(f"Unknown ticker: {trade.ticker}")
        future = self.data[trade.ticker].loc[self.data[trade.ticker].index > trade.decision_date]
        if len(future) < horizon_days:
            raise ValueError("Not enough future data to resolve the requested horizon")
        exit_price = float(future.iloc[horizon_days - 1]["Close"])
        entry_price = trade.entry_price or float(
            self.data[trade.ticker].loc[trade.decision_date, "Close"]
        )
        result = (exit_price / entry_price - 1) * 100
        trade.exit_price = exit_price
        trade.realized_return_pct = result
        trade.status = "RESOLVED"
        return result

    def _equity_at_current_prices(self) -> float:
        """Mark all positions to prices visible on the current date."""
        value = self.cash
        for ticker, position in self.positions.items():
            visible = self.get_visible_data(ticker)
            value += position.shares * float(visible.iloc[-1]["Close"])
        return value

    def _record_equity(self) -> None:
        """Record equity and halt new trades if the drawdown cap is breached."""
        equity = self._equity_at_current_prices()
        self._peak_equity = max(self._peak_equity, equity)
        drawdown = 1 - equity / self._peak_equity
        if drawdown >= self.max_drawdown:
            self.halted = True
        self.equity_curve.append(
            {"date": self.current_date, "equity": equity, "drawdown": drawdown}
        )

    def _apply_stop_losses(self) -> None:
        """Close positions whose visible close is at least 5% below entry."""
        for ticker, position in list(self.positions.items()):
            close = float(self.get_visible_data(ticker).iloc[-1]["Close"])
            if close <= position.entry_price * 0.95:
                self.place_order(ticker, "SELL", 1.0)
