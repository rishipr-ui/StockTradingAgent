"""Run one offline-analysis-to-Alpaca-paper-trade decision for a US ticker.

This script never targets a live Alpaca account. It requires Alpaca paper
credentials in the project-root .env file and submits at most one order.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents import bear_argue, bull_argue, judge_decide
from src.indicators import calculate_indicators
from src.memory import HindsightMemory


def _clients() -> tuple[Any, Any]:
    """Create Alpaca clients with the paper-trading flag enforced."""
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    if not api_key or not secret_key:
        raise RuntimeError("ALPACA_API_KEY and ALPACA_SECRET_KEY are required")

    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.trading.client import TradingClient

    trading_client = TradingClient(api_key, secret_key, paper=True)
    data_client = StockHistoricalDataClient(api_key, secret_key)
    return trading_client, data_client


def _daily_data(data_client: Any, ticker: str) -> pd.DataFrame:
    """Fetch recent daily bars needed by the indicator calculations."""
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    end = datetime.now(timezone.utc)
    request = StockBarsRequest(
        symbol_or_symbols=ticker,
        timeframe=TimeFrame.Day,
        start=end - timedelta(days=120),
        end=end,
    )
    frame = data_client.get_stock_bars(request).df
    if frame.empty:
        raise RuntimeError(f"No daily Alpaca data returned for {ticker}")
    if isinstance(frame.index, pd.MultiIndex):
        frame = frame.xs(ticker, level="symbol")
    frame = frame.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
    )
    return frame[["Open", "High", "Low", "Close", "Volume"]].sort_index()


def _order_quantity(trading_client: Any, size_pct: float, price: float) -> int:
    """Convert the judge's 0-10 allocation into a conservative whole share count."""
    account = trading_client.get_account()
    equity = float(account.equity)
    allocation = equity * max(0.0, min(10.0, size_pct)) / 100
    return max(1, int(allocation // price))


def run(ticker: str = "AAPL") -> None:
    """Analyze one ticker, print the decision, and submit at most one paper order."""
    trading_client, data_client = _clients()
    daily = _daily_data(data_client, ticker)
    indicators = calculate_indicators(daily)
    summary = str(indicators["summary"])

    memory = HindsightMemory(mode="hindsight", run_id="memory")
    setup = "oversold_bounce" if "oversold" in summary.lower() else "mean_reversion"
    regime = "choppy"
    recalled = memory.recall_similar(
        ticker, "diversified", setup, regime, summary, k=5
    )
    track_record = memory.recall_agent_track_record(setup, regime)
    bull_text = bull_argue(ticker, summary, [])
    bear_text = bear_argue(ticker, summary, [])
    decision = judge_decide(
        ticker, summary, bull_text, bear_text, recalled, track_record
    )

    print(f"Ticker: {ticker}")
    print(f"Latest daily close: {float(daily.iloc[-1]['Close']):.2f}")
    print(f"Indicators: {summary}")
    print("\nRecalled memories:")
    if recalled:
        for memory_text in recalled:
            print(f"- {memory_text}")
    else:
        print("- no relevant memory")
    print("\nDecision:")
    print(decision.model_dump_json(indent=2))

    if decision.action == "HOLD":
        print("\nNo paper order submitted: judge decision is HOLD.")
        return

    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import MarketOrderRequest

    price = float(daily.iloc[-1]["Close"])
    quantity = _order_quantity(trading_client, decision.size_pct, price)
    side = OrderSide.BUY if decision.action == "BUY" else OrderSide.SELL
    order = trading_client.submit_order(
        MarketOrderRequest(
            symbol=ticker,
            qty=quantity,
            side=side,
            time_in_force=TimeInForce.DAY,
        )
    )
    print(f"\nSubmitted ONE Alpaca paper order: {side.value} {quantity} {ticker}")
    print(f"Order ID: {order.id}")


def main() -> None:
    """Parse the ticker and execute one paper-trading decision."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="AAPL", help="One US ticker symbol")
    args = parser.parse_args()
    run(args.ticker.upper())


if __name__ == "__main__":
    main()
