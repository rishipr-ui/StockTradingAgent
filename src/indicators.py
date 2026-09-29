"""Technical indicators calculated only from the data supplied by the caller."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def calculate_indicators(visible_data: pd.DataFrame) -> dict[str, Any]:
    """Calculate the latest RSI, MACD, moving averages, volume, and volatility.

    ``visible_data`` is expected to contain ``Close`` and ``Volume`` columns
    ordered chronologically. Rolling calculations use only rows in this frame;
    callers should pass the simulator's visible data rather than a full series.
    Values that need more history than is available are returned as ``None``.
    ``volatility_20`` is the rolling standard deviation of daily percentage
    returns, expressed as a decimal rather than annualized.
    """
    if visible_data.empty:
        raise ValueError("visible_data must contain at least one row")
    required = {"Close", "Volume"}
    missing = required.difference(visible_data.columns)
    if missing:
        raise ValueError(f"visible_data is missing columns: {sorted(missing)}")

    close = pd.to_numeric(visible_data["Close"], errors="coerce")
    volume = pd.to_numeric(visible_data["Volume"], errors="coerce")
    if close.isna().any() or volume.isna().any():
        raise ValueError("Close and Volume must contain only numeric values")

    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    average_gain = gains.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    average_loss = losses.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    relative_strength = average_gain / average_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + relative_strength))
    rsi = rsi.mask((average_loss == 0) & (average_gain > 0), 100)

    ema12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    macd_histogram = macd - macd_signal

    sma20 = close.rolling(20, min_periods=20).mean()
    sma50 = close.rolling(50, min_periods=50).mean()
    average_volume20 = volume.rolling(20, min_periods=20).mean()
    volume_spike_ratio = volume / average_volume20.replace(0, np.nan)
    volatility20 = close.pct_change().rolling(20, min_periods=20).std()

    latest_price = float(close.iloc[-1])
    latest = {
        "rsi": _latest(rsi),
        "macd": _latest(macd),
        "macd_signal": _latest(macd_signal),
        "macd_histogram": _latest(macd_histogram),
        "sma20": _latest(sma20),
        "sma50": _latest(sma50),
        "volume_spike_ratio": _latest(volume_spike_ratio),
        "volatility20": _latest(volatility20),
    }
    latest["summary"] = _summary(latest, latest_price)
    return latest


def _latest(series: pd.Series) -> float | None:
    """Convert a latest series value to a JSON-friendly float or ``None``."""
    value = series.iloc[-1]
    return None if pd.isna(value) else float(value)


def _summary(values: dict[str, float | None], price: float) -> str:
    """Build a concise human-readable description of the latest indicators."""
    parts: list[str] = []
    rsi = values["rsi"]
    if rsi is None:
        parts.append("RSI unavailable")
    elif rsi < 30:
        parts.append(f"RSI {rsi:.0f} (oversold)")
    elif rsi > 70:
        parts.append(f"RSI {rsi:.0f} (overbought)")
    else:
        parts.append(f"RSI {rsi:.0f} (neutral)")

    sma50 = values["sma50"]
    if sma50 is not None:
        parts.append(f"price {'below' if price < sma50 else 'above'} SMA50")
    sma20 = values["sma20"]
    if sma20 is not None and sma50 is not None:
        parts.append(f"SMA20 {'above' if sma20 >= sma50 else 'below'} SMA50")

    volume_ratio = values["volume_spike_ratio"]
    if volume_ratio is not None:
        parts.append(f"volume {volume_ratio:.1f}x average")
    return ", ".join(parts)
