"""Walk-forward backtesting orchestration for the bull-bear-desk agents."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Iterable

import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

from . import agents
from .indicators import calculate_indicators
from .memory import HindsightMemory, NoMemory
from .prompts import LESSON_PROMPT
from .schemas import JudgeDecision, TradeLesson
from .simulator import MarketSim

DEFAULT_TICKERS = [
    "RELIANCE.NS",
    "TCS.NS",
    "INFY.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
]
TICKER_SECTOR = {
    "RELIANCE.NS": "energy",
    "TCS.NS": "it",
    "INFY.NS": "it",
    "HDFCBANK.NS": "banking",
    "ICICIBANK.NS": "banking",
}


def _json_default(value: Any) -> Any:
    """Serialize pandas and Pydantic values for JSONL and cache files."""
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if hasattr(value, "model_dump"):
        return value.model_dump()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


class LLMCache:
    """Small persistent JSON cache keyed by the full prompt inputs."""

    def __init__(self, path: str | Path = "results/llm_cache.json") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.values = json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            self.values = {}

    def get_or_call(self, namespace: str, payload: dict[str, Any], fn: Callable[[], Any]) -> Any:
        """Return a cached value or call, persist, and return the producer result."""
        key_text = json.dumps({"namespace": namespace, "payload": payload}, sort_keys=True)
        key = hashlib.sha256(key_text.encode("utf-8")).hexdigest()
        if key in self.values:
            return self.values[key]
        value = fn()
        self.values[key] = value.model_dump() if hasattr(value, "model_dump") else value
        self.path.write_text(json.dumps(self.values, default=_json_default), encoding="utf-8")
        return value


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=8),
    reraise=True,
)
def _rate_limited_call(fn: Callable[[], Any]) -> Any:
    """Call an LLM operation with a short rate-limit pause and exponential retry."""
    time.sleep(float(os.getenv("GROQ_MIN_INTERVAL_SECONDS", "0.25")))
    return fn()


def _cached_call(
    cache: LLMCache,
    namespace: str,
    payload: dict[str, Any],
    fn: Callable[[], Any],
) -> Any:
    """Run an LLM operation through the persistent cache and backoff wrapper."""
    return cache.get_or_call(namespace, payload, lambda: _rate_limited_call(fn))


def _tiny_indicator_summary(
    indicators: dict[str, Any],
    price: float,
) -> str:
    """Keep only RSI, price-vs-SMA50, and volume for tiny-mode prompts."""
    rsi = indicators.get("rsi")
    rsi_text = "unavailable" if rsi is None else f"{float(rsi):.0f}"
    sma50 = indicators.get("sma50")
    if sma50 is None:
        price_text = "unavailable vs SMA50"
    else:
        price_text = "below SMA50" if price < float(sma50) else "above SMA50"
    volume = indicators.get("volume_spike_ratio")
    volume_text = "unavailable" if volume is None else f"{float(volume):.1f}x average"
    return f"RSI {rsi_text}, price {price_text}, volume {volume_text}"


def _groq_call_with_usage(
    label: str,
    fn: Callable[[], Any],
    token_total: list[int],
) -> Any:
    """Run a Groq call and print its reported and cumulative token usage."""
    value = fn()
    usage = agents.get_last_usage_total_tokens()
    if usage is None:
        print(f"Groq {label}: usage unavailable (cumulative {token_total[0]})")
    else:
        token_total[0] += usage
        print(f"Groq {label}: {usage} tokens (cumulative {token_total[0]})")
    return value


def _normalise_setup(value: str, indicators_summary: str) -> str:
    """Map an agent label or indicator text to the lesson schema vocabulary."""
    allowed = {"oversold_bounce", "breakout", "breakdown", "mean_reversion"}
    if value in allowed:
        return value
    text = indicators_summary.lower()
    if "oversold" in text:
        return "oversold_bounce"
    if "below sma" in text:
        return "breakdown"
    if "above sma" in text:
        return "breakout"
    return "mean_reversion"


def _normalise_regime(value: str, indicators_summary: str) -> str:
    """Map free-text regime descriptions to the lesson schema vocabulary."""
    allowed = {"trending_up", "trending_down", "choppy", "high_vol"}
    if value in allowed:
        return value

    text = f"{value} {indicators_summary}".lower()
    if any(
        marker in text
        for marker in (
            "volatile",
            "volatility",
            "wide-range",
            "wide range",
            "high-vol",
            "high vol",
        )
    ):
        return "high_vol"
    if any(
        marker in text
        for marker in (
            "bullish",
            "bull",
            "uptrend",
            "upward",
            "rising",
            "positive trend",
        )
    ):
        return "trending_up"
    if any(
        marker in text
        for marker in (
            "bearish",
            "bear",
            "downtrend",
            "downward",
            "falling",
            "negative trend",
        )
    ):
        return "trending_down"
    if any(
        marker in text
        for marker in (
            "low-volume",
            "low volume",
            "thin liquidity",
            "thin",
            "quiet",
            "consolidation",
        )
    ):
        return "choppy"
    if "sma20 above sma50" in text:
        return "trending_up"
    if "sma20 below sma50" in text:
        return "trending_down"
    return "choppy"


def _outcome_lesson(
    outcome_pct: float,
    setup: str,
    market_regime: str,
    sector: str,
) -> tuple[str, str]:
    """Derive a structured fallback lesson without an LLM."""
    context = (
        f"When {setup} appears in a {market_regime} market for {sector}, "
    )
    if outcome_pct > 0:
        return (
            "bull",
            f"{context}bull was right because the outcome was positive. "
            "Next time: consider the bullish setup with confirmation.",
        )
    if outcome_pct < 0:
        return (
            "bear",
            f"{context}bear was right because the outcome was negative. "
            "Next time: wait for confirmation or avoid the trade.",
        )
    return (
        "neither",
        f"{context}neither was right because the outcome was flat. "
        "Next time: wait for a clearer edge.",
    )


def _lesson_from_outcome(
    outcome_pct: float,
    setup: str,
    market_regime: str,
    sector: str,
    cache: LLMCache,
    *,
    max_tokens: int | None = None,
    token_total: list[int] | None = None,
) -> tuple[str, str]:
    """Use one outcome-constrained LLM call to write a reusable lesson."""
    prompt = LESSON_PROMPT.format(
        setup=setup,
        market_regime=market_regime,
        sector=sector,
        outcome_pct=outcome_pct,
    )
    fallback_winner, fallback_lesson = _outcome_lesson(
        outcome_pct, setup, market_regime, sector
    )
    try:
        lesson_payload = {
            "outcome_pct": round(outcome_pct, 8),
            "setup": setup,
            "market_regime": market_regime,
            "sector": sector,
        }
        raw = _cached_call(
            cache,
            "outcome_lesson",
            lesson_payload,
            lambda: _groq_call_with_usage(
                "lesson",
                lambda: agents._complete(
                    prompt,
                    json_mode=True,
                    max_tokens=max_tokens,
                ),
                token_total if token_total is not None else [0],
            ),
        )
        payload = json.loads(raw)
        winner = payload["winner_side"]
        lesson = str(payload["lesson"]).strip()
        if winner not in {"bull", "bear", "neither"} or not lesson:
            raise ValueError("Invalid outcome lesson response")
        return winner, lesson
    except Exception:
        return fallback_winner, fallback_lesson


def _equity_for_date(simulator: MarketSim) -> float:
    """Return the current mark-to-market portfolio equity."""
    return float(simulator._equity_at_current_prices())


def _date_range(simulator: MarketSim, start: str, end: str) -> list[pd.Timestamp]:
    """Select simulator dates within the inclusive requested backtest period."""
    start_date = pd.Timestamp(start)
    end_date = pd.Timestamp(end)
    return [date for date in simulator._dates if start_date <= date <= end_date]


def _memory_bank_name(run_name: str) -> str:
    """Share one Hindsight bank across a train/test run pair."""
    for suffix in ("_train", "-train", "_test", "-test"):
        if run_name.endswith(suffix):
            return run_name[: -len(suffix)]
    return run_name


def run_backtest(
    tickers: list[str] | None = None,
    start: str = "2024-01-01",
    end: str = "2024-06-30",
    memory_mode: str = "none",
    bank_name: str = "default",
    seed: int = 0,
    *,
    fast: bool = False,
    tiny: bool = False,
    tickers_override: list[str] | None = None,
    data: dict[str, pd.DataFrame] | None = None,
    results_dir: str | Path = "results",
) -> Path:
    """Run a deterministic walk-forward backtest and write JSONL events."""
    if memory_mode not in {"none", "hindsight"}:
        raise ValueError("memory_mode must be 'none' or 'hindsight'")
    if seed is not None:
        import random

        random.seed(seed)
    selected = list(tickers_override or tickers or DEFAULT_TICKERS)
    if tiny and tickers_override is None and tickers is None:
        selected = DEFAULT_TICKERS[:2]
    elif tiny and tickers_override is not None:
        selected = list(tickers_override)
    elif fast:
        selected = selected[:3]
    if data is not None:
        data = {ticker: data[ticker] for ticker in selected if ticker in data}
    data_start = (
        pd.Timestamp(start) - timedelta(days=120)
    ).strftime("%Y-%m-%d")
    simulator = MarketSim(
        data,
        tickers=selected if data is None else None,
        start=data_start,
        end=(pd.Timestamp(end) + timedelta(days=1)).strftime("%Y-%m-%d"),
    )
    dates = _date_range(simulator, start, end)
    if not dates:
        raise ValueError("No trading dates fall within the requested period")
    simulator.current_date = dates[0]
    simulator.equity_curve = []
    simulator._record_equity()

    memory = (
        HindsightMemory(mode="hindsight", run_id=_memory_bank_name(bank_name))
        if memory_mode == "hindsight"
        else NoMemory()
    )
    cache = LLMCache(Path(results_dir) / "llm_cache.json")
    output_path = Path(results_dir) / f"{bank_name}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if tiny:
        day_dates = dates[::5][:15]
    elif fast:
        day_dates = dates[::2]
    else:
        day_dates = dates
    token_total = [0]

    with output_path.open("a", encoding="utf-8") as output:
        for current_date in day_dates:
            simulator.current_date = current_date
            for ticker in selected:
                visible = simulator.get_visible_data(ticker)
                indicators = calculate_indicators(visible)
                price = float(visible.iloc[-1]["Close"])
                summary = (
                    _tiny_indicator_summary(indicators, price)
                    if tiny
                    else str(indicators["summary"])
                )
                payload = {
                    "ticker": ticker,
                    "date": current_date.isoformat(),
                    "indicators_summary": summary,
                    "recent_headlines": [],
                    "tiny": tiny,
                }
                recalled = (
                    memory.recall_similar(
                        ticker, "unknown", _normalise_setup("", summary),
                        _normalise_regime("", summary), summary, k=5
                    )
                    if memory_mode == "hindsight"
                    else []
                )
                track_record = (
                    memory.recall_agent_track_record(
                        _normalise_setup("", summary), _normalise_regime("", summary)
                    )
                    if memory_mode == "hindsight"
                    else []
                )
                bull = _cached_call(
                    cache,
                    "bull",
                    payload,
                    lambda: _groq_call_with_usage(
                        "bull",
                        lambda: agents.bull_argue(
                            ticker,
                            summary,
                            [],
                            **(
                                {"max_tokens": 80, "argument_word_limit": 40}
                                if tiny
                                else {}
                            ),
                        ),
                        token_total,
                    ),
                )
                bear = _cached_call(
                    cache,
                    "bear",
                    payload,
                    lambda: _groq_call_with_usage(
                        "bear",
                        lambda: agents.bear_argue(
                            ticker,
                            summary,
                            [],
                            **(
                                {"max_tokens": 80, "argument_word_limit": 40}
                                if tiny
                                else {}
                            ),
                        ),
                        token_total,
                    ),
                )
                judge_payload = {
                    **payload,
                    "bull_text": bull,
                    "bear_text": bear,
                    "recalled_lessons": recalled,
                    "agent_track_record": track_record,
                }
                decision = _cached_call(
                    cache,
                    "judge_v4",
                    judge_payload,
                    lambda: _groq_call_with_usage(
                        "judge",
                        lambda: agents.judge_decide(
                            ticker,
                            summary,
                            bull,
                            bear,
                            recalled,
                            track_record,
                            **({"max_tokens": 400} if tiny else {}),
                        ),
                        token_total,
                    ),
                )
                if isinstance(decision, dict):
                    decision = JudgeDecision.model_validate(decision)
                try:
                    trade = simulator.place_order(
                        ticker,
                        decision.action,
                        min(1.0, max(0.0, decision.size_pct / 10)),
                    )
                except RuntimeError:
                    # A drawdown halt is a valid simulator outcome; record a
                    # HOLD rather than interrupting the remaining log rows.
                    trade = simulator.place_order(ticker, "HOLD", 0.0)
                try:
                    outcome = simulator.resolve_trade(trade, horizon_days=5)
                except ValueError:
                    outcome = None
                winner_side = None
                lesson_text = None
                if outcome is not None:
                    setup = _normalise_setup(decision.setup, summary)
                    market_regime = _normalise_regime(decision.market_regime, summary)
                    sector = TICKER_SECTOR.get(ticker, "diversified")
                    if tiny:
                        winner_side, lesson_text = _lesson_from_outcome(
                            outcome,
                            setup,
                            market_regime,
                            sector,
                            cache,
                            max_tokens=80,
                            token_total=token_total,
                        )
                    else:
                        winner_side, lesson_text = _lesson_from_outcome(
                            outcome, setup, market_regime, sector, cache
                        )
                    if memory_mode == "hindsight":
                        lesson = TradeLesson(
                            ticker=ticker,
                            sector=sector,
                            date=current_date.date().isoformat(),
                            setup=setup,
                            market_regime=market_regime,
                            indicators_summary=summary,
                            bull_argument=bull,
                            bear_argument=bear,
                            judge_decision=decision.action,
                            winner_side=winner_side,
                            outcome_pct=outcome,
                            lesson=lesson_text,
                        )
                        memory.retain_lesson(lesson)
                record = {
                    "date": current_date.date().isoformat(),
                    "ticker": ticker,
                    "price": float(visible.iloc[-1]["Close"]),
                    "indicators": indicators,
                    "bull_text": bull,
                    "bear_text": bear,
                    "decision": decision.model_dump(),
                    "recalled_memories": recalled,
                    "agent_track_record": track_record,
                    "outcome": outcome,
                    "winner_side": winner_side,
                    "lesson": lesson_text,
                    "equity": _equity_for_date(simulator),
                }
                output.write(json.dumps(record, default=_json_default) + "\n")
                output.flush()
                print(
                    f"{current_date.date()} {ticker}: "
                    f"{decision.action} (conf {decision.confidence:.2f})"
                )
            if current_date != day_dates[-1]:
                simulator.step_to_next_day()
    print(f"Groq cumulative total: {token_total[0]} tokens")
    return output_path


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    """Parse the command-line backtest options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("none", "hindsight"), required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument(
        "--tickers",
        help="Comma-separated ticker override, e.g. RELIANCE.NS,TCS.NS",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    """Run a backtest from CLI arguments."""
    args = _parse_args(argv)
    run_backtest(
        tickers_override=(
            [ticker.strip() for ticker in args.tickers.split(",") if ticker.strip()]
            if args.tickers
            else None
        ),
        start=args.start,
        end=args.end,
        memory_mode=args.mode,
        bank_name=args.run,
        seed=args.seed,
        fast=args.fast,
        tiny=args.tiny,
    )


if __name__ == "__main__":
    main()
