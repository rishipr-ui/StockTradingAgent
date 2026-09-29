"""Tests for backtest orchestration with mocked agent and memory services."""

import json

import pandas as pd

from src import backtest
from src.schemas import JudgeDecision


def test_normalise_regime_maps_free_text_to_allowed_values():
    assert backtest._normalise_regime(
        "bullish uptrend with thin liquidity", ""
    ) == "trending_up"
    assert backtest._normalise_regime("bearish reversal", "") == "trending_down"
    assert backtest._normalise_regime("low-volume consolidation", "") == "choppy"
    assert backtest._normalise_regime("wide-range volatile session", "") == "high_vol"
    assert backtest._normalise_regime("unclassified", "SMA20 above SMA50") == "trending_up"
    assert backtest._normalise_regime("unclassified", "") == "choppy"


def test_run_backtest_writes_jsonl_and_reuses_cache(tmp_path, monkeypatch):
    dates = pd.date_range("2024-01-01", periods=12, freq="D")
    close = [100 + index for index in range(12)]
    data = {
        "TEST": pd.DataFrame(
            {
                "Open": close,
                "High": [value + 1 for value in close],
                "Low": [value - 1 for value in close],
                "Close": close,
                "Volume": [1000] * 12,
            },
            index=dates,
        )
    }
    calls = {"bull": 0, "bear": 0, "judge": 0}

    def bull(*args):
        calls["bull"] += 1
        return "bull argument"

    def bear(*args):
        calls["bear"] += 1
        return "bear argument"

    def judge(*args):
        calls["judge"] += 1
        return JudgeDecision(
            action="BUY",
            size_pct=1,
            confidence=0.5,
            reasoning="test",
            memories_used=[],
            setup="mean_reversion",
            market_regime="choppy",
        )

    monkeypatch.setattr(backtest.agents, "bull_argue", bull)
    monkeypatch.setattr(backtest.agents, "bear_argue", bear)
    monkeypatch.setattr(backtest.agents, "judge_decide", judge)
    monkeypatch.setattr(
        backtest,
        "_lesson_from_outcome",
        lambda outcome, setup, regime, sector, cache: ("bull", "positive"),
    )

    output = backtest.run_backtest(
        ["TEST"],
        "2024-01-01",
        "2024-01-10",
        "none",
        "test-run",
        1,
        data=data,
        results_dir=tmp_path,
    )

    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == 10
    assert rows[0]["ticker"] == "TEST"
    assert "equity" in rows[0]
    assert calls == {"bull": 10, "bear": 10, "judge": 10}

    backtest.run_backtest(
        ["TEST"],
        "2024-01-01",
        "2024-01-10",
        "none",
        "test-run-cache",
        1,
        data=data,
        results_dir=tmp_path,
    )
    assert (tmp_path / "llm_cache.json").exists()


def test_fast_mode_uses_every_second_day(tmp_path, monkeypatch):
    dates = pd.date_range("2024-01-01", periods=8, freq="D")
    data = {
        "A": pd.DataFrame(
            {"Close": range(100, 108), "Open": range(100, 108),
             "High": range(101, 109), "Low": range(99, 107),
             "Volume": [1000] * 8},
            index=dates,
        )
    }
    monkeypatch.setattr(backtest.agents, "bull_argue", lambda *args: "bull")
    monkeypatch.setattr(backtest.agents, "bear_argue", lambda *args: "bear")
    monkeypatch.setattr(
        backtest.agents,
        "judge_decide",
        lambda *args: JudgeDecision(
            action="HOLD", size_pct=0, confidence=0, reasoning="test",
            memories_used=[], setup="mean_reversion", market_regime="choppy"
        ),
    )
    monkeypatch.setattr(
        backtest,
        "_lesson_from_outcome",
        lambda outcome, setup, regime, sector, cache: ("neither", "flat"),
    )

    output = backtest.run_backtest(
        ["A"], "2024-01-01", "2024-01-08", "none", "fast-run", 1,
        fast=True, data=data, results_dir=tmp_path
    )
    assert len(output.read_text().splitlines()) == 4


def test_tiny_mode_samples_five_days_and_caps_at_fifteen(tmp_path, monkeypatch):
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    close = list(range(100, 200))
    data = {
        ticker: pd.DataFrame(
            {
                "Open": close,
                "High": [value + 1 for value in close],
                "Low": [value - 1 for value in close],
                "Close": close,
                "Volume": [1000] * len(close),
            },
            index=dates,
        )
        for ticker in ["RELIANCE.NS", "TCS.NS"]
    }
    monkeypatch.setattr(backtest.agents, "bull_argue", lambda *args, **kwargs: "bull")
    monkeypatch.setattr(backtest.agents, "bear_argue", lambda *args, **kwargs: "bear")
    monkeypatch.setattr(
        backtest.agents,
        "judge_decide",
        lambda *args, **kwargs: JudgeDecision(
            action="HOLD",
            size_pct=0,
            confidence=0,
            reasoning="test",
            memories_used=[],
            setup="mean_reversion",
            market_regime="choppy",
        ),
    )
    monkeypatch.setattr(
        backtest,
        "_lesson_from_outcome",
        lambda *args, **kwargs: ("neither", "flat"),
    )
    output = backtest.run_backtest(
        start="2024-01-01",
        end="2024-04-09",
        memory_mode="none",
        bank_name="tiny-run",
        tiny=True,
        data=data,
        results_dir=tmp_path,
    )
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len({row["date"] for row in rows}) == 15
    assert {row["ticker"] for row in rows} == {"RELIANCE.NS", "TCS.NS"}
