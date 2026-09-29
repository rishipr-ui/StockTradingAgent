"""Tests for backtest performance and memory metrics."""

import json

import pytest

from src.metrics import compute_run_metrics, summarize_results, write_outputs


def record(
    date,
    ticker,
    price,
    action,
    outcome,
    setup="breakout",
    regime="choppy",
    memories=None,
    equity=100,
):
    return {
        "date": date,
        "ticker": ticker,
        "price": price,
        "decision": {
            "action": action,
            "setup": setup,
            "market_regime": regime,
            "memories_used": memories or [],
        },
        "outcome": outcome,
        "equity": equity,
    }


def test_metrics_track_repeated_mistakes_and_losses_avoided():
    records = [
        record("2024-01-01", "A", 100, "BUY", -2, equity=98),
        record("2024-01-02", "A", 98, "BUY", -3, equity=95),
        record("2024-01-03", "A", 96, "HOLD", -4, memories=["prior loss"], equity=95),
        record("2024-01-04", "A", 95, "BUY", 2, equity=97),
        record("2024-01-05", "A", 97, "HOLD", 1, memories=["prior loss"], equity=97),
    ]

    metrics = compute_run_metrics(records)

    assert metrics["number_of_trades"] == 3
    assert metrics["win_rate"] == pytest.approx(1 / 3)
    assert metrics["repeated_mistake_rate"] == pytest.approx(2 / 3)
    assert metrics["losses_avoided"] == 1
    assert metrics["max_drawdown"] == pytest.approx(3 / 98)


def test_buy_and_hold_and_outputs(tmp_path):
    records = [
        record("2024-01-01", "A", 100, "HOLD", 1, equity=100),
        record("2024-01-02", "A", 110, "HOLD", 1, equity=105),
    ]
    path = tmp_path / "run.jsonl"
    path.write_text("\n".join(json.dumps(item) for item in records), encoding="utf-8")

    summary = summarize_results(tmp_path)
    assert summary["run"]["buy_and_hold_return"] == pytest.approx(0.10)
    write_outputs(summary, tmp_path / "summary.json", tmp_path / "comparison.csv")
    assert (tmp_path / "summary.json").exists()
    assert "run,total_return" in (tmp_path / "comparison.csv").read_text()
