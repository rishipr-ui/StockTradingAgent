"""Tests for Hindsight lesson serialization and memory-off behavior."""

from unittest.mock import Mock

from src.memory import HindsightMemory, NoMemory
from src.schemas import TradeLesson


def lesson() -> TradeLesson:
    return TradeLesson(
        ticker="INFY.NS",
        sector="IT",
        date="2024-01-15",
        setup="oversold_bounce",
        market_regime="choppy",
        indicators_summary="RSI 27 (oversold), price below SMA50, volume 2.1x average",
        bull_argument="Oversold conditions could produce a relief bounce.",
        bear_argument="Earnings risk may keep sellers in control.",
        judge_decision="HOLD",
        winner_side="bear",
        outcome_pct=-4.25,
        lesson="Wait for confirmation before buying an oversold bounce near earnings.",
    )


def test_lesson_text_includes_every_field():
    text = HindsightMemory._lesson_text(lesson())

    for value in lesson().model_dump().values():
        assert str(value) in text


def test_retain_and_recall_use_documented_client_calls(monkeypatch):
    memory = object.__new__(HindsightMemory)
    memory.bank_id = "test-bank"
    memory.client = Mock()
    memory.client.recall.return_value.results = [Mock(text="Winner side: bear")]

    assert memory.retain_lesson(lesson())
    memory.client.retain.assert_called_once()
    assert memory.recall_similar(
        "INFY.NS", "IT", "oversold_bounce", "choppy", "RSI 27", k=1
    ) == ["Winner side: bear"]
    assert memory.recall_agent_track_record("oversold_bounce", "choppy") == ["bear"]
    assert memory.client.recall.call_count == 2


def test_no_memory_has_same_interface_and_returns_empty():
    memory = NoMemory()

    assert memory.retain_lesson(lesson()) is False
    assert (
        memory.recall_similar("INFY.NS", "IT", "breakout", "trending_up", "RSI 55")
        == []
    )
    assert memory.recall_agent_track_record("breakout", "trending_up") == []
