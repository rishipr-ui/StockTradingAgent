"""Tests for agent prompt use, output validation, and fallback behavior."""

from src import agents


def test_arguments_are_bounded_and_use_indicator_input(monkeypatch):
    monkeypatch.setattr(
        agents,
        "_complete",
        lambda prompt, **kwargs: "RSI 27 supports a cautious oversold bounce.",
    )

    text = agents.bull_argue("INFY.NS", "RSI 27 (oversold)", ["headline"])

    assert len(text.split()) <= 120
    assert "RSI 27" in text


def test_judge_validates_json(monkeypatch):
    monkeypatch.setattr(
        agents,
        "_complete",
        lambda prompt, **kwargs: (
            '{"action":"BUY","size_pct":5,"confidence":0.8,'
            '"reasoning":"Indicators support the setup.","memories_used":[],'
            '"setup":"oversold_bounce","market_regime":"choppy"}'
        ),
    )

    decision = agents.judge_decide("INFY.NS", "RSI 27", "bull", "bear", [], [])

    assert decision.action == "BUY"
    assert decision.size_pct == 5
    assert decision.confidence == 0.8


def test_judge_falls_back_after_invalid_json(monkeypatch):
    monkeypatch.setattr(agents, "_complete", lambda prompt, **kwargs: "not json")

    decision = agents.judge_decide("INFY.NS", "RSI 27", "bull", "bear", [], [])

    assert decision.action == "HOLD"
    assert decision.reasoning == "LLM error fallback"
    assert decision.size_pct == 0
