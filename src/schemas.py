"""Shared data schemas for simulation, agents, memory, and results."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class TradeLesson(BaseModel):
    """A structured lesson learned from one simulated or evaluated trade."""

    ticker: str
    sector: str
    date: str
    setup: Literal[
        "oversold_bounce",
        "breakout",
        "breakdown",
        "mean_reversion",
    ]
    market_regime: Literal[
        "trending_up",
        "trending_down",
        "choppy",
        "high_vol",
    ]
    indicators_summary: str
    bull_argument: str
    bear_argument: str
    judge_decision: Literal["BUY", "SELL", "HOLD"]
    winner_side: Literal["bull", "bear", "neither"]
    outcome_pct: float
    lesson: str


class JudgeDecision(BaseModel):
    """Validated structured decision returned by the judge agent."""

    action: Literal["BUY", "SELL", "HOLD"]
    size_pct: float
    confidence: float
    reasoning: str
    memories_used: list[str]
    setup: str
    market_regime: str

    def __init__(self, **data: object) -> None:
        """Validate the numeric decision bounds with Pydantic."""
        super().__init__(**data)
        if not 0 <= self.size_pct <= 10:
            raise ValueError("size_pct must be between 0 and 10")
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
