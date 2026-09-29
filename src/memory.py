"""Hindsight-backed storage and retrieval of structured trade lessons."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any
from uuid import uuid4

from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt

from .schemas import TradeLesson


class HindsightMemory:
    """Store and retrieve lessons in an isolated Hindsight memory bank."""

    def __init__(self, mode: str = "memory-on", run_id: str | None = None) -> None:
        """Create a client using a bank unique to this run and execution mode."""
        project_root = Path(__file__).resolve().parents[1]
        load_dotenv(project_root / ".env")
        api_key = os.getenv("HINDSIGHT_API_KEY")
        base_url = os.getenv("HINDSIGHT_BASE_URL")
        if not api_key or not base_url:
            raise RuntimeError("HINDSIGHT_API_KEY and HINDSIGHT_BASE_URL are required")
        from hindsight_client import Hindsight

        self.bank_id = self._bank_id(mode, run_id)
        self.client = Hindsight(base_url=base_url, api_key=api_key)

    @staticmethod
    def _bank_id(mode: str, run_id: str | None) -> str:
        """Build a Hindsight-safe bank ID that cannot mix modes or runs."""
        clean_mode = re.sub(r"[^a-zA-Z0-9-]", "-", mode).strip("-") or "memory-on"
        clean_run = re.sub(r"[^a-zA-Z0-9-]", "-", run_id or uuid4().hex).strip("-")
        return f"bull-bear-desk-{clean_mode}-{clean_run}"

    @staticmethod
    def _lesson_text(lesson: TradeLesson) -> str:
        """Serialize every lesson field into searchable natural language."""
        return (
            f"Trade lesson for ticker {lesson.ticker} in sector {lesson.sector}, "
            f"date {lesson.date}. Setup: {lesson.setup}. Market regime: "
            f"{lesson.market_regime}. Indicators summary: {lesson.indicators_summary}. "
            f"Bull argument: {lesson.bull_argument}. Bear argument: "
            f"{lesson.bear_argument}. Judge decision: {lesson.judge_decision}. "
            f"Winner side: {lesson.winner_side}. Outcome: {lesson.outcome_pct:.4f}%. "
            f"Lesson: {lesson.lesson}"
        )

    @retry(stop=stop_after_attempt(3), reraise=True)
    def _retain(self, text: str) -> Any:
        """Call the documented synchronous Hindsight retain operation."""
        return self.client.retain(bank_id=self.bank_id, content=text)

    @retry(stop=stop_after_attempt(3), reraise=True)
    def _recall(self, query: str, k: int) -> list[str]:
        """Call the documented synchronous Hindsight recall operation."""
        response = self.client.recall(
            bank_id=self.bank_id,
            query=query,
            max_tokens=max(1, k) * 512,
        )
        return [result.text for result in response.results[:k]]

    def retain_lesson(self, lesson: TradeLesson) -> bool:
        """Retain a lesson, returning false instead of interrupting a run."""
        try:
            self._retain(self._lesson_text(lesson))
            return True
        except Exception:
            return False

    def recall_similar(
        self,
        ticker: str,
        sector: str,
        setup: str,
        market_regime: str,
        indicators_summary: str,
        k: int = 5,
    ) -> list[str]:
        """Return up to ``k`` relevant lesson texts, or an empty list on error."""
        if k < 1:
            return []
        query = (
            f"Similar trade for ticker {ticker}, sector {sector}, setup {setup}, "
            f"market regime {market_regime}, indicators {indicators_summary}"
        )
        try:
            return self._recall(query, k)
        except Exception:
            return []

    def recall_agent_track_record(self, setup: str, market_regime: str) -> list[str]:
        """Return distinct sides that won in recalled lessons for the conditions."""
        try:
            texts = self._recall(
                f"Trade track record: which side won for setup {setup} "
                f"in market regime {market_regime}?",
                50,
            )
            winners: list[str] = []
            for text in texts:
                match = re.search(
                    r"Winner side:\s*(bull|bear|neither)\b", text, flags=re.IGNORECASE
                )
                if match:
                    winner = match.group(1).lower()
                    if winner not in winners:
                        winners.append(winner)
            return winners
        except Exception:
            return []


class NoMemory:
    """Baseline memory implementation that never stores or recalls anything."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Accept the same construction arguments as HindsightMemory."""

    def retain_lesson(self, lesson: TradeLesson) -> bool:
        """Ignore a lesson in the memory-off baseline."""
        return False

    def recall_similar(
        self,
        ticker: str,
        sector: str,
        setup: str,
        market_regime: str,
        indicators_summary: str,
        k: int = 5,
    ) -> list[str]:
        """Return no recalled lessons."""
        return []

    def recall_agent_track_record(self, setup: str, market_regime: str) -> list[str]:
        """Return no track-record information."""
        return []


_default_memory: HindsightMemory | NoMemory | None = None


def _get_default_memory() -> HindsightMemory | NoMemory:
    """Lazily configure the default memory backend without import-time I/O."""
    global _default_memory
    if _default_memory is None:
        try:
            _default_memory = HindsightMemory()
        except Exception:
            _default_memory = NoMemory()
    return _default_memory


def retain_lesson(lesson: TradeLesson) -> bool:
    """Retain a lesson through the default configured memory backend."""
    return _get_default_memory().retain_lesson(lesson)


def recall_similar(
    ticker: str,
    sector: str,
    setup: str,
    market_regime: str,
    indicators_summary: str,
    k: int = 5,
) -> list[str]:
    """Recall similar lessons through the default configured backend."""
    return _get_default_memory().recall_similar(
        ticker, sector, setup, market_regime, indicators_summary, k
    )


def recall_agent_track_record(setup: str, market_regime: str) -> list[str]:
    """Recall winning sides through the default configured backend."""
    return _get_default_memory().recall_agent_track_record(setup, market_regime)
