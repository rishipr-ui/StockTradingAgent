"""Groq-backed bull, bear, and structured judge agents."""

from __future__ import annotations

import json
import os
import re
import sys
from typing import Any, Callable

from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt

from .prompts import BEAR_PROMPT, BULL_PROMPT, JUDGE_PROMPT
from .schemas import JudgeDecision

_last_usage_total_tokens: int | None = None


def get_last_usage_total_tokens() -> int | None:
    """Return the provider-reported token count from the latest response."""
    return _last_usage_total_tokens


def _client() -> Any:
    """Create the Groq client from environment configuration."""
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    model = os.getenv("GROQ_MODEL")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is missing")
    if not model:
        raise RuntimeError("GROQ_MODEL is missing")
    from groq import Groq

    return Groq(api_key=api_key)


def _model() -> str:
    """Return the required model name."""
    load_dotenv()
    model = os.getenv("GROQ_MODEL")
    if not model:
        raise RuntimeError("GROQ_MODEL is missing")
    return model


def _content(response: Any) -> str:
    """Extract normal or function-call JSON content from a Groq response."""
    global _last_usage_total_tokens
    usage = getattr(response, "usage", None)
    total_tokens = getattr(usage, "total_tokens", None)
    _last_usage_total_tokens = (
        int(total_tokens) if total_tokens is not None else None
    )
    message = response.choices[0].message
    content = getattr(message, "content", None)
    if content:
        return str(content)
    tool_calls = getattr(message, "tool_calls", None) or []
    if tool_calls:
        arguments = getattr(tool_calls[0].function, "arguments", None)
        if arguments:
            return str(arguments)
    raise ValueError("Groq response contained neither content nor tool arguments")


def _complete(
    prompt: str,
    *,
    json_mode: bool = False,
    max_tokens: int | None = None,
) -> str:
    """Send one chat completion using only caller-provided prompt data."""
    kwargs: dict[str, Any] = {
        "model": _model(),
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    return _content(_client().chat.completions.create(**kwargs))


def _parse_judge_response(raw: str) -> JudgeDecision:
    """Parse common JSON response wrappers and normalize enum-like fields."""
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise
        payload = json.loads(text[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("Judge response must be a JSON object")
    normalized = dict(payload)
    if isinstance(normalized.get("action"), str):
        normalized["action"] = normalized["action"].strip().upper()
    if isinstance(normalized.get("memories_used"), str):
        normalized["memories_used"] = [normalized["memories_used"]]
    return JudgeDecision.model_validate(normalized)


def _apply_signal_tiebreaker(
    decision: JudgeDecision,
    indicators_summary: str,
) -> JudgeDecision:
    """Turn only strong, explicit indicator signals into a bounded action."""
    if decision.action != "HOLD":
        return decision
    text = indicators_summary.lower()
    rsi_match = re.search(r"rsi\s+(\d+(?:\.\d+)?)", text)
    volume_match = re.search(r"volume\s+(\d+(?:\.\d+)?)x", text)
    if not rsi_match:
        return decision
    rsi = float(rsi_match.group(1))
    volume = float(volume_match.group(1)) if volume_match else None
    below_sma50 = "below sma50" in text
    above_sma50 = "above sma50" in text
    if rsi <= 35 and below_sma50 and (volume is None or volume >= 1):
        return decision.model_copy(
            update={
                "action": "BUY",
                "size_pct": 5.0,
                "confidence": max(decision.confidence, 0.55),
                "reasoning": (
                    f"{decision.reasoning} Signal tie-breaker: RSI {rsi:.0f} "
                    "is oversold with price below SMA50 and sufficient volume."
                ),
            }
        )
    if rsi >= 70 and above_sma50 and volume is not None and volume < 1:
        return decision.model_copy(
            update={
                "action": "SELL",
                "size_pct": 5.0,
                "confidence": max(decision.confidence, 0.55),
                "reasoning": (
                    f"{decision.reasoning} Signal tie-breaker: RSI {rsi:.0f} "
                    "is overbought above SMA50 on weak volume."
                ),
            }
        )
    if rsi < 50 and below_sma50 and (volume is None or volume >= 1):
        return decision.model_copy(
            update={
                "action": "SELL",
                "size_pct": 5.0,
                "confidence": max(decision.confidence, 0.55),
                "reasoning": (
                    f"{decision.reasoning} Signal tie-breaker: RSI {rsi:.0f} "
                    "is weak below SMA50 with active volume."
                ),
            }
        )
    return decision


def _under_word_limit(text: str, limit: int = 120) -> str:
    """Enforce the requested maximum output length."""
    words = text.split()
    return " ".join(words[:limit])


def _argument(
    template: str,
    ticker: str,
    indicators_summary: str,
    recent_headlines: str | list[str],
    *,
    max_tokens: int | None = None,
    argument_word_limit: int = 120,
) -> str:
    """Generate a bounded bull or bear argument."""
    headlines = (
        "\n".join(recent_headlines)
        if isinstance(recent_headlines, list)
        else recent_headlines
    )
    return _under_word_limit(
        _complete(
            template.format(
                ticker=ticker,
                indicators_summary=indicators_summary,
                recent_headlines=headlines,
                argument_word_limit=argument_word_limit,
            )
            ,
            max_tokens=max_tokens,
        )
    )


def bull_argue(
    ticker: str,
    indicators_summary: str,
    recent_headlines: str | list[str],
    *,
    max_tokens: int | None = None,
    argument_word_limit: int = 120,
) -> str:
    """Return the strongest indicator-grounded case for buying."""
    return _argument(
        BULL_PROMPT,
        ticker,
        indicators_summary,
        recent_headlines,
        max_tokens=max_tokens,
        argument_word_limit=argument_word_limit,
    )


def bear_argue(
    ticker: str,
    indicators_summary: str,
    recent_headlines: str | list[str],
    *,
    max_tokens: int | None = None,
    argument_word_limit: int = 120,
) -> str:
    """Return the strongest indicator-grounded case against buying."""
    return _argument(
        BEAR_PROMPT,
        ticker,
        indicators_summary,
        recent_headlines,
        max_tokens=max_tokens,
        argument_word_limit=argument_word_limit,
    )


@retry(stop=stop_after_attempt(3), reraise=True)
def _judge_call(prompt: str, max_tokens: int | None = None) -> JudgeDecision:
    """Generate and validate one structured judge response; retry failures."""
    raw = _complete(prompt, json_mode=True, max_tokens=max_tokens)
    return _parse_judge_response(raw)


def judge_decide(
    ticker: str,
    indicators_summary: str,
    bull_text: str,
    bear_text: str,
    recalled_lessons: list[str],
    agent_track_record: list[str],
    *,
    max_tokens: int | None = None,
) -> JudgeDecision:
    """Return a validated decision, or a safe HOLD after repeated LLM errors."""
    prompt = JUDGE_PROMPT.format(
        ticker=ticker,
        indicators_summary=indicators_summary,
        bull_text=bull_text,
        bear_text=bear_text,
        recalled_lessons="\n".join(recalled_lessons) or "no relevant memory",
        agent_track_record=", ".join(agent_track_record) or "no track record",
    )
    try:
        decision = _judge_call(prompt, max_tokens=max_tokens)
        return _apply_signal_tiebreaker(decision, indicators_summary)
    except Exception as exc:
        print(
            f"Judge fallback for {ticker}: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return JudgeDecision(
            action="HOLD",
            size_pct=0,
            confidence=0,
            reasoning="LLM error fallback",
            memories_used=[],
            setup="unknown",
            market_regime="unknown",
        )
