"""Groq-backed bull, bear, and structured judge agents."""

from __future__ import annotations

import json
import os
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
    payload = json.loads(raw)
    return JudgeDecision.model_validate(payload)


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
        return _judge_call(prompt, max_tokens=max_tokens)
    except Exception:
        return JudgeDecision(
            action="HOLD",
            size_pct=0,
            confidence=0,
            reasoning="LLM error fallback",
            memories_used=[],
            setup="unknown",
            market_regime="unknown",
        )
