"""Claude Code CLI usage payloads -> LiteLLM's `Usage`, with cache-creation and cache-read counts intact."""

from math import isfinite
from typing import Final

from .schemas import PromptTokensDetailsWrapper, Usage


def _token_count(payload: dict[str, object], key: str) -> int:
    value: Final = payload.get(key)
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return max(0, value)
    if isinstance(value, float):
        return max(0, int(value)) if isfinite(value) else 0
    if isinstance(value, str):
        return _parsed_count(value)
    return 0


def _parsed_count(value: str) -> int:
    """JSON from a subprocess can carry counts as text; a real number there must not read as zero."""
    try:
        return _token_count({"value": float(value)}, "value")
    except ValueError:
        return 0


def read_cli_usage(payload: dict[str, object]) -> Usage:
    """Read the CLI's Anthropic-shaped usage payload, coercing missing or malformed counts to zero."""
    input_tokens: Final = _token_count(payload, "input_tokens")
    output_tokens: Final = _token_count(payload, "output_tokens")
    cache_creation_input_tokens: Final = _token_count(payload, "cache_creation_input_tokens")
    cache_read_input_tokens: Final = _token_count(payload, "cache_read_input_tokens")

    prompt_tokens: Final = input_tokens + cache_creation_input_tokens + cache_read_input_tokens

    return Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=output_tokens,
        total_tokens=prompt_tokens + output_tokens,
        prompt_tokens_details=PromptTokensDetailsWrapper(text_tokens=input_tokens),
        cache_creation_input_tokens=cache_creation_input_tokens,
        cache_read_input_tokens=cache_read_input_tokens,
    )
