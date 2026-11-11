"""Claude Code CLI failures as values, mapped to caller-facing errors at one exhaustive boundary."""

import re
from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from typing import Final, TypeGuard, assert_never

from .schemas import AdapterError

_DETAIL_LIMIT: Final = 2000
_DETAIL_KEYS: Final = ("result", "message", "error")
_ANSI_ESCAPE: Final = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
_CONTROL_CHARACTERS: Final = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]|\r\n?|\n")
_MISSING_BINARY: Final = re.compile(
    r"enoent|cannot find the file|command not found|executable file not found"
    r"|\bno such file(?: or directory)?\b|:\s*not found\b",
    re.IGNORECASE,
)
_NOT_AUTHENTICATED: Final = re.compile(
    r"log[\s-]?in|logged[\s-]?in|auth|unauthorized|invalid api key|credential", re.IGNORECASE
)
_AUTHENTICATION_FINE: Final = re.compile(
    r"\bauthenticated successfully\b|\bdeauthenticate\b|\blogin\.log\b|\blogged[\s-]?in successfully\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class CLINotInstalled:
    executable: str


@dataclass(frozen=True, slots=True)
class CLINotAuthenticated:
    detail: str


@dataclass(frozen=True, slots=True)
class RateLimited:
    detail: str


@dataclass(frozen=True, slots=True)
class PromptTooLong:
    detail: str


@dataclass(frozen=True, slots=True)
class TimedOut:
    seconds: float


@dataclass(frozen=True, slots=True)
class QueueFull:
    waiting: int
    retry_after_seconds: int


@dataclass(frozen=True, slots=True)
class UnexpectedExit:
    exit_code: int
    stderr: str


CLIFailure = CLINotInstalled | CLINotAuthenticated | RateLimited | PromptTooLong | TimedOut | QueueFull | UnexpectedExit


def _trim(detail: str) -> str:
    """Bound the text and strip terminal escapes, since it reaches operator terminals and log aggregators."""
    clean: Final = _CONTROL_CHARACTERS.sub(" ", _ANSI_ESCAPE.sub("", detail)).strip()
    return f"{clean[:_DETAIL_LIMIT]}..." if len(clean) > _DETAIL_LIMIT else clean


def _duration(seconds: float) -> str:
    """Render a timeout plainly: no scientific notation, no NaN, no negatives reaching the operator."""
    safe: Final = seconds if isfinite(seconds) and seconds >= 0 else 0.0
    return f"{safe:.4f}".rstrip("0").rstrip(".") or "0"


def raise_public(failure: CLIFailure) -> AdapterError:
    """Map a failure to the caller-facing error. Operator-facing text names the proxy server."""
    match failure:
        case CLINotInstalled(executable):
            return AdapterError(
                status_code=503,
                code="cli_not_installed",
                message=(
                    f"Claude Code CLI is not installed on the proxy server: `{executable}` was not found. "
                    "Install it there, or set CLAUDE_CODE_PATH to its location."
                ),
            )
        case CLINotAuthenticated(detail):
            return AdapterError(
                status_code=503,
                code="cli_not_authenticated",
                message=(
                    "Claude Code CLI is not authenticated on the proxy server. "
                    "Run `claude` there and log in as the same OS user the proxy runs as."
                    + (f" Upstream said: {upstream}" if (upstream := _trim(detail)) else "")
                ),
            )
        case RateLimited(detail):
            return AdapterError(
                status_code=429,
                message=(
                    f"Claude Code CLI was rate limited: {limit}"
                    if (limit := _trim(detail))
                    else "Claude Code CLI was rate limited."
                ),
            )
        case PromptTooLong(detail):
            return AdapterError(
                status_code=400,
                message=(
                    f"Prompt is too long for the model: {overflow}"
                    if (overflow := _trim(detail))
                    else "Prompt is too long for the model."
                ),
            )
        case TimedOut(seconds):
            return AdapterError(
                status_code=504,
                message=f"Claude Code CLI timed out after {_duration(seconds)}s on the proxy server.",
            )
        case QueueFull(waiting, retry_after_seconds):
            return AdapterError(
                status_code=429,
                message=(
                    f"Claude Code CLI is busy on the proxy server: {waiting} requests are already waiting. "
                    f"Try again in about {retry_after_seconds}s."
                ),
                num_retries=0,
                retry_after=retry_after_seconds,
            )
        case UnexpectedExit(exit_code, stderr):
            return AdapterError(
                status_code=502,
                message=(
                    f"Claude Code CLI exited with code {exit_code} on the proxy server."
                    + (f" Error output: {output}" if (output := _trim(stderr)) else "")
                ),
            )
        case _:
            assert_never(failure)


def _is_event(event: object) -> TypeGuard[Mapping[str, object]]:
    return isinstance(event, Mapping)


def _event_text(event: Mapping[str, object]) -> str:
    """The `errors` list is the only place the CLI explains some exits, e.g. running out of turns."""
    listed: Final = event.get("errors")
    candidates: Final = (
        *(event.get(key) for key in _DETAIL_KEYS),
        *(listed if isinstance(listed, (list, tuple)) else ()),
    )
    values: Final = tuple(value.strip() for value in candidates if isinstance(value, str) and value.strip())
    return " ".join(dict.fromkeys(values))


def _is_rate_limit(event: Mapping[str, object]) -> bool:
    return event.get("api_error_status") in (429, "429") or _mentions_rate_limit(_event_text(event))


def _mentions_rate_limit(text: str) -> bool:
    lowered: Final = text.lower()
    return "rate limit" in lowered or "rate_limit" in lowered or "429" in lowered or "too many requests" in lowered


def _mentions_long_prompt(text: str) -> bool:
    lowered: Final = text.lower()
    return any(
        phrase in lowered
        for phrase in ("prompt is too long", "prompt too long", "prompt_too_long", "context length exceeded")
    )


def _mentions_missing_binary(text: str, executable: str) -> bool:
    """`no such file` alone also describes a missing workspace file, so it only counts near the executable name."""
    if not (match := _MISSING_BINARY.search(text)):
        return False
    if "no such file" not in match.group().lower():
        return True
    return executable.lower() in text.lower()


def _mentions_missing_auth(text: str) -> bool:
    return bool(_NOT_AUTHENTICATED.search(text)) and not _AUTHENTICATION_FINE.search(text)


def _rate_limit_detail(events: tuple[Mapping[str, object], ...]) -> str | None:
    hits: Final = tuple(_event_text(event) for event in events if _is_rate_limit(event))
    return next((text for text in hits if text), next(iter(hits), None))


def classify_failure(
    *,
    exit_code: int,
    stderr: str,
    stdout_events: tuple[object, ...] = (),
    executable: str = "claude",
    spawn_error: OSError | None = None,
) -> CLIFailure:
    """Decide which failure a non-zero CLI run represents. Timeouts are built by the caller, not detected here.

    `stdout_events` is whatever parsing the CLI's output produced, so entries are checked rather than trusted.
    """
    if isinstance(spawn_error, FileNotFoundError) or _mentions_missing_binary(stderr, executable):
        return CLINotInstalled(executable=executable)
    if spawn_error is not None:
        return UnexpectedExit(exit_code=exit_code, stderr=f"{stderr} {spawn_error}".strip())

    events: Final = tuple(event for event in stdout_events if _is_event(event))
    event_text: Final = " ".join(_event_text(event) for event in events).strip()

    # A long prompt is not retryable, so it must win over a rate limit the same output may also mention.
    if _mentions_long_prompt(stderr):
        return PromptTooLong(detail=stderr.strip())
    if _mentions_long_prompt(event_text):
        return PromptTooLong(detail=event_text)

    # A missing login stays broken until someone fixes it, so it must not be reported as a retryable rate limit.
    if _mentions_missing_auth(stderr):
        return CLINotAuthenticated(detail=stderr.strip())
    if _mentions_missing_auth(event_text):
        return CLINotAuthenticated(detail=event_text)

    if (detail := _rate_limit_detail(events)) is not None:
        return RateLimited(detail=detail)
    if _mentions_rate_limit(stderr):
        return RateLimited(detail=stderr.strip())

    return UnexpectedExit(exit_code=exit_code, stderr=_trim(stderr.strip() or event_text))
