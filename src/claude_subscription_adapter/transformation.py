"""OpenAI chat messages -> the system prompt and alternating conversation the Claude Code CLI expects."""

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from itertools import groupby
from types import MappingProxyType
from typing import Final, Literal, TypeGuard

from pydantic import BaseModel, ConfigDict, TypeAdapter

from .schemas import AllMessageValues

ConversationRole = Literal["user", "assistant"]


_SYSTEM_ROLES: Final[frozenset[str]] = frozenset({"system", "developer"})
_TOOL_ROLES: Final[frozenset[str]] = frozenset({"tool", "function"})
_PART_SEPARATOR: Final = "\n"
_TURN_SEPARATOR: Final = "\n\n"
_EMPTY_RESULT: Final = "(empty result)"


class _CalledFunction(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str = ""
    arguments: object = None


class _ToolCall(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str = ""
    function: _CalledFunction = _CalledFunction()


class _Message(BaseModel):
    """The only fields of an incoming message this conversion reads, validated into known types."""

    model_config = ConfigDict(extra="ignore", arbitrary_types_allowed=True)

    role: str = ""
    content: object = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: tuple[_ToolCall, ...] | None = None


_MESSAGE_ADAPTER: Final = TypeAdapter(_Message)


@dataclass(frozen=True, slots=True)
class ConversationTurn:
    role: ConversationRole
    content: str


@dataclass(frozen=True, slots=True)
class CLIInput:
    system_prompt: str
    conversation: tuple[ConversationTurn, ...]


def _stringify_part(part: object) -> str:
    if isinstance(part, str):
        return part
    if part is None:
        return ""
    if isinstance(part, (bytes, bytearray)):
        return part.decode("utf-8", errors="replace")
    if _is_part_mapping(part):
        text: Final = part.get("text")
        if part.get("type") == "text" and isinstance(text, str):
            return text
        return _serialise(part)
    if _is_part_sequence(part):
        return _stringify_content(part)
    return str(part)


def _is_part_mapping(content: object) -> TypeGuard[Mapping[object, object]]:
    return isinstance(content, Mapping)


def _serialise(part: object) -> str:
    """Render a non-text part without ever raising, whatever the provider put in it."""
    try:
        return json.dumps(part, ensure_ascii=False, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return str(part)


def _is_part_sequence(content: object) -> TypeGuard[Iterable[object]]:
    return isinstance(content, Iterable) and not isinstance(content, (str, bytes, Mapping))


def _stringify_content(content: object) -> str:
    if _is_part_sequence(content):
        parts: Final = tuple(_stringify_part(part) for part in content)
        return _PART_SEPARATOR.join(part for part in parts if part.strip())
    return _stringify_part(content)


def _render_call(call: _ToolCall) -> str:
    arguments: Final = call.function.arguments
    rendered: Final = arguments if isinstance(arguments, str) else _serialise(arguments)
    return f"Called tool `{call.function.name}` (id {call.id}) with arguments: {rendered}"


def _body(message: _Message) -> str:
    """An assistant turn that only called tools has no content, but the calls are what it said."""
    text: Final = _stringify_content(message.content).strip()
    calls: Final = tuple(_render_call(call) for call in message.tool_calls or ()) if message.role == "assistant" else ()
    return _PART_SEPARATOR.join(part for part in (text, *calls) if part)


def _to_turn(message: _Message, body: str, tool_names: Mapping[str, str]) -> ConversationTurn | None:
    if message.role == "assistant":
        return ConversationTurn(role="assistant", content=body)
    if message.role in _TOOL_ROLES:
        call_id: Final = message.tool_call_id or ""
        name: Final = message.name or tool_names.get(call_id) or call_id or "tool"
        call_label: Final = f" (id {call_id})" if call_id and call_id != name else ""
        return ConversationTurn(role="user", content=f"Tool result from `{name}`{call_label}:\n{body or _EMPTY_RESULT}")
    if message.role == "user":
        return ConversationTurn(role="user", content=body)
    return None


def _turn_role(turn: ConversationTurn) -> ConversationRole:
    return turn.role


def _merge_consecutive(turns: tuple[ConversationTurn, ...]) -> tuple[ConversationTurn, ...]:
    grouped: Final = groupby(turns, key=_turn_role)
    return tuple(
        ConversationTurn(role=role, content=_TURN_SEPARATOR.join(turn.content for turn in group))
        for role, group in grouped
    )


def prepare_cli_input(messages: list[AllMessageValues], *, skip_turns: int = 0) -> CLIInput:
    """Split OpenAI messages into one system prompt plus a strictly alternating conversation.

    `skip_turns` drops that many turns off the front of the merged conversation, for a resumed session
    that already has them in its CLI-side transcript: only the unsent tail needs to go over stdin. A
    `skip_turns` larger than this call's own conversation is clamped to it rather than trusted blindly,
    since a tracked count from a racing sibling request is stale evidence about this caller's own
    history, and it must never make a turn silently vanish from what reaches the CLI.
    """
    validated: Final = tuple(_MESSAGE_ADAPTER.validate_python(message) for message in messages)
    bodies: Final = tuple((message, _body(message)) for message in validated)
    tool_names: Final = MappingProxyType(
        {call.id: call.function.name for message in validated for call in message.tool_calls or () if call.id}
    )

    system_prompt: Final = _TURN_SEPARATOR.join(
        body for message, body in bodies if message.role in _SYSTEM_ROLES and body
    )
    turns: Final = tuple(
        turn
        for turn in (
            _to_turn(message, body, tool_names) for message, body in bodies if body or message.role in _TOOL_ROLES
        )
        if turn is not None
    )
    merged: Final = _merge_consecutive(turns)
    clamped_skip: Final = min(skip_turns, max(len(merged) - 1, 0))
    return CLIInput(system_prompt=system_prompt, conversation=merged[clamped_skip:])
