"""Claude Code CLI events -> streaming chunks and the assembled answer, as one fold both paths share."""

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from typing import (
    Any,  # GenericStreamingChunk.provider_specific_fields is dict[str, Any]
    Final,
    Literal,
    TypeAlias,
)

from pydantic import TypeAdapter, ValidationError

from .schemas import (
    ChatCompletionToolCallChunk,
    ChatCompletionToolCallFunctionChunk,
    ChatCompletionUsageBlock,
    GenericStreamingChunk,
    PromptTokensDetailsWrapper,
    Usage,
    map_finish_reason,
)
from .tools import ClientTool, caller_name
from .usage import read_cli_usage

DEFAULT_FINISH_REASON: Final = "stop"
TOOL_CALLS_FINISH_REASON: Final = "tool_calls"
_WEB_SEARCH_TOOL_NAME: Final = "WebSearch"
_PROGRESS_FIELD: Final = "claude_code_cli_web_search"

SearchProgressKind: TypeAlias = Literal["started", "results"]


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A call to one of the caller's tools, under the caller's own name, for the caller to run."""

    id: str
    name: str
    arguments: str


@dataclass(frozen=True, slots=True)
class StreamState:
    """Everything the fold needs between events: the answer so far, the dedup flag, and the best usage seen."""

    text_parts: tuple[str, ...] = ()
    saw_delta: bool = False
    snapshot_usage: Mapping[str, object] | None = None
    delta_usage: Mapping[str, object] | None = None
    result_usage: Mapping[str, object] | None = None
    stop_reason: str = ""
    pending_search_ids: tuple[str, ...] = ()
    search_progress: tuple[SearchProgressKind, ...] = ()
    client_tools: tuple[ClientTool, ...] = ()
    tool_calls: tuple[ToolCall, ...] = ()
    awaiting_caller: bool = False


@dataclass(frozen=True, slots=True)
class AssembledAnswer:
    """The non-streaming view of a run: one complete text, plus the same usage the closing chunk carries."""

    text: str
    finish_reason: str
    usage: Usage
    tool_calls: tuple[ToolCall, ...] = ()


_BLOCKS_ADAPTER: Final = TypeAdapter(tuple[dict[str, object], ...])
_OBJECT_ADAPTER: Final = TypeAdapter(dict[str, object])


def _object_at(payload: Mapping[str, object], key: str) -> Mapping[str, object] | None:
    """CLI events are untrusted JSON, so a nested value becomes a typed mapping only once it validates."""
    try:
        return _OBJECT_ADAPTER.validate_python(payload.get(key))
    except ValidationError:
        return None


def _text_of(value: object) -> str:
    return value if isinstance(value, str) else ""


def _snapshot_text(message: Mapping[str, object]) -> str:
    """Only `text` blocks are the answer; a thinking block is reasoning the caller never asked to see."""
    try:
        blocks: Final = _BLOCKS_ADAPTER.validate_python(message.get("content"))
    except ValidationError:
        return ""
    return "".join(_text_of(block.get("text")) for block in blocks if block.get("type") == "text")


def _usage_of(payload: Mapping[str, object], key: str = "usage") -> Mapping[str, object] | None:
    return _object_at(payload, key)


def _fold_stream_event(state: StreamState, inner: Mapping[str, object]) -> StreamState:
    inner_type: Final = inner.get("type")
    if inner_type == "content_block_delta":
        delta: Final = _object_at(inner, "delta")
        # `thinking_delta` and `signature_delta` share this event shape; neither belongs in the answer.
        if delta is None or delta.get("type") != "text_delta" or not (text := _text_of(delta.get("text"))):
            return state
        return replace(state, text_parts=(*state.text_parts, text), saw_delta=True)
    if inner_type == "message_delta":
        delta_body: Final = _object_at(inner, "delta")
        stop_reason: Final = _text_of(delta_body.get("stop_reason")) if delta_body is not None else ""
        return replace(
            state,
            delta_usage=_usage_of(inner) or state.delta_usage,
            stop_reason=stop_reason or state.stop_reason,
            # Every snapshot of the turn precedes its `message_delta`, so all parallel calls are in by now.
            awaiting_caller=state.awaiting_caller or (stop_reason == "tool_use" and bool(state.tool_calls)),
        )
    if inner_type == "content_block_start":
        block: Final = _object_at(inner, "content_block")
        if block is None or block.get("type") != "tool_use" or block.get("name") != _WEB_SEARCH_TOOL_NAME:
            return state
        tool_use_id: Final = block.get("id")
        if not isinstance(tool_use_id, str):
            return state
        return replace(
            state,
            pending_search_ids=(*state.pending_search_ids, tool_use_id),
            search_progress=(*state.search_progress, "started"),
        )
    return state


def _arguments(value: object) -> str:
    try:
        return json.dumps(value if value is not None else {}, ensure_ascii=False)
    except (TypeError, ValueError):
        return "{}"


def _caller_tool_calls(state: StreamState, message: Mapping[str, object]) -> tuple[ToolCall, ...]:
    """Only calls to the caller's tools are returned; a built-in like WebSearch stays inside the CLI."""
    try:
        blocks: Final = _BLOCKS_ADAPTER.validate_python(message.get("content"))
    except ValidationError:
        return ()
    seen: Final = frozenset(call.id for call in state.tool_calls)
    return tuple(
        ToolCall(id=block_id, name=name, arguments=_arguments(block.get("input")))
        for block in blocks
        if block.get("type") == "tool_use"
        and isinstance(block_id := block.get("id"), str)
        and block_id not in seen
        and (name := caller_name(state.client_tools, _text_of(block.get("name")))) is not None
    )


def _fold_assistant(state: StreamState, message: Mapping[str, object]) -> StreamState:
    """Deltas already carried this text, so a snapshot after them would double the answer."""
    snapshot: Final = "" if state.saw_delta else _snapshot_text(message)
    return replace(
        state,
        text_parts=(*state.text_parts, snapshot) if snapshot else state.text_parts,
        snapshot_usage=_usage_of(message) or state.snapshot_usage,
        tool_calls=(*state.tool_calls, *_caller_tool_calls(state, message)),
    )


def _tool_result_id(message: Mapping[str, object]) -> str | None:
    try:
        blocks: Final = _BLOCKS_ADAPTER.validate_python(message.get("content"))
    except ValidationError:
        return None
    if not blocks or blocks[0].get("type") != "tool_result":
        return None
    tool_use_id: Final = blocks[0].get("tool_use_id")
    return tool_use_id if isinstance(tool_use_id, str) else None


def _fold_user(state: StreamState, message: Mapping[str, object]) -> StreamState:
    tool_use_id: Final = _tool_result_id(message)
    if tool_use_id is None or tool_use_id not in state.pending_search_ids:
        return state
    return replace(
        state,
        pending_search_ids=tuple(id_ for id_ in state.pending_search_ids if id_ != tool_use_id),
        search_progress=(*state.search_progress, "results"),
    )


def _fold(state: StreamState, event: Mapping[str, object]) -> StreamState:
    match event.get("type"):
        case "stream_event":
            inner: Final = _object_at(event, "event")
            return _fold_stream_event(state, inner) if inner is not None else state
        case "assistant":
            message: Final = _object_at(event, "message")
            return _fold_assistant(state, message) if message is not None else state
        case "user":
            user_message: Final = _object_at(event, "message")
            return _fold_user(state, user_message) if user_message is not None else state
        case "result":
            return replace(
                state,
                result_usage=_usage_of(event) or state.result_usage,
                stop_reason=_text_of(event.get("stop_reason")) or state.stop_reason,
            )
        case _:
            return state


def advance(state: StreamState, event: Mapping[str, object]) -> tuple[StreamState, tuple[GenericStreamingChunk, ...]]:
    """Fold one event and emit the chunks it newly contributed. The incremental path calls this per event."""
    folded: Final = _fold(state, event)
    new_text: Final = tuple(text_chunk(text) for text in folded.text_parts[len(state.text_parts) :])
    new_progress: Final = tuple(
        search_progress_chunk(kind) for kind in folded.search_progress[len(state.search_progress) :]
    )
    new_calls: Final = tuple(
        tool_call_chunk(call, index) for index, call in enumerate(folded.tool_calls) if index >= len(state.tool_calls)
    )
    return folded, (*new_progress, *new_text, *new_calls)


def _final_usage(state: StreamState) -> Usage:
    """The terminal `result` totals win; a mid-stream `message_delta` is the next best; a snapshot is a floor."""
    payload: Final = state.result_usage or state.delta_usage or state.snapshot_usage
    return read_cli_usage(dict(payload)) if payload is not None else read_cli_usage({})


def _finish_reason(state: StreamState) -> str:
    if state.awaiting_caller:
        return TOOL_CALLS_FINISH_REASON
    return map_finish_reason(state.stop_reason) if state.stop_reason else DEFAULT_FINISH_REASON


def _count_of(source: object, field: str) -> int:
    """`Usage` carries the cache counts as dynamic extras, so they are only reachable by name."""
    value: Final = getattr(source, field, None) if source is not None else None
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _usage_block(usage: Usage) -> ChatCompletionUsageBlock:
    """The cache counts must ride along, or a streamed run reports no cache spend and its cost is wrong."""
    cache_read: Final = _count_of(usage, "cache_read_input_tokens")
    cache_creation: Final = _count_of(usage, "cache_creation_input_tokens")
    return ChatCompletionUsageBlock(
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        total_tokens=usage.total_tokens,
        prompt_tokens_details=PromptTokensDetailsWrapper(
            text_tokens=_count_of(usage.prompt_tokens_details, "text_tokens"),
            cached_tokens=cache_read,
            cache_write_tokens=cache_creation,
        ).model_dump(),
        completion_tokens_details=None,
        **_cache_totals(cache_read, cache_creation),
    )


def _cache_totals(cache_read: int, cache_creation: int) -> dict[str, int]:
    """`Usage(**block)` keeps unknown keys, and these two are the only route to the top-level counts."""
    return {"cache_read_input_tokens": cache_read, "cache_creation_input_tokens": cache_creation}


def text_chunk(text: str) -> GenericStreamingChunk:
    return GenericStreamingChunk(text=text, is_finished=False, finish_reason="", usage=None, index=0)


def search_progress_chunk(kind: SearchProgressKind) -> GenericStreamingChunk:
    """A caller-visible marker that a web search started or finished, carried outside `text`.

    Progress rides `provider_specific_fields` rather than `text` so `assemble_answer`'s text join
    stays answer-only with no new filtering logic, and a caller cannot mistake progress for content.
    """
    fields: Final[dict[str, Any]] = {_PROGRESS_FIELD: kind}  # mutable-ok: GenericStreamingChunk requires dict[str, Any]
    return GenericStreamingChunk(
        text="",
        is_finished=False,
        finish_reason="",
        usage=None,
        index=0,
        provider_specific_fields=fields,
    )


def tool_call_chunk(call: ToolCall, index: int) -> GenericStreamingChunk:
    return GenericStreamingChunk(
        text="",
        is_finished=False,
        finish_reason="",
        usage=None,
        index=0,
        tool_use=ChatCompletionToolCallChunk(
            id=call.id,
            type="function",
            function=ChatCompletionToolCallFunctionChunk(name=call.name, arguments=call.arguments),
            index=index,
        ),
    )


def closing_chunk(*, finish_reason: str, usage: Usage) -> GenericStreamingChunk:
    return GenericStreamingChunk(
        text="",
        is_finished=True,
        finish_reason=finish_reason,
        usage=_usage_block(usage),
        index=0,
    )


def _drain(
    events: Iterable[Mapping[str, object]],
    client_tools: tuple[ClientTool, ...] = (),
) -> tuple[StreamState, tuple[GenericStreamingChunk, ...]]:
    """One pass, one fold, so a one-shot iterator works and the two paths cannot see different events."""
    state: StreamState = StreamState(client_tools=client_tools)  # rebind-ok: a fold over a stream of unknown length
    chunks: list[GenericStreamingChunk] = []  # mutable-ok: accumulating one chunk per contributing event
    for event in events:
        state, emitted = advance(state, event)
        chunks.extend(emitted)
    return state, tuple(chunks)


def close_stream(state: StreamState) -> GenericStreamingChunk:
    """The one chunk that carries the finish reason and the usage. Every path ends with exactly this."""
    return closing_chunk(finish_reason=_finish_reason(state), usage=_final_usage(state))


def stream_chunks(
    events: Iterable[Mapping[str, object]],
    client_tools: tuple[ClientTool, ...] = (),
) -> tuple[GenericStreamingChunk, ...]:
    """Convert a whole recorded run in one call: the same fold the incremental path runs, drained eagerly."""
    state, chunks = _drain(events, client_tools)
    return (*chunks, close_stream(state))


def answer_of(state: StreamState) -> AssembledAnswer:
    """The non-streaming view of a fold, so it can never disagree with the streamed text."""
    return AssembledAnswer(
        text="".join(state.text_parts),
        finish_reason=_finish_reason(state),
        usage=_final_usage(state),
        tool_calls=state.tool_calls,
    )


def assemble_answer(
    events: Iterable[Mapping[str, object]],
    client_tools: tuple[ClientTool, ...] = (),
) -> AssembledAnswer:
    return answer_of(_drain(events, client_tools)[0])


__all__ = [
    "DEFAULT_FINISH_REASON",
    "TOOL_CALLS_FINISH_REASON",
    "AssembledAnswer",
    "SearchProgressKind",
    "StreamState",
    "ToolCall",
    "advance",
    "answer_of",
    "assemble_answer",
    "close_stream",
    "closing_chunk",
    "search_progress_chunk",
    "stream_chunks",
    "text_chunk",
    "tool_call_chunk",
]
