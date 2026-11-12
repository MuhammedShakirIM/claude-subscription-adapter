"""The seam LiteLLM calls: prepare once, run the transport, hand back a response or a stream of chunks."""

from collections.abc import AsyncGenerator, AsyncIterator, Callable, Mapping
from contextlib import aclosing
from dataclasses import dataclass, replace
from hashlib import sha256
from math import isfinite
from os import environ
from typing import Final

from pydantic import TypeAdapter, ValidationError

from .errors import raise_public
from .schemas import AdapterError, AllMessageValues, Choices, GenericStreamingChunk, Message, ModelResponse
from .sessions import NewSession, ResumeSession, SessionDecision, SessionStore
from .streaming import StreamState, ToolCall, advance, answer_of, close_stream
from .tools import ClientTool, read_client_tools, with_caller_machine_note
from .transformation import CLIInput, prepare_cli_input
from .transport import CLISettings, CLISpawner, read_settings, spawn_subprocess, stream_cli

_SYNC_NOT_SUPPORTED: Final = (
    "The Claude Code CLI provider is async only. Call it through `acompletion`/`astreaming` "
    "(the LiteLLM proxy already does), not the synchronous entry points."
)

_MAX_TOKENS_PARAMS: Final[tuple[str, ...]] = ("max_tokens", "max_completion_tokens")
_PARAMS_ADAPTER: Final = TypeAdapter(dict[str, object])


@dataclass(frozen=True, slots=True)
class _PreparedCall:
    """One conversion of the request, shared by both async paths so they cannot read it differently."""

    settings: CLISettings
    cli_input: CLIInput
    model: str
    client_tools: tuple[ClientTool, ...]
    session: SessionDecision | None
    session_key: str | None
    turns_sent: int
    system_prompt: str


def _positive_int(name: str, value: object) -> int:
    """A non-integral count is refused rather than truncated: 3.9 silently becoming 3 is a silent drop."""
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(value)
        or value != int(value)
        or int(value) < 1
    ):
        raise AdapterError(
            status_code=400,
            message=f"`{name}` must be a positive whole number of tokens, got {value!r}.",
        )
    return int(value)


def _positive_float(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) or value <= 0:
        raise AdapterError(
            status_code=400,
            message=f"`{name}` must be a positive number of seconds, got {value!r}.",
        )
    return float(value)


def _required_model(model: str) -> str:
    """An empty name reaches the CLI as `--model ''` and fails late as an opaque error, so refuse it here."""
    if not (named := model.strip()):
        raise AdapterError(
            status_code=400,
            message="No model was named for the Claude Code CLI provider. Pass the model the CLI should run.",
        )
    return named


def _required_conversation(cli_input: CLIInput) -> CLIInput:
    """A run with no user turn still spawns the CLI and bills for it, so refuse before anything is spawned."""
    if not cli_input.conversation:
        raise AdapterError(
            status_code=400,
            message="The Claude Code CLI provider needs at least one user or assistant message to answer.",
        )
    return cli_input


def _request_options(optional_params: object) -> Mapping[str, object]:
    """The dispatcher hands this over untyped, so it becomes a typed mapping only once it validates."""
    if optional_params is None:
        return {}
    try:
        return _PARAMS_ADAPTER.validate_python(optional_params)
    except ValidationError:
        kind: Final = type(optional_params).__name__
        raise AdapterError(
            status_code=400,
            message=f"`optional_params` must be a mapping of option names to values, got {kind}.",
        ) from None


def _apply_request_options(
    settings: CLISettings,
    optional_params: Mapping[str, object],
    timeout: object,
) -> CLISettings:
    """Honour what the transport supports (`max_tokens`/`max_completion_tokens`, `timeout`); drop the rest.

    The CLI takes no argument for options like `temperature` or `response_format` that shape the answer
    itself, so there is nothing to honour them with. Clients also attach routing and bookkeeping fields
    (`user`, `prompt_cache_key`) to every request whether or not the caller asked for them. Both kinds are
    dropped the same way: silently, since neither can be applied to this transport. `tools` is read
    separately, by `read_client_tools`.

    A caller's cap can only raise the configured limit, never lower it: the CLI aborts the whole request
    with an error as soon as the reply passes the cap, so a small cap (clients send 10 for title requests)
    would turn a normal request into a 500.
    """
    supplied: Final = {name: value for name, value in optional_params.items() if value is not None}

    requested: Final = tuple(
        (name, _positive_int(name, supplied[name])) for name in _MAX_TOKENS_PARAMS if name in supplied
    )
    if len({limit for _, limit in requested}) > 1:
        raise AdapterError(
            status_code=400,
            message=(
                "`max_tokens` and `max_completion_tokens` were both supplied with different values "
                f"({', '.join(f'{name}={limit}' for name, limit in requested)}). Pass only one, since "
                "honouring either would mean silently discarding the other."
            ),
        )
    max_tokens: Final = max(requested[0][1], settings.max_output_tokens) if requested else settings.max_output_tokens
    return replace(
        settings,
        max_output_tokens=max_tokens,
        timeout_seconds=settings.timeout_seconds if timeout is None else _positive_float("timeout", timeout),
    )


def _tool_call(call: ToolCall) -> dict[str, object]:
    return {"id": call.id, "type": "function", "function": {"name": call.name, "arguments": call.arguments}}


def _session_id_from_user(user: object) -> str | None:
    """Claude Code's `user_id` is a JSON blob keyed by `session_id`, not a plain id.

    Every conversation on one machine shares the same `device_id`, so truncating that blob (what
    `prompt_cache_key` does) collides all of them onto one key. Reading `session_id` out of the
    untruncated JSON instead gives a real per-conversation key. Any shape that isn't this exact
    JSON object with a string `session_id` falls through to the caller's `prompt_cache_key`.

    Takes precedence over `prompt_cache_key` whenever both are present, since both are derived from
    the same `metadata.user_id` by the same bridge on this path — a real caller cannot set one without
    the other, so there is no live case where they intentionally disagree.
    """
    if not isinstance(user, str):
        return None
    try:
        parsed: Final = _PARAMS_ADAPTER.validate_json(user)
    except ValidationError:
        return None
    session_id: Final = parsed.get("session_id")
    return session_id if isinstance(session_id, str) and session_id else None


def _session_id_from_metadata(request_metadata: object) -> str | None:
    """`request_metadata["metadata"]["user_id"]` carries the same JSON blob, forwarded by the Anthropic bridge."""
    try:
        params: Final = _PARAMS_ADAPTER.validate_python(request_metadata)
    except ValidationError:
        return None
    try:
        metadata: Final = _PARAMS_ADAPTER.validate_python(params.get("metadata"))
    except ValidationError:
        return None
    return _session_id_from_user(metadata.get("user_id"))


def _fingerprint(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()[:16]


def _identity(system_prompt: str) -> str:
    """The prompt's opening line, which says what kind of request this is.

    Claude Code sends side requests (e.g. session titling) under the conversation's session_id with a
    different opening line, and appends per-turn notes to the end of the prompt, so the whole prompt
    changes every turn while the opening line does not. The billing header line varies by request kind.
    """
    return next(
        (
            line
            for line in system_prompt.splitlines()
            if line.strip() and not line.startswith("x-anthropic-billing-header")
        ),
        "",
    )


def _without_saved_reply(cli_input: CLIInput, session: ResumeSession) -> CLIInput:
    """The CLI already saved its own last reply, so the caller's echo of it is not sent again.

    A run stopped on a caller tool call was killed, not finished, so that turn is still sent.
    """
    conversation: Final = cli_input.conversation
    if not session.last_reply_saved or len(conversation) < 2 or conversation[0].role != "assistant":
        return cli_input
    return replace(cli_input, conversation=conversation[1:])


def _with_new_context(cli_input: CLIInput, new_context: str) -> CLIInput:
    """A resumed CLI replays its first-turn system prompt, so notes appended since reach the model here."""
    if not new_context.strip() or not cli_input.conversation:
        return cli_input
    *earlier, last = cli_input.conversation
    return replace(
        cli_input,
        conversation=(*earlier, replace(last, content=f"{new_context.strip()}\n\n{last.content}")),
    )


def _session_key(optional_params: Mapping[str, object], request_metadata: object) -> str | None:
    """`--session-id`/`--resume` is used only when the caller sent a per-conversation identifier."""
    if (from_metadata := _session_id_from_metadata(request_metadata)) is not None:
        return from_metadata
    key: Final = optional_params.get("prompt_cache_key")
    return key if isinstance(key, str) and key else None


class CompletionService:
    """Joins conversion, transport and the streaming fold. Owns no parsing or classification of its own."""

    def __init__(
        self,
        spawner: CLISpawner = spawn_subprocess,
        environment: Mapping[str, str] | None = None,
        sessions: SessionStore | None = None,
    ) -> None:
        self._spawner: Final = spawner
        self._environment: Final = environ if environment is None else environment
        self._sessions: Final = SessionStore() if sessions is None else sessions

    @property
    def sessions(self) -> SessionStore:
        """Exposed so the proxy can start the cleanup sweep against this instance's own store."""
        return self._sessions

    async def _prepare(
        self,
        model: str,
        messages: list[AllMessageValues],
        optional_params: object,
        timeout: object,
        request_metadata: object = None,
    ) -> _PreparedCall:
        params: Final = _request_options(optional_params)
        client_tools: Final = read_client_tools(params.get("tools"), params.get("tool_choice"))
        caller_key: Final = _session_key(params, request_metadata)
        system_prompt: Final = prepare_cli_input(messages).system_prompt
        session_key: Final = None if caller_key is None else f"{caller_key}:{_fingerprint(_identity(system_prompt))}"
        session: Final = (
            await self._sessions.resolve_session(session_key, system_prompt=system_prompt)
            if session_key is not None
            else None
        )
        skip_turns: Final = session.turns_already_sent if isinstance(session, ResumeSession) else 0
        unsent: Final = _required_conversation(prepare_cli_input(messages, skip_turns=skip_turns))
        cli_input: Final = (
            _with_new_context(_without_saved_reply(unsent, session), system_prompt[len(session.seen_system_prompt) :])
            if isinstance(session, ResumeSession)
            else unsent
        )
        return _PreparedCall(
            settings=_apply_request_options(read_settings(self._environment), params, timeout),
            cli_input=replace(cli_input, system_prompt=with_caller_machine_note(cli_input.system_prompt))
            if client_tools
            else cli_input,
            model=_required_model(model),
            client_tools=client_tools,
            session=session,
            session_key=session_key,
            turns_sent=skip_turns + len(unsent.conversation),
            system_prompt=system_prompt,
        )

    async def _fold_run(
        self, call: _PreparedCall, on_started: Callable[[float], None] | None = None
    ) -> AsyncGenerator[tuple[StreamState, tuple[GenericStreamingChunk, ...]], None]:
        """Both paths fold the same run and stop it the moment the turn ends on the caller's tools.

        The stub server never answers a call, so the CLI would otherwise sit there until the timeout.
        Closing the event stream kills and awaits the child.
        """
        state: StreamState = StreamState(client_tools=call.client_tools)  # rebind-ok: a fold over a stream
        failure = None  # rebind-ok: set once if the transport reports one, raised after the drain
        events: Final = stream_cli(
            spawner=self._spawner,
            settings=call.settings,
            environment=self._environment,
            model=call.model,
            system_prompt=call.cli_input.system_prompt,
            conversation=call.cli_input.conversation,
            client_tools=call.client_tools,
            session=call.session,
            on_started=on_started,
        )
        try:
            async for item in events:
                if not isinstance(item, Mapping):
                    failure = item
                    break
                state, chunks = advance(state, item)
                yield state, chunks
                if state.awaiting_caller:
                    break
        finally:
            await events.aclose()
        if failure is not None:
            if call.session_key is not None and isinstance(call.session, NewSession):
                await self._sessions.discard(call.session_key, call.session.session_id)
            raise raise_public(failure)
        if call.session_key is not None:
            await self._sessions.touch(
                call.session_key,
                turns_sent=call.turns_sent,
                seen_system_prompt=call.system_prompt,
                last_reply_saved=not state.awaiting_caller,
            )

    async def acompletion(
        self,
        model: str,
        messages: list[AllMessageValues],
        *,
        optional_params: object = None,
        timeout: object = None,
        request_metadata: object = None,
        on_started: Callable[[float], None] | None = None,
    ) -> ModelResponse:
        call: Final = await self._prepare(model, messages, optional_params, timeout, request_metadata)
        state: StreamState = StreamState(client_tools=call.client_tools)  # rebind-ok: last state of the fold
        async with aclosing(self._fold_run(call, on_started)) as runs:
            async for state, _ in runs:  # only the final state is needed
                pass
        answer: Final = answer_of(state)
        return ModelResponse(
            model=call.model,
            choices=[
                Choices(
                    finish_reason=answer.finish_reason,
                    index=0,
                    message=Message(
                        content=answer.text or (None if answer.tool_calls else ""),
                        tool_calls=[_tool_call(tool_call) for tool_call in answer.tool_calls] or None,
                    ),
                )
            ],
            usage=answer.usage,
        )

    async def astreaming(  # pyright: ignore[reportIncompatibleMethodOverride]  # base shape is unawaitable here
        self,
        model: str,
        messages: list[AllMessageValues],
        *,
        optional_params: object = None,
        timeout: object = None,
        request_metadata: object = None,
        on_started: Callable[[float], None] | None = None,
    ) -> AsyncIterator[GenericStreamingChunk]:
        """Stream folded events, closing the subprocess when iteration ends."""
        call: Final = await self._prepare(model, messages, optional_params, timeout, request_metadata)
        state: StreamState = StreamState(client_tools=call.client_tools)  # rebind-ok: last state of the fold
        # A client that stops reading mid-answer must still get the CLI child killed, not left to GC.
        async with aclosing(self._fold_run(call, on_started)) as runs:
            async for state, chunks in runs:  # the final state closes the stream
                for chunk in chunks:
                    yield chunk
        yield close_stream(state)
