import asyncio
import json
from pathlib import Path

import pytest

from claude_subscription_adapter.errors import (
    CLIFailure,
    CLINotAuthenticated,
    CLINotInstalled,
    PromptTooLong,
    QueueFull,
    RateLimited,
    TimedOut,
    UnexpectedExit,
)
from claude_subscription_adapter.schemas import AdapterError as CustomLLMError
from claude_subscription_adapter.service import CompletionService as ClaudeCodeCLI
from claude_subscription_adapter.sessions import SessionStore
from claude_subscription_adapter.streaming import assemble_answer, stream_chunks
from claude_subscription_adapter.transport import (
    DEFAULT_MAX_CONCURRENCY,
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_TIMEOUT_SECONDS,
)

_FIXTURES = Path(__file__).parents[1] / "fixtures"


def _lines(name):
    """Replay a run recorded from the real CLI, so no handler test invents CLI output."""
    text = (_FIXTURES / name).read_text(encoding="utf-8")
    return [f"{line}\n".encode() for line in text.splitlines() if line.strip()]


def _events(name):
    return tuple(json.loads(line) for line in _lines(name))


DELTA_LINES = _lines("delta_run.ndjson")
DELTA_RUN = _events("delta_run.ndjson")
SNAPSHOT_RUN = _events("snapshot_run.ndjson")

MESSAGES = [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}]


class FakeProcess:
    def __init__(self, *, lines=(), stderr=b"", exit_code=0, on_spawn=None, on_exit=None, delay=0.0):
        self._lines = list(lines)
        self._stderr = stderr
        self._exit_code = exit_code
        self._on_exit = on_exit
        self._delay = delay
        self.returncode = None
        self.stdin_payload = None
        if on_spawn is not None:
            on_spawn()

    async def write_stdin(self, payload):
        self.stdin_payload = payload

    async def read_stdout_line(self):
        if self._delay:
            await asyncio.sleep(self._delay)
        return self._lines.pop(0) if self._lines else b""

    async def read_stderr(self):
        return self._stderr

    def kill(self):
        self.returncode = -9
        if self._on_exit is not None:
            self._on_exit()

    async def wait(self):
        if self.returncode is None:
            self.returncode = self._exit_code
            if self._on_exit is not None:
                self._on_exit()
        return self.returncode


class FakeSpawner:
    def __init__(self, *, lines=(), stderr=b"", exit_code=0, error=None, factory=None):
        self._lines = lines
        self._stderr = stderr
        self._exit_code = exit_code
        self._error = error
        self._factory = factory
        self.calls = []

    async def __call__(self, *, executable, args, env, cwd):
        self.calls.append({"executable": executable, "args": tuple(args), "env": dict(env), "cwd": cwd})
        if self._error is not None:
            raise self._error
        if self._factory is not None:
            return self._factory()
        return FakeProcess(lines=list(self._lines), stderr=self._stderr, exit_code=self._exit_code)


_ENVIRONMENT = {
    "CLAUDE_CODE_PATH": "/fake/claude",
    "CLAUDE_CODE_CWD": "/fake/cwd",
    "CLAUDE_CODE_TIMEOUT_SECONDS": "5",
}


def _handler(spawner, environment=None):
    return ClaudeCodeCLI(spawner=spawner, environment=_ENVIRONMENT if environment is None else environment)


async def _collect(handler, *, model="claude-sonnet-4-5", messages=None, **kwargs):
    """Iterate WITHOUT awaiting the call, exactly as the dispatcher does."""
    turns = MESSAGES if messages is None else messages
    return [chunk async for chunk in handler.astreaming(model=model, messages=turns, **kwargs)]


# --------------------------------------------------------------------------------------
# The shape the dispatcher requires
# --------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------
# Answers
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_non_streaming_returns_the_assembled_text_finish_reason_and_usage():
    response = await _handler(FakeSpawner(lines=DELTA_LINES)).acompletion(model="claude-sonnet-4-5", messages=MESSAGES)
    expected = assemble_answer(DELTA_RUN)

    assert expected.text
    assert response.choices[0].message.content == expected.text
    assert response.choices[0].finish_reason == expected.finish_reason
    assert response.usage.prompt_tokens == expected.usage.prompt_tokens
    assert response.usage.completion_tokens == expected.usage.completion_tokens
    assert response.usage.total_tokens == expected.usage.total_tokens


@pytest.mark.asyncio
async def test_non_streaming_usage_keeps_the_cache_counts_the_run_reported():
    response = await _handler(FakeSpawner(lines=DELTA_LINES)).acompletion(model="claude-sonnet-4-5", messages=MESSAGES)
    recorded = next(event["usage"] for event in DELTA_RUN if event.get("type") == "result")

    assert recorded["cache_read_input_tokens"] > 0
    assert response.usage.cache_read_input_tokens == recorded["cache_read_input_tokens"]
    assert response.usage.cache_creation_input_tokens == recorded["cache_creation_input_tokens"]


@pytest.mark.asyncio
async def test_streaming_yields_chunks_in_arrival_order_with_only_the_last_finished():
    chunks = await _collect(_handler(FakeSpawner(lines=DELTA_LINES)))

    assert [chunk["is_finished"] for chunk in chunks] == [False] * (len(chunks) - 1) + [True]
    assert [chunk["usage"] is not None for chunk in chunks] == [False] * (len(chunks) - 1) + [True]
    assert tuple(chunk["text"] for chunk in chunks) == tuple(chunk["text"] for chunk in stream_chunks(DELTA_RUN))


@pytest.mark.asyncio
async def test_the_streamed_text_and_usage_match_the_non_streaming_answer_for_the_same_run():
    handler = _handler(FakeSpawner(lines=DELTA_LINES))

    chunks = await _collect(handler)
    response = await handler.acompletion(model="claude-sonnet-4-5", messages=MESSAGES)

    assert "".join(chunk["text"] for chunk in chunks) == response.choices[0].message.content
    assert chunks[-1]["usage"]["prompt_tokens"] == response.usage.prompt_tokens
    assert chunks[-1]["usage"]["completion_tokens"] == response.usage.completion_tokens
    assert chunks[-1]["finish_reason"] == response.choices[0].finish_reason


@pytest.mark.asyncio
async def test_a_snapshot_only_run_streams_and_assembles_the_same_answer():
    handler = _handler(FakeSpawner(lines=_lines("snapshot_run.ndjson")))

    chunks = await _collect(handler)
    response = await handler.acompletion(model="claude-sonnet-4-5", messages=MESSAGES)

    assert assemble_answer(SNAPSHOT_RUN).text
    assert "".join(chunk["text"] for chunk in chunks) == response.choices[0].message.content


@pytest.mark.asyncio
async def test_an_empty_run_still_returns_one_closing_chunk_and_an_empty_answer():
    handler = _handler(FakeSpawner(lines=_lines("empty_run.ndjson")))

    chunks = await _collect(handler)
    response = await handler.acompletion(model="claude-sonnet-4-5", messages=MESSAGES)

    assert len(chunks) == 1 and chunks[0]["is_finished"]
    assert response.choices[0].message.content == ""


# --------------------------------------------------------------------------------------
# Model passthrough
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["claude-sonnet-9-9", "some-unreleased-name", "claude-opus-4-1"])
async def test_any_model_string_reaches_the_transport_untouched(model):
    spawner = FakeSpawner(lines=DELTA_LINES)

    response = await _handler(spawner).acompletion(model=model, messages=MESSAGES)

    args = spawner.calls[0]["args"]
    assert args[args.index("--model") + 1] == model
    assert response.model == model


@pytest.mark.asyncio
async def test_the_streaming_path_passes_the_model_through_too():
    spawner = FakeSpawner(lines=DELTA_LINES)

    await _collect(_handler(spawner), model="totally-made-up-model")

    args = spawner.calls[0]["args"]
    assert args[args.index("--model") + 1] == "totally-made-up-model"


@pytest.mark.asyncio
async def test_the_system_and_user_messages_are_split_before_reaching_the_transport():
    spawner = FakeSpawner(lines=DELTA_LINES)

    await _handler(spawner).acompletion(model="claude-sonnet-4-5", messages=MESSAGES)

    args = spawner.calls[0]["args"]
    assert args[args.index("--system-prompt") + 1] == "be brief"


# --------------------------------------------------------------------------------------
# Web search is unconditional
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_web_search_is_always_pre_approved_on_the_non_streaming_path():
    spawner = FakeSpawner(lines=DELTA_LINES)

    await _handler(spawner).acompletion(model="m", messages=MESSAGES)

    args = spawner.calls[0]["args"]
    assert args[args.index("--allowedTools") + 1] == "WebSearch"


@pytest.mark.asyncio
async def test_web_search_is_always_pre_approved_on_the_streaming_path():
    spawner = FakeSpawner(lines=DELTA_LINES)

    await _collect(_handler(spawner))

    args = spawner.calls[0]["args"]
    assert args[args.index("--allowedTools") + 1] == "WebSearch"


# --------------------------------------------------------------------------------------
# Request options: honoured or refused, never silently dropped
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["max_tokens", "max_completion_tokens"])
async def test_a_caller_supplied_max_tokens_reaches_the_child_environment(name):
    spawner = FakeSpawner(lines=DELTA_LINES)

    await _handler(spawner).acompletion(model="m", messages=MESSAGES, optional_params={name: 64000})

    assert spawner.calls[0]["env"]["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] == "64000"


@pytest.mark.asyncio
@pytest.mark.parametrize("small_cap", [1, 10, DEFAULT_MAX_OUTPUT_TOKENS - 1])
async def test_a_cap_below_the_default_is_raised_to_the_default_so_the_cli_never_aborts(small_cap):
    spawner = FakeSpawner(lines=DELTA_LINES)

    await _handler(spawner).acompletion(model="m", messages=MESSAGES, optional_params={"max_tokens": small_cap})

    assert spawner.calls[0]["env"]["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] == str(DEFAULT_MAX_OUTPUT_TOKENS)


@pytest.mark.asyncio
async def test_a_small_cap_is_raised_on_the_streaming_path_too():
    spawner = FakeSpawner(lines=DELTA_LINES)

    await _collect(_handler(spawner), optional_params={"max_tokens": 10})

    assert spawner.calls[0]["env"]["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] == str(DEFAULT_MAX_OUTPUT_TOKENS)


@pytest.mark.asyncio
async def test_max_tokens_is_honoured_on_the_streaming_path_too():
    spawner = FakeSpawner(lines=DELTA_LINES)

    await _collect(_handler(spawner), optional_params={"max_tokens": 77000})

    assert spawner.calls[0]["env"]["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] == "77000"


@pytest.mark.asyncio
async def test_no_max_tokens_leaves_the_configured_default_in_place():
    """Without this, flipping the fallback to something else would pass unnoticed."""
    spawner = FakeSpawner(lines=DELTA_LINES)

    await _handler(spawner).acompletion(model="m", messages=MESSAGES)

    assert spawner.calls[0]["env"]["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] == str(DEFAULT_MAX_OUTPUT_TOKENS)


@pytest.mark.asyncio
async def test_a_per_request_timeout_replaces_the_configured_one():
    """The transport reports the budget it actually used, so a wrong one shows up in the 504 text."""

    def hang():
        return FakeProcess(delay=3600)

    with pytest.raises(CustomLLMError) as caught:
        await _handler(FakeSpawner(factory=hang)).acompletion(model="m", messages=MESSAGES, timeout=0.05)

    assert "0.05s" in caught.value.message


@pytest.mark.asyncio
async def test_no_per_request_timeout_falls_back_to_the_environment_value():
    def hang():
        return FakeProcess(delay=3600)

    with pytest.raises(CustomLLMError) as caught:
        await _handler(FakeSpawner(factory=hang), {"CLAUDE_CODE_TIMEOUT_SECONDS": "0.04"}).acompletion(
            model="m", messages=MESSAGES
        )

    assert "0.04s" in caught.value.message


def test_the_unset_timeout_default_is_the_documented_one():
    """A test that never omits the argument cannot notice the default changing."""
    assert DEFAULT_TIMEOUT_SECONDS == 600.0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "option",
    [{"temperature": 0.7}, {"top_p": 0.9}, {"stop": ["\n"]}, {"response_format": {"type": "json"}}],
)
async def test_an_option_the_cli_cannot_honour_is_dropped_rather_than_refused(option):
    """The CLI has no argument for these, so the request is answered with the option silently dropped."""
    response = await _handler(FakeSpawner(lines=DELTA_LINES)).acompletion(
        model="m", messages=MESSAGES, optional_params=option
    )

    assert response.choices[0].message.content


@pytest.mark.asyncio
async def test_an_unhonourable_option_is_dropped_on_the_streaming_path_too():
    chunks = await _collect(_handler(FakeSpawner(lines=DELTA_LINES)), optional_params={"temperature": 0.1})

    assert chunks


@pytest.mark.asyncio
@pytest.mark.parametrize("tools", [[], [{"type": "function", "function": {"name": "noop", "parameters": {}}}]])
async def test_offered_tools_the_model_does_not_call_leave_a_plain_text_answer(tools):
    response = await _handler(FakeSpawner(lines=DELTA_LINES)).acompletion(
        model="m", messages=MESSAGES, optional_params={"tools": tools}
    )

    assert response.choices[0].message.content == assemble_answer(DELTA_RUN).text
    assert response.choices[0].finish_reason == "stop"
    assert not response.choices[0].message.tool_calls


CLIENT_TOOL_LINES = _lines("client_tool_run.ndjson")
CREATE_FILE = {
    "type": "function",
    "function": {
        "name": "create_file",
        "description": "Create a file",
        "parameters": {"type": "object", "properties": {"filePath": {"type": "string"}}},
    },
}
EXPECTED_CALLS = [
    ("toolu_01CLIENTTOOLPLACEHOLDER01", "create_file", {"filePath": "a.txt", "content": "a"}),
    ("toolu_01CLIENTTOOLPLACEHOLDER02", "create_file", {"filePath": "b.txt", "content": "b"}),
]


class _Recording:
    """Spawns the fixture replay and keeps the process, so a test can see it was killed, not left running."""

    def __init__(self, lines):
        self.processes = []
        self.spawner = FakeSpawner(factory=self._spawn)
        self._lines = lines

    def _spawn(self):
        process = FakeProcess(lines=list(self._lines))
        self.processes.append(process)
        return process


@pytest.mark.asyncio
async def test_a_caller_tool_call_comes_back_as_tool_calls_when_not_streaming_and_the_cli_is_stopped():
    recording = _Recording(CLIENT_TOOL_LINES)
    response = await _handler(recording.spawner).acompletion(
        model="m", messages=MESSAGES, optional_params={"tools": [CREATE_FILE]}
    )

    choice = response.choices[0]
    assert choice.finish_reason == "tool_calls"
    assert choice.message.content == "On it."
    assert [(c.id, c.function.name, json.loads(c.function.arguments)) for c in choice.message.tool_calls] == (
        EXPECTED_CALLS
    )
    assert response.usage.completion_tokens == 367
    assert recording.processes[0].returncode == -9


@pytest.mark.asyncio
async def test_a_caller_tool_call_streams_one_chunk_per_call_then_closes_with_tool_calls():
    recording = _Recording(CLIENT_TOOL_LINES)
    chunks = await _collect(_handler(recording.spawner), optional_params={"tools": [CREATE_FILE]})

    calls = [chunk["tool_use"] for chunk in chunks if chunk.get("tool_use")]
    assert [(c["id"], c["function"]["name"], json.loads(c["function"]["arguments"]), c["index"]) for c in calls] == [
        (*call, index) for index, call in enumerate(EXPECTED_CALLS)
    ]
    assert chunks[-1]["is_finished"]
    assert chunks[-1]["finish_reason"] == "tool_calls"
    assert chunks[-1]["usage"]["completion_tokens"] == 367
    assert recording.processes[0].returncode == -9


@pytest.mark.asyncio
async def test_caller_tools_reach_the_cli_through_the_stub_server():
    spawner = FakeSpawner(lines=DELTA_LINES)
    await _handler(spawner).acompletion(model="m", messages=MESSAGES, optional_params={"tools": [CREATE_FILE]})

    args = spawner.calls[0]["args"]
    config = json.loads(args[args.index("--mcp-config") + 1])
    assert "mcp__c" in args[args.index("--allowedTools") + 1 : args.index("--max-turns")]
    assert config["mcpServers"]["c"]["args"][0].endswith("client_tools_server.py")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "optional_params",
    [{}, {"tools": None}, {"tools": [CREATE_FILE], "tool_choice": "none"}],
    ids=["no-tools", "null-tools", "tool-choice-none"],
)
async def test_no_stub_server_is_started_when_no_tools_are_offered(optional_params):
    spawner = FakeSpawner(lines=DELTA_LINES)
    await _handler(spawner).acompletion(model="m", messages=MESSAGES, optional_params=optional_params)

    assert "--mcp-config" not in spawner.calls[0]["args"]
    assert spawner.calls[0]["args"][spawner.calls[0]["args"].index("--tools") + 1] == "WebSearch"


@pytest.mark.asyncio
async def test_routing_and_bookkeeping_options_are_ignored_rather_than_refused():
    """Claude Code sends these on every request unasked, and dropping them changes no answer."""
    response = await _handler(FakeSpawner(lines=DELTA_LINES)).acompletion(
        model="m",
        messages=MESSAGES,
        optional_params={
            "user": '{"device_id":"abc","session_id":"def"}',
            "prompt_cache_key": "some-cache-key",
            "prompt_cache_retention": "24h",
            "tool_choice": "auto",
            "parallel_tool_calls": True,
            "metadata": {"trace": "1"},
            "store": False,
            "service_tier": "default",
            "safety_identifier": "sid",
        },
    )

    assert response.choices[0].message.content


@pytest.mark.asyncio
async def test_an_answer_shaping_option_is_dropped_alongside_ignored_ones():
    """An answer-shaping option dropped alongside routing ones must not stop the request from answering."""
    response = await _handler(FakeSpawner(lines=DELTA_LINES)).acompletion(
        model="m",
        messages=MESSAGES,
        optional_params={"prompt_cache_key": "k", "tools": [], "temperature": 0.7},
    )

    assert response.choices[0].message.content


@pytest.mark.asyncio
async def test_routing_flags_and_absent_options_are_not_mistaken_for_unsupported_ones():
    """`stream` selects the entry point rather than changing the request, so refusing it breaks streaming."""
    spawner = FakeSpawner(lines=DELTA_LINES)
    params = {"stream": True, "stream_options": None, "temperature": None, "max_tokens": 64000}

    await _handler(spawner).acompletion(model="m", messages=MESSAGES, optional_params=params)

    assert spawner.calls[0]["env"]["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] == "64000"


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", [0, -5, 2.5e400, "many", True])
async def test_a_nonsensical_max_tokens_is_refused_with_a_clear_message(bad):
    with pytest.raises(CustomLLMError) as caught:
        await _handler(FakeSpawner()).acompletion(model="m", messages=MESSAGES, optional_params={"max_tokens": bad})

    assert caught.value.status_code == 400
    assert "`max_tokens`" in caught.value.message


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", [0, -1, "soon"])
async def test_a_nonsensical_timeout_is_refused_with_a_clear_message(bad):
    with pytest.raises(CustomLLMError) as caught:
        await _handler(FakeSpawner()).acompletion(model="m", messages=MESSAGES, timeout=bad)

    assert caught.value.status_code == 400
    assert "`timeout`" in caught.value.message


# --------------------------------------------------------------------------------------
# Failures
# --------------------------------------------------------------------------------------

_FAILURES = (
    (
        {"error": FileNotFoundError(2, "No such file")},
        CLINotInstalled,
        503,
        ("not installed", "/fake/claude", "CLAUDE_CODE_PATH"),
    ),
    ({"stderr": b"invalid api key", "exit_code": 1}, CLINotAuthenticated, 503, ("not authenticated", "log in")),
    ({"stderr": b"rate limit exceeded", "exit_code": 1}, RateLimited, 429, ("rate limited", "rate limit exceeded")),
    ({"stderr": b"prompt is too long", "exit_code": 1}, PromptTooLong, 400, ("too long",)),
    ({"stderr": b"segfault", "exit_code": 139}, UnexpectedExit, 502, ("exited with code 139", "segfault")),
)


@pytest.mark.asyncio
@pytest.mark.parametrize(("spawn", "failure", "status_code", "phrases"), _FAILURES)
async def test_every_failure_variant_becomes_its_own_public_error_when_not_streaming(
    spawn, failure, status_code, phrases
):
    with pytest.raises(CustomLLMError) as caught:
        await _handler(FakeSpawner(**spawn)).acompletion(model="m", messages=MESSAGES)

    assert caught.value.status_code == status_code
    for phrase in phrases:
        assert phrase in caught.value.message


@pytest.mark.asyncio
@pytest.mark.parametrize(("spawn", "failure", "status_code", "phrases"), _FAILURES)
async def test_every_failure_variant_becomes_its_own_public_error_when_streaming(spawn, failure, status_code, phrases):
    with pytest.raises(CustomLLMError) as caught:
        await _collect(_handler(FakeSpawner(**spawn)))

    assert caught.value.status_code == status_code
    for phrase in phrases:
        assert phrase in caught.value.message


@pytest.mark.asyncio
async def test_the_timeout_variant_becomes_a_504_naming_the_budget_on_both_paths():
    def hang():
        return FakeProcess(delay=3600)

    slow = {"CLAUDE_CODE_TIMEOUT_SECONDS": "0.02"}

    with pytest.raises(CustomLLMError) as non_streaming:
        await _handler(FakeSpawner(factory=hang), slow).acompletion(model="m", messages=MESSAGES)
    with pytest.raises(CustomLLMError) as streaming:
        await _collect(_handler(FakeSpawner(factory=hang), slow), model="m")

    for caught in (non_streaming, streaming):
        assert caught.value.status_code == 504
        assert "timed out after 0.02s" in caught.value.message


@pytest.mark.asyncio
async def test_a_request_arriving_at_a_full_line_gets_a_busy_429_and_spawns_nothing():
    spawner = FakeSpawner(factory=lambda: FakeProcess(delay=0.3))
    handler = _handler(spawner, {"CLAUDE_CODE_MAX_CONCURRENCY": "1", "CLAUDE_CODE_MAX_QUEUE_LENGTH": "1"})
    admitted = [asyncio.create_task(handler.acompletion(model="m", messages=MESSAGES)) for _ in range(2)]
    await asyncio.sleep(0.05)

    with pytest.raises(CustomLLMError) as caught:
        await handler.acompletion(model="m", messages=MESSAGES)

    assert caught.value.status_code == 429
    assert "busy on the proxy server" in caught.value.message
    assert len(spawner.calls) == 1
    await asyncio.gather(*admitted)


def test_every_failure_variant_is_exercised_by_the_public_error_tests():
    """A new `CLIFailure` member must fail this list rather than quietly reach callers as a generic 500."""
    covered = {failure for _, failure, _, _ in _FAILURES} | {TimedOut, QueueFull}

    assert covered == set(CLIFailure.__args__)


@pytest.mark.asyncio
async def test_a_streaming_failure_raises_only_after_the_earlier_chunks_were_delivered():
    """A truncated answer with no error would look like a complete one to the caller."""
    spawner = FakeSpawner(lines=DELTA_LINES, stderr=b"exploded", exit_code=1)
    seen = []

    with pytest.raises(CustomLLMError) as caught:
        async for chunk in _handler(spawner).astreaming(model="claude-sonnet-4-5", messages=MESSAGES):
            seen.append(chunk)

    assert seen and all(chunk["text"] for chunk in seen)
    assert not any(chunk["is_finished"] for chunk in seen)
    assert "exploded" in caught.value.message


# --------------------------------------------------------------------------------------
# Concurrency and settings
# --------------------------------------------------------------------------------------


async def _peak_live_processes(*, environment, calls):
    live = {"now": 0, "peak": 0}

    def entered():
        live["now"] += 1
        live["peak"] = max(live["peak"], live["now"])

    def left():
        live["now"] -= 1

    def make():
        return FakeProcess(on_spawn=entered, on_exit=left, delay=0.01)

    handler = ClaudeCodeCLI(spawner=FakeSpawner(factory=make), environment=environment)
    await asyncio.gather(*(handler.acompletion(model="m", messages=MESSAGES) for _ in range(calls)))
    return live["peak"]


@pytest.mark.asyncio
async def test_a_configured_cap_holds_back_calls_beyond_it():
    peak = await _peak_live_processes(environment={"CLAUDE_CODE_MAX_CONCURRENCY": "2"}, calls=10)

    assert peak <= 2


@pytest.mark.asyncio
async def test_no_configured_cap_still_applies_a_finite_one():
    """Unlimited concurrency here forks one CLI per request and takes the proxy host down."""
    peak = await _peak_live_processes(environment={}, calls=DEFAULT_MAX_CONCURRENCY * 4)

    assert peak <= DEFAULT_MAX_CONCURRENCY


@pytest.mark.asyncio
async def test_the_injected_environment_supplies_the_executable_and_working_directory():
    spawner = FakeSpawner(lines=DELTA_LINES)

    await _handler(spawner).acompletion(model="claude-sonnet-4-5", messages=MESSAGES)

    call = spawner.calls[0]
    assert call["executable"] == "/fake/claude"
    assert call["cwd"] == "/fake/cwd"


@pytest.mark.asyncio
async def test_nothing_from_the_real_process_environment_leaks_into_the_child():
    spawner = FakeSpawner(lines=DELTA_LINES)

    await _handler(spawner, {"ONLY_KEY": "only-value"}).acompletion(model="m", messages=MESSAGES)

    env = spawner.calls[0]["env"]
    assert env["ONLY_KEY"] == "only-value"
    assert "PATH" not in env


def test_the_default_environment_is_the_real_process_environment():
    """Defaulting to an empty mapping would silently ignore every operator override in production."""
    from os import environ

    assert ClaudeCodeCLI(spawner=FakeSpawner())._environment is environ


# --------------------------------------------------------------------------------------
# Options LiteLLM injects itself, seen through the real dispatcher
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_two_conflicting_token_limits_are_refused_rather_than_one_being_discarded():
    """Taking the first and dropping the other is the silent drop this story set out to eliminate.

    LiteLLM collapses the pair before the handler sees it, so the conflict only reaches us on a direct
    call. The guard still has to hold there, since nothing else would catch a genuine disagreement.
    """
    spawner = FakeSpawner(lines=DELTA_LINES)
    both = {"max_tokens": 100, "max_completion_tokens": 200}

    with pytest.raises(CustomLLMError) as caught:
        await _handler(spawner).acompletion(model="m", messages=MESSAGES, optional_params=both)

    assert caught.value.status_code == 400
    assert "100" in caught.value.message and "200" in caught.value.message
    assert spawner.calls == []


@pytest.mark.asyncio
async def test_conflicting_token_limits_are_refused_on_the_streaming_path_too():
    both = {"max_tokens": 8, "max_completion_tokens": 9}

    with pytest.raises(CustomLLMError) as caught:
        await _collect(_handler(FakeSpawner(lines=DELTA_LINES)), optional_params=both)

    assert caught.value.status_code == 400


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", [3.9, 0.5, -2.5])
async def test_a_non_integral_token_limit_is_refused_rather_than_truncated(bad):
    """3.9 quietly becoming 3 hands back fewer tokens than asked for with no indication why."""
    with pytest.raises(CustomLLMError) as caught:
        await _handler(FakeSpawner()).acompletion(model="m", messages=MESSAGES, optional_params={"max_tokens": bad})

    assert caught.value.status_code == 400
    assert "whole number" in caught.value.message


@pytest.mark.asyncio
async def test_a_whole_number_token_limit_given_as_a_float_is_accepted():
    """`100.0` names a whole number of tokens, so refusing it would reject a valid request."""
    spawner = FakeSpawner(lines=DELTA_LINES)

    await _handler(spawner).acompletion(model="m", messages=MESSAGES, optional_params={"max_tokens": 100000.0})

    assert spawner.calls[0]["env"]["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] == "100000"


@pytest.mark.asyncio
@pytest.mark.parametrize("blank", ["", "   ", "\t"])
async def test_a_blank_model_is_refused_before_the_cli_is_spawned(blank):
    """Otherwise the CLI is launched with `--model ''` and fails late as an opaque error."""
    spawner = FakeSpawner(lines=DELTA_LINES)

    with pytest.raises(CustomLLMError) as caught:
        await _handler(spawner).acompletion(model=blank, messages=MESSAGES)

    assert caught.value.status_code == 400
    assert "No model was named" in caught.value.message
    assert spawner.calls == []


@pytest.mark.asyncio
async def test_a_blank_model_is_refused_on_the_streaming_path_too():
    spawner = FakeSpawner(lines=DELTA_LINES)

    with pytest.raises(CustomLLMError) as caught:
        await _collect(_handler(spawner), model="  ")

    assert caught.value.status_code == 400
    assert spawner.calls == []


@pytest.mark.asyncio
async def test_surrounding_whitespace_is_trimmed_but_the_model_name_is_otherwise_untouched():
    spawner = FakeSpawner(lines=DELTA_LINES)

    await _handler(spawner).acompletion(model="  some-unreleased-name  ", messages=MESSAGES)

    args = spawner.calls[0]["args"]
    assert args[args.index("--model") + 1] == "some-unreleased-name"


@pytest.mark.asyncio
@pytest.mark.parametrize("messages", [[], [{"role": "system", "content": "be brief"}]])
async def test_a_request_with_no_user_turn_is_refused_before_the_cli_is_spawned(messages):
    """Spawning here bills a full CLI run for a request that carries nothing to answer."""
    spawner = FakeSpawner(lines=DELTA_LINES)
    handler = _handler(spawner)

    with pytest.raises(CustomLLMError) as non_streaming:
        await handler.acompletion(model="m", messages=messages)

    with pytest.raises(CustomLLMError) as streaming:
        await _collect(handler, messages=messages)

    assert non_streaming.value.status_code == 400
    assert streaming.value.status_code == 400
    assert spawner.calls == []


@pytest.mark.asyncio
async def test_a_client_that_stops_reading_mid_stream_gets_the_cli_child_killed():
    """Copilot cancels answers all the time; the child must not keep running on the proxy server."""
    recording = _Recording(DELTA_LINES)
    stream = _handler(recording.spawner).astreaming(model="m", messages=MESSAGES)

    await anext(stream)
    await stream.aclose()

    assert recording.processes[0].returncode == -9


@pytest.mark.asyncio
async def test_with_caller_tools_claude_is_told_paths_belong_to_the_callers_machine():
    spawner = FakeSpawner(lines=DELTA_LINES)
    await _handler(spawner).acompletion(model="m", messages=MESSAGES, optional_params={"tools": [CREATE_FILE]})

    args = spawner.calls[0]["args"]
    prompt = args[args.index("--system-prompt") + 1]
    assert prompt.startswith("be brief\n\n")
    assert "the user's own computer" in prompt


@pytest.mark.asyncio
async def test_without_caller_tools_the_system_prompt_is_passed_unchanged():
    spawner = FakeSpawner(lines=DELTA_LINES)
    await _handler(spawner).acompletion(model="m", messages=MESSAGES)

    args = spawner.calls[0]["args"]
    assert args[args.index("--system-prompt") + 1] == "be brief"


# --------------------------------------------------------------------------------------
# Session resume, keyed on prompt_cache_key
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_request_with_no_prompt_cache_key_stays_stateless():
    spawner = FakeSpawner(lines=DELTA_LINES)

    await _handler(spawner).acompletion(model="m", messages=MESSAGES)

    args = spawner.calls[0]["args"]
    assert "--no-session-persistence" in args
    assert "--session-id" not in args
    assert "--resume" not in args


@pytest.mark.asyncio
async def test_the_first_request_for_a_cache_key_starts_a_new_session_with_full_history():
    spawner = FakeSpawner(lines=DELTA_LINES)

    await _handler(spawner).acompletion(model="m", messages=MESSAGES, optional_params={"prompt_cache_key": "caller-a"})

    args = spawner.calls[0]["args"]
    assert "--session-id" in args
    assert "--resume" not in args
    assert "--no-session-persistence" not in args


@pytest.mark.asyncio
async def test_a_repeat_request_for_the_same_cache_key_resumes_instead_of_starting_fresh():
    handler = _handler(FakeSpawner(lines=DELTA_LINES))
    params = {"prompt_cache_key": "caller-a"}

    await handler.acompletion(model="m", messages=MESSAGES, optional_params=params)

    followup = [*MESSAGES, {"role": "assistant", "content": "hi there"}, {"role": "user", "content": "and then?"}]
    second_spawner = FakeSpawner(lines=DELTA_LINES)
    handler._spawner = second_spawner
    await handler.acompletion(model="m", messages=followup, optional_params=params)

    args = second_spawner.calls[0]["args"]
    assert "--resume" in args
    assert "--session-id" not in args
    assert "--no-session-persistence" not in args


@pytest.mark.asyncio
async def test_a_retry_after_a_failed_first_run_starts_a_new_session_instead_of_resuming():
    handler = _handler(FakeSpawner(stderr=b"segfault", exit_code=139))
    params = {"prompt_cache_key": "caller-a"}
    with pytest.raises(CustomLLMError):
        await handler.acompletion(model="m", messages=MESSAGES, optional_params=params)

    retry_spawner = FakeSpawner(lines=DELTA_LINES)
    handler._spawner = retry_spawner
    await handler.acompletion(model="m", messages=MESSAGES, optional_params=params)

    args = retry_spawner.calls[0]["args"]
    assert "--session-id" in args
    assert "--resume" not in args


@pytest.mark.asyncio
async def test_the_resumed_session_id_matches_the_first_requests_new_session_id():
    handler = _handler(FakeSpawner(lines=DELTA_LINES))
    params = {"prompt_cache_key": "caller-a"}

    first_spawner = handler._spawner
    await handler.acompletion(model="m", messages=MESSAGES, optional_params=params)
    first_args = first_spawner.calls[0]["args"]
    new_session_id = first_args[first_args.index("--session-id") + 1]

    followup = [*MESSAGES, {"role": "assistant", "content": "hi there"}, {"role": "user", "content": "and then?"}]
    second_spawner = FakeSpawner(lines=DELTA_LINES)
    handler._spawner = second_spawner
    await handler.acompletion(model="m", messages=followup, optional_params=params)
    second_args = second_spawner.calls[0]["args"]

    assert second_args[second_args.index("--resume") + 1] == new_session_id


@pytest.mark.asyncio
async def test_a_repeat_request_writes_only_the_new_turn_not_history_nor_the_reply_the_cli_already_saved():
    """Live-captured transcripts showed the caller's echo of the CLI's own last reply saved a second time."""
    processes = []

    def spawn(*, lines):
        def factory():
            process = FakeProcess(lines=list(lines))
            processes.append(process)
            return process

        return FakeSpawner(factory=factory)

    handler = _handler(spawn(lines=DELTA_LINES))
    params = {"prompt_cache_key": "caller-a"}
    await handler.acompletion(model="m", messages=MESSAGES, optional_params=params)

    followup = [*MESSAGES, {"role": "assistant", "content": "hi there"}, {"role": "user", "content": "and then?"}]
    handler._spawner = spawn(lines=DELTA_LINES)
    await handler.acompletion(model="m", messages=followup, optional_params=params)

    second_stdin = json.loads(processes[1].stdin_payload)
    assert second_stdin == [{"role": "user", "content": "and then?"}]


@pytest.mark.asyncio
async def test_after_a_run_stopped_on_a_caller_tool_call_the_tool_call_turn_is_still_sent():
    """The proxy kills the CLI when it calls a caller tool, so that turn may not be saved; resend it."""
    tool_run = _Recording(CLIENT_TOOL_LINES)
    handler = _handler(tool_run.spawner)
    params = {"prompt_cache_key": "caller-a", "tools": [CREATE_FILE]}
    await handler.acompletion(model="m", messages=MESSAGES, optional_params=params)

    followup = [
        *MESSAGES,
        {
            "role": "assistant",
            "content": "On it.",
            "tool_calls": [{"id": "t1", "type": "function", "function": {"name": "create_file", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "t1", "content": "done"},
    ]
    next_run = _Recording(DELTA_LINES)
    handler._spawner = next_run.spawner
    await handler.acompletion(model="m", messages=followup, optional_params=params)

    roles = [turn["role"] for turn in json.loads(next_run.processes[0].stdin_payload)]
    assert roles == ["assistant", "user"]


@pytest.mark.asyncio
async def test_two_conversations_sharing_a_device_id_get_distinct_sessions_from_user_session_id():
    """`request_metadata["metadata"]["user_id"]` is a JSON blob; truncating it (prompt_cache_key) collides
    same-machine conversations onto one key, since they share `device_id`. Reading `session_id` out of it
    must not."""
    handler = _handler(FakeSpawner(lines=DELTA_LINES))
    first = await handler.acompletion(
        model="m",
        messages=MESSAGES,
        request_metadata={
            "metadata": {"user_id": '{"device_id":"same-machine","account_uuid":"","session_id":"conv-a"}'}
        },
    )
    first_args = handler._spawner.calls[0]["args"]
    first_session_id = first_args[first_args.index("--session-id") + 1]

    second_spawner = FakeSpawner(lines=DELTA_LINES)
    handler._spawner = second_spawner
    await handler.acompletion(
        model="m",
        messages=MESSAGES,
        request_metadata={
            "metadata": {"user_id": '{"device_id":"same-machine","account_uuid":"","session_id":"conv-b"}'}
        },
    )
    second_args = second_spawner.calls[0]["args"]
    second_session_id = second_args[second_args.index("--session-id") + 1]

    assert first is not None
    assert first_session_id != second_session_id


@pytest.mark.asyncio
async def test_repeat_requests_for_the_same_user_session_id_resume_the_same_session():
    handler = _handler(FakeSpawner(lines=DELTA_LINES))
    request_metadata = {"metadata": {"user_id": '{"device_id":"d","account_uuid":"","session_id":"conv-a"}'}}

    await handler.acompletion(model="m", messages=MESSAGES, request_metadata=request_metadata)

    followup = [*MESSAGES, {"role": "assistant", "content": "hi there"}, {"role": "user", "content": "and then?"}]
    second_spawner = FakeSpawner(lines=DELTA_LINES)
    handler._spawner = second_spawner
    await handler.acompletion(model="m", messages=followup, request_metadata=request_metadata)

    args = second_spawner.calls[0]["args"]
    assert "--resume" in args
    assert "--session-id" not in args


@pytest.mark.asyncio
async def test_a_side_request_sharing_the_session_id_does_not_hijack_the_main_conversation():
    """Live-observed: Claude Code sends a titling-style side request (different system prompt, one message)
    under the same session_id as the main conversation; each must keep its own CLI session."""
    request_metadata = {"metadata": {"user_id": '{"device_id":"d","account_uuid":"","session_id":"conv-a"}'}}
    main_turn_1 = [{"role": "system", "content": "main prompt"}, {"role": "user", "content": "hi"}]
    side_request = [{"role": "system", "content": "side prompt"}, {"role": "user", "content": "title this"}]
    main_turn_2 = [*main_turn_1, {"role": "assistant", "content": "hello"}, {"role": "user", "content": "more"}]
    handler = _handler(FakeSpawner(lines=DELTA_LINES))

    await handler.acompletion(model="m", messages=main_turn_1, request_metadata=request_metadata)
    main_args = handler._spawner.calls[0]["args"]
    main_session_id = main_args[main_args.index("--session-id") + 1]

    side_spawner = FakeSpawner(lines=DELTA_LINES)
    handler._spawner = side_spawner
    await handler.acompletion(model="m", messages=side_request, request_metadata=request_metadata)
    side_args = side_spawner.calls[0]["args"]

    followup_spawner = FakeSpawner(lines=DELTA_LINES)
    handler._spawner = followup_spawner
    await handler.acompletion(model="m", messages=main_turn_2, request_metadata=request_metadata)
    followup_args = followup_spawner.calls[0]["args"]

    assert "--session-id" in side_args
    assert side_args[side_args.index("--session-id") + 1] != main_session_id
    assert followup_args[followup_args.index("--resume") + 1] == main_session_id


@pytest.mark.asyncio
async def test_a_prompt_that_grows_each_turn_still_resumes_and_only_the_newest_note_reaches_the_model():
    """Live-captured: Claude Code appends per-turn notes (hook output) to the end of its system prompt, while
    a resumed CLI replays only the first-turn prompt. Each note must reach the model once, with the new turn."""
    processes = []
    spawners = []

    def spawn():
        def factory():
            process = FakeProcess(lines=list(DELTA_LINES))
            processes.append(process)
            return process

        spawners.append(FakeSpawner(factory=factory))
        return spawners[-1]

    request_metadata = {"metadata": {"user_id": '{"device_id":"d","account_uuid":"","session_id":"conv-a"}'}}
    base = "x-anthropic-billing-header: cc_version=1.abc;\nYou are Claude Code.\nrules"
    turn_1 = [{"role": "system", "content": base}, {"role": "user", "content": "hi"}]
    turn_2 = [
        {"role": "system", "content": f"{base}\n<note>A</note>"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "second"},
    ]
    turn_3 = [
        {"role": "system", "content": f"{base}\n<note>A</note>\n<note>B</note>"},
        *turn_2[1:],
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "third"},
    ]
    handler = ClaudeCodeCLI(spawner=spawn(), environment=_ENVIRONMENT)
    for turn in (turn_1, turn_2, turn_3):
        handler._spawner = spawn()
        await handler.acompletion(model="m", messages=turn, request_metadata=request_metadata)

    turn_args = [spawner.calls[0]["args"] for spawner in spawners[1:]]
    session_id = turn_args[0][turn_args[0].index("--session-id") + 1]
    assert turn_args[1][turn_args[1].index("--resume") + 1] == session_id
    assert turn_args[2][turn_args[2].index("--resume") + 1] == session_id
    assert json.loads(processes[1].stdin_payload)[-1]["content"] == "<note>A</note>\n\nsecond"
    assert json.loads(processes[2].stdin_payload)[-1]["content"] == "<note>B</note>\n\nthird"


@pytest.mark.asyncio
async def test_a_session_run_lets_the_cli_save_the_transcript_that_the_next_turn_resumes():
    """Live-reproduced: with CLAUDE_CODE_SKIP_PROMPT_HISTORY set, turn 2's `--resume` fails
    "No conversation found" because turn 1 never saved anything."""
    session_spawner = FakeSpawner(lines=DELTA_LINES)
    await _handler(session_spawner).acompletion(
        model="m",
        messages=MESSAGES,
        request_metadata={"metadata": {"user_id": '{"device_id":"d","account_uuid":"","session_id":"conv-a"}'}},
    )
    stateless_spawner = FakeSpawner(lines=DELTA_LINES)
    await _handler(stateless_spawner).acompletion(model="m", messages=MESSAGES)

    assert "CLAUDE_CODE_SKIP_PROMPT_HISTORY" not in session_spawner.calls[0]["env"]
    assert stateless_spawner.calls[0]["env"]["CLAUDE_CODE_SKIP_PROMPT_HISTORY"] == "1"


@pytest.mark.asyncio
async def test_the_streaming_path_also_resumes_the_same_session_from_request_metadata():
    """`astreaming` derives its own `_PreparedCall` independently of `acompletion`; must thread
    `request_metadata` through the same way, or streaming callers silently lose session resume."""
    handler = _handler(FakeSpawner(lines=DELTA_LINES))
    request_metadata = {"metadata": {"user_id": '{"device_id":"d","account_uuid":"","session_id":"conv-a"}'}}

    await _collect(handler, request_metadata=request_metadata)

    followup = [*MESSAGES, {"role": "assistant", "content": "hi there"}, {"role": "user", "content": "and then?"}]
    second_spawner = FakeSpawner(lines=DELTA_LINES)
    handler._spawner = second_spawner
    await _collect(handler, messages=followup, request_metadata=request_metadata)

    args = second_spawner.calls[0]["args"]
    assert "--resume" in args
    assert "--session-id" not in args


@pytest.mark.asyncio
async def test_non_json_user_falls_back_to_prompt_cache_key():
    handler = _handler(FakeSpawner(lines=DELTA_LINES))
    request_metadata = {"metadata": {"user_id": "plain-string-id"}}
    params = {"prompt_cache_key": "caller-a"}

    await handler.acompletion(model="m", messages=MESSAGES, optional_params=params, request_metadata=request_metadata)

    followup = [*MESSAGES, {"role": "assistant", "content": "hi there"}, {"role": "user", "content": "and then?"}]
    second_spawner = FakeSpawner(lines=DELTA_LINES)
    handler._spawner = second_spawner
    await handler.acompletion(model="m", messages=followup, optional_params=params, request_metadata=request_metadata)

    args = second_spawner.calls[0]["args"]
    assert "--resume" in args


@pytest.mark.asyncio
async def test_json_user_without_session_id_falls_back_to_prompt_cache_key():
    handler = _handler(FakeSpawner(lines=DELTA_LINES))
    request_metadata = {"metadata": {"user_id": '{"foo":"bar"}'}}
    params = {"prompt_cache_key": "caller-a"}

    await handler.acompletion(model="m", messages=MESSAGES, optional_params=params, request_metadata=request_metadata)

    followup = [*MESSAGES, {"role": "assistant", "content": "hi there"}, {"role": "user", "content": "and then?"}]
    second_spawner = FakeSpawner(lines=DELTA_LINES)
    handler._spawner = second_spawner
    await handler.acompletion(model="m", messages=followup, optional_params=params, request_metadata=request_metadata)

    args = second_spawner.calls[0]["args"]
    assert "--resume" in args


@pytest.mark.asyncio
async def test_concurrent_first_requests_sharing_a_new_cache_key_use_one_session_through_the_handler():
    """Two near-simultaneous *first* requests for a brand-new key must resolve to one session UUID.

    Each caller sends its own full history so far, as a real client does; this checks that racing
    `resolve_session` calls for the same never-seen key never mint two UUIDs.
    """
    store = SessionStore()
    spawners = [FakeSpawner(lines=list(DELTA_LINES)) for _ in range(5)]
    handler = ClaudeCodeCLI(spawner=spawners[0], environment=_ENVIRONMENT, sessions=store)

    async def call(spawner):
        handler._spawner = spawner
        await handler.acompletion(model="m", messages=MESSAGES, optional_params={"prompt_cache_key": "shared"})

    await asyncio.gather(*(call(spawner) for spawner in spawners))

    def _session_id(args):
        return args[args.index("--session-id") + 1] if "--session-id" in args else args[args.index("--resume") + 1]

    session_ids = {_session_id(spawner.calls[0]["args"]) for spawner in spawners}
    assert len(session_ids) == 1
