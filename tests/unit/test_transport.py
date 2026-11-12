import asyncio
import json
import sys
from pathlib import Path
from uuid import uuid4

import pytest

from claude_subscription_adapter.errors import CLINotInstalled, QueueFull, TimedOut, UnexpectedExit
from claude_subscription_adapter.sessions import NewSession, ResumeSession
from claude_subscription_adapter.tools import read_client_tools
from claude_subscription_adapter.transformation import ConversationTurn
from claude_subscription_adapter.transport import (
    DEFAULT_INACTIVITY_WINDOW_SECONDS,
    DEFAULT_MAX_CONCURRENCY,
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_MAX_QUEUE_LENGTH,
    DEFAULT_RETRY_AFTER_SECONDS,
    DEFAULT_TIMEOUT_SECONDS,
    DEFAULT_WORKING_DIRECTORY,
    MAX_SYSTEM_PROMPT_LENGTH,
    MAX_TURNS,
    CLIProcess,
    CLISettings,
    _SubprocessCLIProcess,
    build_process_args,
    build_process_env,
    concurrency_guard,
    prepare_prompt_file,
    read_settings,
    run_cli,
    spawn_subprocess,
    stream_cli,
    transcripts_dir,
)

_TURNS = (ConversationTurn(role="user", content="hello"),)


class FakeProcess:
    """Scripted stdout with recorded lifecycle, so timeout and kill are observable without a binary."""

    def __init__(
        self, *, lines=(), stderr=b"", exit_code=0, hang=False, hang_after_lines=False, on_spawn=None, on_exit=None
    ):
        self._lines = list(lines)
        self._hang_after_lines = hang_after_lines
        self._stderr = stderr
        self._exit_code = exit_code
        self._hang = hang
        self._on_exit = on_exit
        self.returncode = None
        self.stdin_payload = None
        self.stdin_closed = False
        self.killed = False
        self.waited = False
        if on_spawn is not None:
            on_spawn()

    async def write_stdin(self, payload):
        self.stdin_payload = payload
        self.stdin_closed = True

    async def read_stdout_line(self):
        if self._hang:
            await asyncio.sleep(3600)
        if self._lines:
            return self._lines.pop(0)
        if self._hang_after_lines:
            await asyncio.sleep(3600)
        return b""

    async def read_stderr(self):
        return self._stderr

    def kill(self):
        self.killed = True
        self.returncode = -9
        if self._on_exit is not None:
            self._on_exit()

    async def wait(self):
        self.waited = True
        if self.returncode is None:
            self.returncode = self._exit_code
            if self._on_exit is not None:
                self._on_exit()
        return self.returncode


class FakeSpawner:
    def __init__(self, process=None, error=None, factory=None):
        self._process = process
        self._error = error
        self._factory = factory
        self.calls = []

    async def __call__(self, *, executable, args, env, cwd):
        self.calls.append({"executable": executable, "args": tuple(args), "env": dict(env), "cwd": cwd})
        if self._error is not None:
            raise self._error
        return self._process if self._factory is None else self._factory()


def _settings(**overrides):
    base = {"executable": "claude", "working_directory": "/srv/proxy", "timeout_seconds": 5.0}
    return CLISettings(**{**base, **overrides})


async def _run(spawner, *, settings=None, system_prompt="be helpful", model="claude-sonnet-4-5", **overrides):
    return await run_cli(
        spawner=spawner,
        settings=settings or _settings(),
        environment={},
        model=model,
        system_prompt=system_prompt,
        conversation=_TURNS,
        **overrides,
    )


def test_fake_process_satisfies_the_protocol():
    assert isinstance(FakeProcess(), CLIProcess)


@pytest.mark.asyncio
async def test_successful_run_returns_parsed_events_in_arrival_order():
    process = FakeProcess(lines=[b'{"type":"system"}\n', b'{"type":"assistant"}\n', b'{"type":"result"}\n'])

    result = await _run(FakeSpawner(process))

    assert result == ({"type": "system"}, {"type": "assistant"}, {"type": "result"})


@pytest.mark.asyncio
async def test_arguments_carry_the_model_output_format_and_system_prompt():
    spawner = FakeSpawner(FakeProcess())

    await _run(spawner, settings=_settings(executable="/opt/claude"), model="claude-opus-4-1")

    call = spawner.calls[0]
    args = call["args"]
    assert call["executable"] == "/opt/claude"
    assert call["cwd"] == "/srv/proxy"
    assert args[args.index("--model") + 1] == "claude-opus-4-1"
    assert args[args.index("--output-format") + 1] == "stream-json"
    assert args[args.index("--system-prompt") + 1] == "be helpful"


@pytest.mark.asyncio
async def test_conversation_is_written_to_stdin_as_json_then_closed():
    process = FakeProcess()

    await run_cli(
        spawner=FakeSpawner(process),
        settings=_settings(),
        environment={},
        model="claude-sonnet-4-5",
        system_prompt="",
        conversation=(ConversationTurn(role="user", content="hi"), ConversationTurn(role="assistant", content="yo")),
    )

    assert json.loads(process.stdin_payload) == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "yo"},
    ]
    assert process.stdin_closed


def test_short_system_prompt_is_passed_inline_with_no_temp_file():
    value, temp_file = prepare_prompt_file("short")

    assert (value, temp_file) == ("short", None)


def test_long_system_prompt_is_written_to_a_temp_file():
    prompt = "x" * (MAX_SYSTEM_PROMPT_LENGTH + 1)

    value, temp_file = prepare_prompt_file(prompt)

    try:
        assert temp_file is not None and value == temp_file
        assert Path(temp_file).read_text(encoding="utf-8") == prompt
    finally:
        Path(temp_file).unlink(missing_ok=True)


def test_long_prompt_switches_the_flag_to_the_file_variant():
    inline = build_process_args(system_prompt="p", model="m", use_prompt_file=False)
    from_file = build_process_args(system_prompt="/tmp/p.txt", model="m", use_prompt_file=True)

    assert "--system-prompt" in inline and "--system-prompt-file" not in inline
    assert from_file[0] == "--system-prompt-file" and from_file[1] == "/tmp/p.txt"


@pytest.mark.asyncio
@pytest.mark.parametrize("exit_code", [0, 1])
async def test_temp_prompt_file_is_removed_on_success_and_on_failure(exit_code):
    spawner = FakeSpawner(FakeProcess(exit_code=exit_code, stderr=b"boom"))

    await _run(spawner, system_prompt="y" * (MAX_SYSTEM_PROMPT_LENGTH + 1))

    prompt_path = spawner.calls[0]["args"][1]
    assert not Path(prompt_path).exists()


@pytest.mark.asyncio
async def test_temp_prompt_file_is_removed_after_a_timeout():
    spawner = FakeSpawner(FakeProcess(hang=True))

    result = await _run(
        spawner,
        settings=_settings(timeout_seconds=0.01),
        system_prompt="z" * (MAX_SYSTEM_PROMPT_LENGTH + 1),
    )

    assert isinstance(result, TimedOut)
    assert not Path(spawner.calls[0]["args"][1]).exists()


@pytest.mark.asyncio
async def test_timeout_returns_timedout_carrying_the_configured_seconds():
    result = await _run(FakeSpawner(FakeProcess(hang=True)), settings=_settings(timeout_seconds=0.02))

    assert result == TimedOut(seconds=0.02)


@pytest.mark.asyncio
async def test_timeout_kills_the_child_and_awaits_it():
    process = FakeProcess(hang=True)

    await _run(FakeSpawner(process), settings=_settings(timeout_seconds=0.01))

    assert process.killed
    assert process.waited


@pytest.mark.asyncio
async def test_unparseable_line_is_skipped_and_the_rest_survive():
    process = FakeProcess(lines=[b'{"type":"a"}\n', b"not json at all\n", b'{"type":"b"}\n'])

    assert await _run(FakeSpawner(process)) == ({"type": "a"}, {"type": "b"})


@pytest.mark.asyncio
async def test_non_object_json_line_is_skipped():
    process = FakeProcess(lines=[b"[1, 2, 3]\n", b'{"type":"a"}\n'])

    assert await _run(FakeSpawner(process)) == ({"type": "a"},)


@pytest.mark.asyncio
async def test_blank_lines_are_skipped_without_ending_the_stream():
    process = FakeProcess(lines=[b"\n", b'{"type":"a"}\n', b"   \n", b'{"type":"b"}\n'])

    assert await _run(FakeSpawner(process)) == ({"type": "a"}, {"type": "b"})


@pytest.mark.asyncio
async def test_non_zero_exit_returns_the_classified_failure_with_exit_code_and_stderr():
    process = FakeProcess(lines=[b'{"type":"a"}\n'], stderr=b"disk on fire", exit_code=1)

    result = await _run(FakeSpawner(process))

    assert result == UnexpectedExit(exit_code=1, stderr="disk on fire")


@pytest.mark.asyncio
async def test_non_zero_exit_classifies_using_the_parsed_events():
    process = FakeProcess(lines=[b'{"type":"result","result":"rate limit exceeded"}\n'], exit_code=1)

    result = await _run(FakeSpawner(process))

    assert type(result).__name__ == "RateLimited"


@pytest.mark.asyncio
async def test_spawn_filenotfound_returns_not_installed_naming_the_configured_executable():
    spawner = FakeSpawner(error=FileNotFoundError(2, "No such file"))

    result = await _run(spawner, settings=_settings(executable="/opt/bin/claude"))

    assert result == CLINotInstalled(executable="/opt/bin/claude")


@pytest.mark.asyncio
async def test_other_spawn_oserror_returns_a_failure_rather_than_raising():
    result = await _run(FakeSpawner(error=PermissionError(13, "Permission denied")))

    assert isinstance(result, UnexpectedExit)
    assert "Permission denied" in result.stderr


class ConcurrencyProbe:
    def __init__(self):
        self.live = 0
        self.peak = 0

    def enter(self):
        self.live += 1
        self.peak = max(self.peak, self.live)

    def exit(self):
        self.live -= 1


async def _peak_live_processes(*, cap, calls):
    probe = ConcurrencyProbe()

    async def slow_line():
        await asyncio.sleep(0.01)
        return b""

    def make():
        process = FakeProcess(on_spawn=probe.enter, on_exit=probe.exit)
        process.read_stdout_line = slow_line
        return process

    await asyncio.gather(
        *(
            run_cli(
                spawner=FakeSpawner(factory=make),
                settings=_settings(max_concurrency=cap),
                environment={},
                model="claude-sonnet-4-5",
                system_prompt="",
                conversation=_TURNS,
            )
            for _ in range(calls)
        )
    )
    return probe.peak


@pytest.mark.asyncio
async def test_default_cap_limits_simultaneously_live_processes():
    assert await _peak_live_processes(cap=DEFAULT_MAX_CONCURRENCY, calls=12) <= DEFAULT_MAX_CONCURRENCY


@pytest.mark.asyncio
async def test_a_raised_cap_lets_more_run_at_once_than_the_default():
    peak = await _peak_live_processes(cap=50, calls=20)

    assert DEFAULT_MAX_CONCURRENCY < peak <= 50


@pytest.mark.asyncio
async def test_settings_from_the_environment_reach_the_spawned_process():
    environment = {
        "CLAUDE_CODE_PATH": "/custom/claude",
        "CLAUDE_CODE_CWD": "/var/work",
        "CLAUDE_CODE_TIMEOUT_SECONDS": "42",
        "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "1234",
        "CLAUDE_CODE_MAX_CONCURRENCY": "50",
    }
    settings = read_settings(environment)
    spawner = FakeSpawner(FakeProcess())

    await run_cli(
        spawner=spawner,
        settings=settings,
        environment=environment,
        model="claude-sonnet-4-5",
        system_prompt="",
        conversation=_TURNS,
    )

    call = spawner.calls[0]
    resolved_cwd = str(Path("/var/work").resolve())
    assert settings == CLISettings(
        executable="/custom/claude",
        working_directory=resolved_cwd,
        timeout_seconds=42.0,
        max_output_tokens=1234,
        max_concurrency=50,
    )
    assert call["executable"] == "/custom/claude"
    assert call["cwd"] == resolved_cwd
    assert call["env"]["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] == "1234"


def test_unset_environment_falls_back_to_the_documented_defaults():
    settings = read_settings({})

    assert settings == CLISettings(
        executable="claude",
        working_directory=DEFAULT_WORKING_DIRECTORY,
        timeout_seconds=DEFAULT_TIMEOUT_SECONDS,
        max_output_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
        max_concurrency=DEFAULT_MAX_CONCURRENCY,
    )


@pytest.mark.parametrize("raw", ["abc", "", "   ", "1,2"])
def test_unparseable_numeric_settings_fall_back_to_the_default(raw):
    settings = read_settings({"CLAUDE_CODE_TIMEOUT_SECONDS": raw, "CLAUDE_CODE_MAX_OUTPUT_TOKENS": raw})

    assert settings.timeout_seconds == DEFAULT_TIMEOUT_SECONDS
    assert settings.max_output_tokens == DEFAULT_MAX_OUTPUT_TOKENS


@pytest.mark.parametrize("raw", ["", "   "])
def test_blank_path_and_cwd_fall_back_to_the_default(raw):
    settings = read_settings({"CLAUDE_CODE_PATH": raw, "CLAUDE_CODE_CWD": raw})

    assert settings.executable == "claude"
    assert settings.working_directory == DEFAULT_WORKING_DIRECTORY


@pytest.mark.parametrize("raw", ["0", "-1", "-40"])
def test_out_of_range_cap_falls_back_to_the_default(raw):
    assert read_settings({"CLAUDE_CODE_MAX_CONCURRENCY": raw}).max_concurrency == DEFAULT_MAX_CONCURRENCY


def test_environment_builder_sets_output_tokens_and_drops_alternate_provider_credentials_and_routing():
    env = build_process_env(
        {
            "ANTHROPIC_API_KEY": "sk-leak",
            "ANTHROPIC_BASE_URL": "https://gateway.example",
            "AWS_ACCESS_KEY_ID": "access-key",
            "GOOGLE_APPLICATION_CREDENTIALS": "/tmp/credentials.json",
            "AZURE_OPENAI_API_KEY": "azure-key",
            "CLAUDE_CODE_USE_BEDROCK": "1",
            "CLAUDE_CODE_USE_VERTEX": "1",
            "CLAUDE_CODE_USE_FOUNDRY": "1",
            "PATH": "/usr/bin",
        },
        CLISettings(max_output_tokens=999),
    )

    assert not any(key.startswith(("ANTHROPIC_", "AWS_", "AZURE_", "GOOGLE_", "VERTEX_")) for key in env)
    assert not {"CLAUDE_CODE_USE_BEDROCK", "CLAUDE_CODE_USE_VERTEX", "CLAUDE_CODE_USE_FOUNDRY"} & env.keys()
    assert env["PATH"] == "/usr/bin"
    assert env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] == "999"


def test_a_stateless_run_still_skips_prompt_history():
    assert build_process_env({}, CLISettings())["CLAUDE_CODE_SKIP_PROMPT_HISTORY"] == "1"


def test_a_session_run_never_skips_prompt_history_since_that_also_stops_the_transcript_resume_needs():
    env = build_process_env({"CLAUDE_CODE_SKIP_PROMPT_HISTORY": "1"}, CLISettings(), persist_session=True)

    assert "CLAUDE_CODE_SKIP_PROMPT_HISTORY" not in env


def test_environment_builder_keeps_an_operator_override_of_nonessential_traffic():
    env = build_process_env({"CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "0"}, CLISettings())

    assert env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] == "0"


@pytest.mark.asyncio
async def test_an_abandoned_child_is_killed_and_awaited_even_after_a_clean_read():
    process = FakeProcess(lines=[b'{"type":"a"}\n'])

    await _run(FakeSpawner(process))

    assert process.waited
    assert process.returncode == 0


def test_argv_pins_every_hardening_flag_for_an_inline_prompt():
    """The sandbox flags are this provider's only guarantee; dropping one must break a test, not ship."""
    assert build_process_args(system_prompt="be helpful", model="claude-opus-4-1", use_prompt_file=False) == (
        "--system-prompt",
        "be helpful",
        "--verbose",
        "--output-format",
        "stream-json",
        "--disable-slash-commands",
        "--setting-sources",
        "",
        "--strict-mcp-config",
        "--no-session-persistence",
        "--tools",
        "WebSearch",
        "--allowedTools",
        "WebSearch",
        "--max-turns",
        MAX_TURNS,
        "--model",
        "claude-opus-4-1",
        "--autocompact",
        "1000000",
        "-p",
    )


def test_caller_tools_arrive_through_the_stub_server_and_are_pre_approved():
    args = build_process_args(system_prompt="p", model="m", use_prompt_file=False, mcp_config='{"mcpServers": {}}')

    assert args[args.index("--mcp-config") + 1] == '{"mcpServers": {}}'
    assert args[args.index("--tools") + 1 : args.index("--allowedTools")] == ("WebSearch",)
    assert args[args.index("--allowedTools") + 1 : args.index("--max-turns")] == ("WebSearch", "mcp__c")


def test_without_caller_tools_no_stub_server_is_configured():
    args = build_process_args(system_prompt="p", model="m", use_prompt_file=False)

    assert "--mcp-config" not in args
    assert args[args.index("--allowedTools") + 1 : args.index("--max-turns")] == ("WebSearch",)


def test_web_search_is_always_pre_approved_with_a_raised_turn_limit():
    args = build_process_args(system_prompt="p", model="m", use_prompt_file=False)

    assert args[args.index("--allowedTools") + 1] == "WebSearch"
    assert args[args.index("--max-turns") + 1] == MAX_TURNS
    assert args[args.index("--max-turns") + 1] != "1"


def test_web_search_flags_land_before_the_prompt_flag():
    args = build_process_args(system_prompt="p", model="m", use_prompt_file=False)

    assert args.index("--allowedTools") < args.index("-p")
    assert args.index("--max-turns") < args.index("-p")


def test_argv_switches_only_the_prompt_flag_when_a_file_is_used():
    inline = build_process_args(system_prompt="p", model="m", use_prompt_file=False)
    from_file = build_process_args(system_prompt="p", model="m", use_prompt_file=True)

    assert from_file[0] == "--system-prompt-file"
    assert inline[0] == "--system-prompt"
    assert from_file[1:] == inline[1:]


def test_environment_builder_adds_exactly_the_hardening_keys():
    """Auto-memory, CLAUDE.md loading and prompt history all read or write proxy-host state."""
    env = build_process_env({}, CLISettings(max_output_tokens=32000))

    assert env == {
        "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "32000",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
        "CLAUDE_CODE_DISABLE_CLAUDE_MDS": "1",
        "CLAUDE_CODE_SKIP_PROMPT_HISTORY": "1",
        "DISABLE_NON_ESSENTIAL_MODEL_CALLS": "1",
    }


class RecordingStdin:
    def __init__(self):
        self.written = b""
        self.drained = False
        self.closed = False

    def write(self, payload):
        self.written += payload

    async def drain(self):
        self.drained = True

    def close(self):
        self.closed = True


class StubAsyncioProcess:
    def __init__(self, stdin):
        self.stdin = stdin
        self.stdout = None
        self.stderr = None
        self.returncode = None


@pytest.mark.asyncio
async def test_real_wrapper_closes_stdin_so_the_cli_sees_eof():
    """Without the close the CLI waits for an EOF that never comes and every request hangs to the timeout."""
    stdin = RecordingStdin()
    process = _SubprocessCLIProcess(process=StubAsyncioProcess(stdin))

    await process.write_stdin(b'[{"role":"user"}]')

    assert stdin.written == b'[{"role":"user"}]'
    assert stdin.drained
    assert stdin.closed


@pytest.mark.asyncio
async def test_real_wrapper_tolerates_a_child_with_no_pipes():
    process = _SubprocessCLIProcess(process=StubAsyncioProcess(None))

    await process.write_stdin(b"ignored")

    assert await process.read_stdout_line() == b""
    assert await process.read_stderr() == b""


@pytest.mark.asyncio
async def test_real_spawner_reads_an_output_line_far_past_asyncios_64_kib_default(tmp_path):
    process = await spawn_subprocess(
        executable=sys.executable,
        args=("-c", "print('x' * (1024 * 1024))"),
        env={},
        cwd=str(tmp_path),
    )

    assert await process.read_stdout_line() == b"x" * (1024 * 1024) + b"\n"
    await process.wait()


async def _stream(spawner, *, settings=None, system_prompt="be helpful", model="claude-sonnet-4-5", **overrides):
    return [
        item
        async for item in stream_cli(
            spawner=spawner,
            settings=settings or _settings(),
            environment={},
            model=model,
            system_prompt=system_prompt,
            conversation=_TURNS,
            **overrides,
        )
    ]


@pytest.mark.asyncio
async def test_streaming_yields_events_in_arrival_order():
    process = FakeProcess(lines=[b'{"type":"system"}\n', b'{"type":"assistant"}\n', b'{"type":"result"}\n'])

    assert await _stream(FakeSpawner(process)) == [
        {"type": "system"},
        {"type": "assistant"},
        {"type": "result"},
    ]


@pytest.mark.asyncio
async def test_streaming_delivers_each_event_before_the_process_exits():
    """A buffered read would hand over nothing until exit, which is the whole point of this story."""
    process = FakeProcess(lines=[b'{"n":1}\n', b'{"n":2}\n'])
    seen_before_exit = []

    async for item in stream_cli(
        spawner=FakeSpawner(process),
        settings=_settings(),
        environment={},
        model="claude-sonnet-4-5",
        system_prompt="be helpful",
        conversation=_TURNS,
    ):
        seen_before_exit.append((item, process.returncode))

    assert [returncode for _, returncode in seen_before_exit] == [None, None]


@pytest.mark.asyncio
async def test_streaming_non_zero_exit_yields_the_failure_as_the_last_item():
    process = FakeProcess(lines=[b'{"type":"system"}\n'], stderr=b"boom", exit_code=3)

    items = await _stream(FakeSpawner(process))

    assert items[0] == {"type": "system"}
    assert items[-1] == UnexpectedExit(exit_code=3, stderr="boom")


@pytest.mark.asyncio
async def test_streaming_spawn_failure_yields_not_installed_rather_than_raising():
    items = await _stream(FakeSpawner(error=FileNotFoundError("no claude")))

    assert items == [CLINotInstalled(executable="claude")]


@pytest.mark.asyncio
async def test_streaming_timeout_yields_timedout_carrying_the_configured_seconds():
    process = FakeProcess(hang=True)

    items = await _stream(FakeSpawner(process), settings=_settings(timeout_seconds=0.05))

    assert items == [TimedOut(seconds=0.05)]


@pytest.mark.asyncio
async def test_streaming_timeout_kills_the_child_and_awaits_it():
    process = FakeProcess(hang=True)

    await _stream(FakeSpawner(process), settings=_settings(timeout_seconds=0.05))

    assert process.killed
    assert process.waited


@pytest.mark.asyncio
async def test_a_slow_consumer_still_receives_timedout_as_a_value():
    """A deadline spanning a `yield` cancels the consumer's task instead, so a real client crashes."""
    process = FakeProcess(lines=[b'{"n":1}\n'], hang_after_lines=True)
    items = []

    async for item in stream_cli(
        spawner=FakeSpawner(process),
        settings=_settings(timeout_seconds=0.2),
        environment={},
        model="claude-sonnet-4-5",
        system_prompt="be helpful",
        conversation=_TURNS,
    ):
        items.append(item)
        await asyncio.sleep(0.05)

    assert items == [{"n": 1}, TimedOut(seconds=0.2)]


@pytest.mark.asyncio
async def test_a_slow_consumer_inside_the_budget_is_never_cut_short():
    """Time spent in the consumer must not count against the child's read budget."""
    process = FakeProcess(lines=[b'{"n":1}\n', b'{"n":2}\n', b'{"n":3}\n'])
    items = []

    async for item in stream_cli(
        spawner=FakeSpawner(process),
        settings=_settings(timeout_seconds=0.3),
        environment={},
        model="claude-sonnet-4-5",
        system_prompt="be helpful",
        conversation=_TURNS,
    ):
        items.append(item)
        await asyncio.sleep(0.15)

    assert items == [{"n": 1}, {"n": 2}, {"n": 3}]


@pytest.mark.asyncio
async def test_abandoning_the_stream_part_way_kills_the_child_and_unlinks_the_prompt():
    process = FakeProcess(lines=[b'{"n":1}\n', b'{"n":2}\n'])
    stream = stream_cli(
        spawner=FakeSpawner(process),
        settings=_settings(),
        environment={},
        model="claude-sonnet-4-5",
        system_prompt="x" * (MAX_SYSTEM_PROMPT_LENGTH + 1),
        conversation=_TURNS,
    )

    first = await stream.__anext__()
    await stream.aclose()

    assert first == {"n": 1}
    assert process.killed
    assert process.waited


@pytest.mark.asyncio
async def test_streaming_removes_the_temp_prompt_file_after_a_full_drain():
    spawner = FakeSpawner(FakeProcess())

    await _stream(spawner, system_prompt="x" * (MAX_SYSTEM_PROMPT_LENGTH + 1))

    assert not Path(spawner.calls[0]["args"][1]).exists()


@pytest.mark.asyncio
async def test_streaming_asks_the_cli_for_partial_messages_by_default():
    """The default is the contract: flip it and every streamed answer silently arrives as one late blob."""
    spawner = FakeSpawner(FakeProcess())

    await _stream(spawner)

    assert "--include-partial-messages" in spawner.calls[0]["args"]


@pytest.mark.asyncio
async def test_the_non_streaming_run_does_not_ask_for_partial_messages():
    spawner = FakeSpawner(FakeProcess())

    await _run(spawner)

    assert "--include-partial-messages" not in spawner.calls[0]["args"]


def test_the_partial_messages_flag_lands_before_the_prompt_flag():
    args = build_process_args(system_prompt="s", model="m", use_prompt_file=False, include_partial_messages=True)

    assert args.index("--include-partial-messages") < args.index("-p")


@pytest.mark.asyncio
async def test_stream_cli_always_forwards_the_web_search_flags_to_the_spawned_argv():
    spawner = FakeSpawner(FakeProcess())

    await _stream(spawner)

    args = spawner.calls[0]["args"]
    assert args[args.index("--allowedTools") + 1] == "WebSearch"


@pytest.mark.asyncio
async def test_run_cli_always_forwards_the_web_search_flags_to_the_spawned_argv():
    spawner = FakeSpawner(FakeProcess())

    await _run(spawner)

    args = spawner.calls[0]["args"]
    assert args[args.index("--allowedTools") + 1] == "WebSearch"
    assert args[args.index("--max-turns") + 1] == MAX_TURNS


@pytest.mark.asyncio
async def test_streaming_respects_the_concurrency_cap():
    live = {"now": 0, "peak": 0}

    def entered():
        live["now"] += 1
        live["peak"] = max(live["peak"], live["now"])

    def left():
        live["now"] -= 1

    spawner = FakeSpawner(factory=lambda: FakeProcess(lines=[b'{"n":1}\n'], on_spawn=entered, on_exit=left))
    settings = _settings(max_concurrency=2)

    await asyncio.gather(*(_stream(spawner, settings=settings) for _ in range(8)))

    assert live["peak"] <= 2


def test_the_guard_throttles_a_second_event_loop_instead_of_raising():
    """A semaphore binds to the loop that first awaited it.

    A cache keyed on the cap alone hands a second loop one bound to the first, and every acquire
    there dies with "is bound to a different event loop" rather than throttling.
    """

    async def peak_under_one_loop():
        probe = ConcurrencyProbe()

        async def slow_line():
            await asyncio.sleep(0.01)
            return b""

        def make():
            process = FakeProcess(on_spawn=probe.enter, on_exit=probe.exit)
            process.read_stdout_line = slow_line
            return process

        await asyncio.gather(
            *(
                run_cli(
                    spawner=FakeSpawner(factory=make),
                    settings=_settings(max_concurrency=2),
                    environment={},
                    model="claude-sonnet-4-5",
                    system_prompt="",
                    conversation=_TURNS,
                )
                for _ in range(6)
            )
        )
        return probe.peak

    first = asyncio.run(peak_under_one_loop())
    second = asyncio.run(peak_under_one_loop())

    assert first <= 2
    assert second <= 2


def _gated_spawner(gate):
    async def held_line():
        await gate.wait()
        return b""

    def make():
        process = FakeProcess()
        process.read_stdout_line = held_line
        return process

    return FakeSpawner(factory=make)


async def _until_spawned(spawner, count):
    while len(spawner.calls) < count:
        await asyncio.sleep(0)


async def _let_waiters_queue():
    for _ in range(5):
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_a_full_line_refuses_the_next_request_without_spawning_anything():
    gate = asyncio.Event()
    spawner = _gated_spawner(gate)
    settings = _settings(max_concurrency=1, max_queue_length=2)
    admitted = [asyncio.create_task(_stream(spawner, settings=settings)) for _ in range(3)]
    await _until_spawned(spawner, 1)
    await _let_waiters_queue()

    refused = await _stream(spawner, settings=settings)

    assert refused == [QueueFull(waiting=2, retry_after_seconds=int(DEFAULT_RETRY_AFTER_SECONDS))]
    assert len(spawner.calls) == 1
    gate.set()
    await asyncio.gather(*admitted)
    assert len(spawner.calls) == 3


@pytest.mark.asyncio
async def test_a_waiter_that_gives_up_frees_its_place_in_the_line():
    gate = asyncio.Event()
    spawner = _gated_spawner(gate)
    settings = _settings(max_concurrency=1, max_queue_length=1)
    running = asyncio.create_task(_stream(spawner, settings=settings))
    await _until_spawned(spawner, 1)
    waiter = asyncio.create_task(_stream(spawner, settings=settings))
    await _let_waiters_queue()
    waiter.cancel()
    await asyncio.gather(waiter, return_exceptions=True)

    replacement = asyncio.create_task(_stream(spawner, settings=settings))
    await _let_waiters_queue()
    gate.set()

    assert not any(isinstance(item, QueueFull) for item in await replacement)
    await running


@pytest.mark.asyncio
async def test_the_retry_estimate_is_how_long_the_last_run_took():
    settings = _settings(max_concurrency=1, max_queue_length=1)
    gate = asyncio.Event()
    spawner = _gated_spawner(gate)
    held = asyncio.create_task(_stream(spawner, settings=settings))
    await _until_spawned(spawner, 1)
    await asyncio.sleep(1.2)
    gate.set()
    await held
    gate.clear()
    again = asyncio.create_task(_stream(spawner, settings=settings))
    await _until_spawned(spawner, 2)
    await _let_waiters_queue()
    queued = asyncio.create_task(_stream(spawner, settings=settings))
    await _let_waiters_queue()

    refused = await _stream(spawner, settings=settings)

    assert refused == [QueueFull(waiting=1, retry_after_seconds=2)]
    gate.set()
    await asyncio.gather(again, queued)


def test_the_queue_limit_defaults_and_reads_from_the_environment():
    assert read_settings({}).max_queue_length == DEFAULT_MAX_QUEUE_LENGTH
    assert read_settings({"CLAUDE_CODE_MAX_QUEUE_LENGTH": "7"}).max_queue_length == 7


@pytest.mark.parametrize("raw", ["0", "-3", "nan", "abc"])
def test_an_invalid_queue_limit_falls_back_to_the_default(raw):
    assert read_settings({"CLAUDE_CODE_MAX_QUEUE_LENGTH": raw}).max_queue_length == DEFAULT_MAX_QUEUE_LENGTH


def test_each_event_loop_gets_its_own_semaphore_for_the_same_cap():
    async def guard():
        return concurrency_guard(7)

    assert asyncio.run(guard()) is not asyncio.run(guard())


@pytest.mark.asyncio
async def test_the_same_loop_and_cap_reuse_one_semaphore():
    """A fresh semaphore per call would cap nothing at all."""
    assert concurrency_guard(9) is concurrency_guard(9)


@pytest.mark.asyncio
@pytest.mark.parametrize("exit_code", [0, 1])
async def test_the_tools_file_holds_the_caller_schemas_and_is_removed_afterwards(exit_code):
    tools = read_client_tools(
        [{"type": "function", "function": {"name": "create_file", "parameters": {"type": "object"}}}], None
    )
    seen = {}

    def spawn():
        args = spawner.calls[0]["args"]
        tools_file = json.loads(args[args.index("--mcp-config") + 1])["mcpServers"]["c"]["args"][1]
        seen["path"] = tools_file
        seen["tools"] = json.loads(Path(tools_file).read_text(encoding="utf-8"))
        return FakeProcess(exit_code=exit_code)

    spawner = FakeSpawner(factory=spawn)
    await _stream(spawner, client_tools=tools)

    assert seen["tools"] == [{"name": "create_file", "description": "", "inputSchema": {"type": "object"}}]
    assert not Path(seen["path"]).exists()


def test_the_default_working_folder_is_neutral_and_outside_any_project():
    """The CLI shows the model its folder and git details, so the proxy's checkout must never be the default."""
    assert Path(DEFAULT_WORKING_DIRECTORY).name == "claude-code-cli"
    assert not Path(DEFAULT_WORKING_DIRECTORY).is_relative_to(Path.cwd())


@pytest.mark.asyncio
async def test_the_default_working_folder_is_created_and_used_for_the_child():
    spawner = FakeSpawner(FakeProcess())

    await _stream(spawner, settings=read_settings({}))

    assert spawner.calls[0]["cwd"] == DEFAULT_WORKING_DIRECTORY
    assert Path(DEFAULT_WORKING_DIRECTORY).is_dir()


def test_the_turn_limit_leaves_room_for_several_searches_before_answering():
    """A research-then-write request used all 3 turns on searching and failed with no message."""
    assert int(MAX_TURNS) >= 8


# --------------------------------------------------------------------------------------
# Session resume argv and settings
# --------------------------------------------------------------------------------------


def test_no_session_argument_keeps_the_stateless_flag_unchanged():
    args = build_process_args(system_prompt="p", model="m", use_prompt_file=False)

    assert "--no-session-persistence" in args
    assert "--session-id" not in args
    assert "--resume" not in args


def test_a_new_session_passes_session_id_and_drops_no_session_persistence():
    session_id = uuid4()

    args = build_process_args(system_prompt="p", model="m", use_prompt_file=False, session=NewSession(session_id))

    assert args[args.index("--session-id") + 1] == str(session_id)
    assert "--no-session-persistence" not in args
    assert "--resume" not in args


def test_a_resumed_session_passes_resume_and_drops_no_session_persistence():
    session_id = uuid4()

    args = build_process_args(
        system_prompt="p",
        model="m",
        use_prompt_file=False,
        session=ResumeSession(session_id=session_id, turns_already_sent=2),
    )

    assert args[args.index("--resume") + 1] == str(session_id)
    assert "--no-session-persistence" not in args
    assert "--session-id" not in args


def test_session_flags_land_before_the_prompt_flag():
    args = build_process_args(system_prompt="p", model="m", use_prompt_file=False, session=NewSession(uuid4()))

    assert args.index("--session-id") < args.index("-p")


def test_the_unset_inactivity_window_defaults_to_four_hours():
    assert DEFAULT_INACTIVITY_WINDOW_SECONDS == 4 * 60 * 60.0
    assert read_settings({}).session_inactivity_window_seconds == DEFAULT_INACTIVITY_WINDOW_SECONDS


def test_the_inactivity_window_is_honoured_from_the_environment():
    settings = read_settings({"CLAUDE_CODE_SESSION_INACTIVITY_WINDOW_SECONDS": "30"})

    assert settings.session_inactivity_window_seconds == 30.0


def test_transcripts_dir_replaces_every_non_alphanumeric_character():
    assert transcripts_dir("/tmp/claude-code-cli") == Path.home() / ".claude" / "projects" / "-tmp-claude-code-cli"


def test_cwd_override_through_a_symlink_resolves_to_the_real_path(tmp_path):
    """Mirrors macOS: `tempfile.gettempdir()` and any operator-supplied `CLAUDE_CODE_CWD` can be a symlink,
    but the CLI resolves its own real cwd before computing the transcript folder, so ours must match."""
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    symlinked_cwd = tmp_path / "symlinked"
    symlinked_cwd.symlink_to(real_dir)

    settings = read_settings({"CLAUDE_CODE_CWD": str(symlinked_cwd)})

    assert settings.working_directory == str(real_dir.resolve())
    assert settings.working_directory != str(symlinked_cwd)
