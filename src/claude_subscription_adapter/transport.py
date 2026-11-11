"""The only part of the Claude Code CLI provider that touches the operating system: spawn, feed, read, kill."""

import asyncio
import json
import logging
import os
import re
import signal
import tempfile
from collections.abc import AsyncGenerator, AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from math import ceil, isfinite
from pathlib import Path
from typing import Final, Protocol, assert_never, runtime_checkable
from weakref import WeakKeyDictionary

from pydantic import TypeAdapter, ValidationError

verbose_logger = logging.getLogger(__name__)

from .errors import CLIFailure, QueueFull, TimedOut, classify_failure
from .sessions import NewSession, ResumeSession, SessionDecision
from .tools import MCP_SERVER_NAME, ClientTool, mcp_config, tools_file_payload
from .transformation import ConversationTurn

MAX_SYSTEM_PROMPT_LENGTH: Final = 65536

DEFAULT_EXECUTABLE: Final = "claude"
DEFAULT_TIMEOUT_SECONDS: Final = 600.0
DEFAULT_MAX_OUTPUT_TOKENS: Final = 32000
DEFAULT_MAX_CONCURRENCY: Final = 4
DEFAULT_MAX_QUEUE_LENGTH: Final = 20
DEFAULT_RETRY_AFTER_SECONDS: Final = 30.0
DEFAULT_INACTIVITY_WINDOW_SECONDS: Final = 4 * 60 * 60.0

_LOG_LINE_LIMIT: Final = 200
STDOUT_LINE_LIMIT_BYTES: Final = 64 * 1024 * 1024
# The CLI tells the model its working folder, plus git details when that folder is a repo. Running from the
# proxy's own checkout would show every caller the server's project name, branch and commits.
DEFAULT_WORKING_DIRECTORY: Final = str((Path(tempfile.gettempdir()) / "claude-code-cli").resolve())


@dataclass(frozen=True, slots=True)
class CLISettings:
    """Per-box tuning, all overridable by environment variable so no deployment needs a code change."""

    executable: str = DEFAULT_EXECUTABLE
    working_directory: str = ""
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY
    max_queue_length: int = DEFAULT_MAX_QUEUE_LENGTH
    session_inactivity_window_seconds: float = DEFAULT_INACTIVITY_WINDOW_SECONDS


@runtime_checkable
class CLIProcess(Protocol):
    """What the transport needs from a spawned child, so tests can supply one without a binary."""

    @property
    def returncode(self) -> int | None: ...

    async def write_stdin(self, payload: bytes) -> None: ...

    async def read_stdout_line(self) -> bytes: ...

    async def read_stderr(self) -> bytes: ...

    def kill(self) -> None: ...

    async def wait(self) -> int: ...


class CLISpawner(Protocol):
    """Starts a process. Injected, never reached for at module scope, so no test can spawn a real CLI."""

    async def __call__(
        self,
        *,
        executable: str,
        args: Sequence[str],
        env: Mapping[str, str],
        cwd: str,
    ) -> CLIProcess: ...


@dataclass(frozen=True, slots=True)
class _SubprocessCLIProcess:
    """The real child, wrapped so the transport only ever sees the small `CLIProcess` surface."""

    process: asyncio.subprocess.Process

    @property
    def returncode(self) -> int | None:
        return self.process.returncode

    async def write_stdin(self, payload: bytes) -> None:
        if (stdin := self.process.stdin) is None:
            return
        stdin.write(payload)
        await stdin.drain()
        stdin.close()

    async def read_stdout_line(self) -> bytes:
        return b"" if (stdout := self.process.stdout) is None else await stdout.readline()

    async def read_stderr(self) -> bytes:
        return b"" if (stderr := self.process.stderr) is None else await stderr.read()

    def kill(self) -> None:
        if os.name == "posix":
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        elif self.process.returncode is None:
            self.process.kill()

    async def wait(self) -> int:
        return await self.process.wait()


async def spawn_subprocess(
    *,
    executable: str,
    args: Sequence[str],
    env: Mapping[str, str],
    cwd: str,
) -> CLIProcess:
    """The production spawner. Passed in by the handler, so no test path can reach it."""
    return _SubprocessCLIProcess(
        process=await asyncio.create_subprocess_exec(
            executable,
            *args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
            env=dict(env),
            limit=STDOUT_LINE_LIMIT_BYTES,
            start_new_session=os.name == "posix",
        )
    )


def _positive_number(environment: Mapping[str, str], key: str, default: float) -> float:
    """`inf` would disable the timeout and `nan` would make every comparison against it silently false."""
    raw: Final = environment.get(key, "").strip()
    if not raw:
        return default
    try:
        parsed: Final = float(raw)
    except ValueError:
        verbose_logger.warning("claude_code_cli: %s=%r is not a number, using %s", key, raw, default)
        return default
    if not isfinite(parsed) or parsed <= 0:
        verbose_logger.warning("claude_code_cli: %s=%r must be a positive finite number, using %s", key, raw, default)
        return default
    return parsed


def _positive_count(environment: Mapping[str, str], key: str, default: int) -> int:
    """A cap that floors to 0 makes a semaphore that is locked from the start, so every request would hang."""
    parsed: Final = _positive_number(environment, key, float(default))
    if (floored := int(parsed)) >= 1:
        return floored
    verbose_logger.warning("claude_code_cli: %s=%r rounds down below 1, using %s", key, environment.get(key), default)
    return default


def read_settings(environment: Mapping[str, str]) -> CLISettings:
    """Pure over the mapping it is handed, so tests never have to touch the real environment."""
    executable: Final = environment.get("CLAUDE_CODE_PATH", "").strip() or DEFAULT_EXECUTABLE
    cwd_override: Final = environment.get("CLAUDE_CODE_CWD", "").strip()
    working_directory: Final = str(Path(cwd_override).resolve()) if cwd_override else DEFAULT_WORKING_DIRECTORY
    return CLISettings(
        executable=executable,
        working_directory=working_directory,
        timeout_seconds=_positive_number(environment, "CLAUDE_CODE_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS),
        max_output_tokens=_positive_count(environment, "CLAUDE_CODE_MAX_OUTPUT_TOKENS", DEFAULT_MAX_OUTPUT_TOKENS),
        max_concurrency=_positive_count(environment, "CLAUDE_CODE_MAX_CONCURRENCY", DEFAULT_MAX_CONCURRENCY),
        max_queue_length=_positive_count(environment, "CLAUDE_CODE_MAX_QUEUE_LENGTH", DEFAULT_MAX_QUEUE_LENGTH),
        session_inactivity_window_seconds=_positive_number(
            environment, "CLAUDE_CODE_SESSION_INACTIVITY_WINDOW_SECONDS", DEFAULT_INACTIVITY_WINDOW_SECONDS
        ),
    )


def transcripts_dir(working_directory: str) -> Path:
    """Where the CLI writes `<session-id>.jsonl` for a run launched from `working_directory`.

    Confirmed by direct inspection of a real `~/.claude/projects/` tree: every character outside
    `[A-Za-z0-9]` in the absolute cwd (slashes, dots, ...) becomes `-`.
    """
    escaped: Final = re.sub(r"[^A-Za-z0-9]", "-", working_directory)
    return Path.home() / ".claude" / "projects" / escaped


# Each web search costs a turn, and a caller asking to research several things can use five or six before it
# answers or calls a tool. Running out ends the run with an error, so leave room.
MAX_TURNS: Final = "10"


def _session_args(session: SessionDecision | None) -> tuple[str, ...]:
    """No `prompt_cache_key` on the request means `session` is None: unchanged, stateless behavior.

    A resolved session drops `--no-session-persistence` so the CLI writes the transcript `--resume` reads.
    """
    match session:
        case None:
            return ("--no-session-persistence",)
        case NewSession(session_id):
            return ("--session-id", str(session_id))
        case ResumeSession(session_id):
            return ("--resume", str(session_id))
        case _:
            assert_never(session)


def build_process_args(
    *,
    system_prompt: str,
    model: str,
    use_prompt_file: bool,
    include_partial_messages: bool = False,
    mcp_config: str | None = None,
    session: SessionDecision | None = None,
) -> tuple[str, ...]:
    """The headless single-shot invocation. Extra flags belong before `-p`, which introduces the prompt.

    `--tools WebSearch` removes every other built-in tool, so nothing a caller asks for can read, write
    or run anything on the proxy server. The caller's own tools arrive through `mcp_config` instead.
    """
    return (
        "--system-prompt-file" if use_prompt_file else "--system-prompt",
        system_prompt,
        "--verbose",
        "--output-format",
        "stream-json",
        # Without this the CLI sends only whole assistant messages, so a streaming caller gets no deltas.
        *(("--include-partial-messages",) if include_partial_messages else ()),
        "--disable-slash-commands",
        # Loads none of the host's user, project or local settings: their extra folders, hooks and
        # permissions belong to whoever runs the proxy and must not reach callers.
        "--setting-sources",
        "",
        "--strict-mcp-config",
        *(("--mcp-config", mcp_config) if mcp_config is not None else ()),
        *_session_args(session),
        "--tools",
        "WebSearch",
        "--allowedTools",
        "WebSearch",
        *((f"mcp__{MCP_SERVER_NAME}",) if mcp_config is not None else ()),
        "--max-turns",
        MAX_TURNS,
        "--model",
        model,
        "--autocompact",
        "1000000",
        "-p",
    )


def build_process_env(
    environment: Mapping[str, str], settings: CLISettings, *, persist_session: bool = False
) -> Mapping[str, str]:
    """Inherit the proxy's environment, minus anything that would make the CLI bill an API key or phone home.

    `CLAUDE_CODE_SKIP_PROMPT_HISTORY` also stops the CLI saving the transcript `--resume` reads, so a run
    carrying a session must not have it, not even inherited from the proxy's own environment.
    """
    dropped: Final = frozenset(
        ("ANTHROPIC_API_KEY", "CLAUDE_CODE_SKIP_PROMPT_HISTORY") if persist_session else ("ANTHROPIC_API_KEY",)
    )
    inherited: Final = {key: value for key, value in environment.items() if key not in dropped}
    return {
        **inherited,
        "CLAUDE_CODE_MAX_OUTPUT_TOKENS": str(settings.max_output_tokens),
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": inherited.get("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "1"),
        "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
        "CLAUDE_CODE_DISABLE_CLAUDE_MDS": "1",
        **({} if persist_session else {"CLAUDE_CODE_SKIP_PROMPT_HISTORY": "1"}),
        "DISABLE_NON_ESSENTIAL_MODEL_CALLS": inherited.get("DISABLE_NON_ESSENTIAL_MODEL_CALLS", "1"),
    }


def prepare_prompt_file(system_prompt: str) -> tuple[str, str | None]:
    """Return the argument value plus the temp file to delete, if the prompt was too big to pass inline."""
    if len(system_prompt) <= MAX_SYSTEM_PROMPT_LENGTH and os.name != "nt":
        return system_prompt, None
    with tempfile.NamedTemporaryFile(
        "w", delete=False, encoding="utf-8", prefix="claude-code-prompt-", suffix=".txt"
    ) as handle:
        handle.write(system_prompt)
        return handle.name, handle.name


def prepare_tools_file(client_tools: Sequence[ClientTool]) -> str | None:
    """The tool list for the stub server, or None when the caller sent no tools."""
    if not client_tools:
        return None
    with tempfile.NamedTemporaryFile(
        "w", delete=False, encoding="utf-8", prefix="claude-code-tools-", suffix=".json"
    ) as handle:
        handle.write(tools_file_payload(client_tools))
        return handle.name


def _stdin_payload(conversation: Sequence[ConversationTurn]) -> bytes:
    return json.dumps([{"role": turn.role, "content": turn.content} for turn in conversation]).encode("utf-8")


_EVENT_ADAPTER: Final = TypeAdapter(dict[str, object])


def _parse_line(line: str) -> Mapping[str, object] | None:
    """CLI output is untrusted text, so a line becomes an event only once it validates as a JSON object."""
    try:
        return _EVENT_ADAPTER.validate_json(line)
    except ValidationError:
        verbose_logger.warning("claude_code_cli: skipping unparseable output line %r", line[:_LOG_LINE_LIMIT])
        return None


# mutable-ok: one semaphore per loop and cap, cached across calls; entries die with their loop
_SEMAPHORES: Final["WeakKeyDictionary[asyncio.AbstractEventLoop, dict[int, asyncio.Semaphore]]"] = WeakKeyDictionary()


def concurrency_guard(max_concurrency: int) -> asyncio.Semaphore:
    """One semaphore per running loop and cap value; a fresh one per call would cap nothing.

    A semaphore binds to the loop that first awaited it, so a cache keyed on the cap alone hands a
    second loop one that raises instead of throttling.
    """
    per_loop: Final = _SEMAPHORES.setdefault(asyncio.get_running_loop(), {})
    if (existing := per_loop.get(max_concurrency)) is not None:
        return existing
    created: Final = asyncio.Semaphore(max_concurrency)
    per_loop[max_concurrency] = created
    return created


@dataclass(slots=True)
class _Line:
    """Who is waiting behind one guard, and how long the last run took, which is when a slot likely frees."""

    waiting: int = 0
    last_run_seconds: float = DEFAULT_RETRY_AFTER_SECONDS


# mutable-ok: one line per guard, so the waiting count is shared by every request queued on it
_LINES: Final["WeakKeyDictionary[asyncio.Semaphore, _Line]"] = WeakKeyDictionary()


@asynccontextmanager
async def _slot(settings: CLISettings) -> AsyncIterator[QueueFull | None]:
    """Hold a run slot, waiting if all are busy. Yields `QueueFull` instead, holding nothing, if the line is full."""
    guard: Final = concurrency_guard(settings.max_concurrency)
    line: Final = _LINES.setdefault(guard, _Line())
    if guard.locked() and line.waiting >= settings.max_queue_length:
        yield QueueFull(waiting=line.waiting, retry_after_seconds=ceil(line.last_run_seconds))
        return
    line.waiting += 1
    try:
        await guard.acquire()
    finally:
        line.waiting -= 1
    started: Final = asyncio.get_running_loop().time()
    try:
        yield None
    finally:
        line.last_run_seconds = asyncio.get_running_loop().time() - started
        guard.release()


async def _terminate(process: CLIProcess) -> None:
    if process.returncode is None or isinstance(process, _SubprocessCLIProcess):
        process.kill()
        await process.wait()


async def _read_line_within(process: CLIProcess, deadline: float) -> bytes:
    """Bound this one read by what is left of the run's budget, so no deadline ever spans a `yield`."""
    async with asyncio.timeout_at(deadline):
        return await process.read_stdout_line()


async def _stream_events(process: CLIProcess, deadline: float) -> AsyncIterator[Mapping[str, object]]:
    while raw := await _read_line_within(process, deadline):
        if not (line := raw.decode("utf-8", errors="replace").strip()):
            continue
        if (event := _parse_line(line)) is not None:
            yield event


async def _finish_stream(
    process: CLIProcess,
    *,
    events: tuple[Mapping[str, object], ...],
    settings: CLISettings,
    deadline: float,
) -> CLIFailure | None:
    async with asyncio.timeout_at(deadline):
        stderr: Final = (await process.read_stderr()).decode("utf-8", errors="replace")
        exit_code: Final = await process.wait()
    if exit_code == 0:
        return None
    return classify_failure(
        exit_code=exit_code,
        stderr=stderr,
        stdout_events=events,
        executable=settings.executable,
    )


async def _stream_process(
    process: CLIProcess,
    *,
    conversation: Sequence[ConversationTurn],
    settings: CLISettings,
) -> AsyncIterator[Mapping[str, object] | CLIFailure]:
    """Yield each event as its line arrives, then the failure if there was one. Never raises a timeout."""
    deadline: Final = asyncio.get_running_loop().time() + settings.timeout_seconds
    seen: list[Mapping[str, object]] = []  # mutable-ok: the classifier needs the events a failing run produced
    try:
        async with asyncio.timeout_at(deadline):
            await process.write_stdin(_stdin_payload(conversation))
        async for event in _stream_events(process, deadline):
            seen.append(event)
            yield event
        if (
            failure := await _finish_stream(process, events=tuple(seen), settings=settings, deadline=deadline)
        ) is not None:
            yield failure
    except TimeoutError:
        yield TimedOut(seconds=settings.timeout_seconds)


async def stream_cli(
    *,
    spawner: CLISpawner,
    settings: CLISettings,
    environment: Mapping[str, str],
    model: str,
    system_prompt: str,
    conversation: Sequence[ConversationTurn],
    include_partial_messages: bool = True,
    client_tools: Sequence[ClientTool] = (),
    session: SessionDecision | None = None,
) -> AsyncGenerator[Mapping[str, object] | CLIFailure, None]:
    """Stream one CLI invocation. Events arrive as they are read; a failure arrives as the last item."""
    prompt_file: str | None = None  # rebind-ok: assigned inside the try so a later raise still unlinks it
    tools_file: str | None = None  # rebind-ok: assigned inside the try so a later raise still unlinks it
    try:
        prompt_value, prompt_file = prepare_prompt_file(system_prompt)
        tools_file = prepare_tools_file(client_tools)
        args: Final = build_process_args(
            system_prompt=prompt_value,
            model=model,
            use_prompt_file=prompt_file is not None,
            include_partial_messages=include_partial_messages,
            mcp_config=mcp_config(tools_file) if tools_file is not None else None,
            session=session,
        )
        env: Final = build_process_env(environment, settings, persist_session=session is not None)

        async with _slot(settings) as refusal:
            if refusal is not None:
                yield refusal
                return
            try:
                if settings.working_directory == DEFAULT_WORKING_DIRECTORY:
                    Path(DEFAULT_WORKING_DIRECTORY).mkdir(exist_ok=True)
                process: Final = await spawner(
                    executable=settings.executable,
                    args=args,
                    env=env,
                    cwd=settings.working_directory,
                )
            except OSError as spawn_error:
                yield classify_failure(
                    exit_code=-1,
                    stderr="",
                    executable=settings.executable,
                    spawn_error=spawn_error,
                )
                return
            try:
                async for item in _stream_process(process, conversation=conversation, settings=settings):
                    yield item
            finally:
                await _terminate(process)
    finally:
        for temp_file in (prompt_file, tools_file):
            if temp_file is not None:
                Path(temp_file).unlink(missing_ok=True)


async def run_cli(
    *,
    spawner: CLISpawner,
    settings: CLISettings,
    environment: Mapping[str, str],
    model: str,
    system_prompt: str,
    conversation: Sequence[ConversationTurn],
) -> tuple[Mapping[str, object], ...] | CLIFailure:
    """Run one CLI invocation to completion. Failures come back as values; only cancellation propagates."""
    events: list[Mapping[str, object]] = []  # mutable-ok: accumulating a stream of unknown length
    stream: Final = stream_cli(
        spawner=spawner,
        settings=settings,
        environment=environment,
        model=model,
        system_prompt=system_prompt,
        conversation=conversation,
        include_partial_messages=False,
    )
    # Without the explicit close, abandoning the generator early leaves the child and the temp
    # prompt file to garbage-collection time instead of this call.
    try:
        async for item in stream:
            if not isinstance(item, Mapping):
                return item
            events.append(item)
    finally:
        await stream.aclose()
    return tuple(events)


__all__ = [
    "DEFAULT_INACTIVITY_WINDOW_SECONDS",
    "DEFAULT_WORKING_DIRECTORY",
    "MAX_TURNS",
    "CLIProcess",
    "CLISettings",
    "CLISpawner",
    "build_process_args",
    "build_process_env",
    "concurrency_guard",
    "prepare_prompt_file",
    "prepare_tools_file",
    "read_settings",
    "run_cli",
    "spawn_subprocess",
    "stream_cli",
    "transcripts_dir",
]
