# Migration manifest

Implementation started 11 November 2026, authorized by the request to start building from architecture.md.

Source: `/Users/shakir/projects/projekts-litellm-v1`, revision `801fcf0af09a85f6c3b9616e36e0cd6ed9ce743d`. Source checkout was read only. Owner reports a successful 20-person deployment of the original provider.

Copied `litellm/llms/claude_code_cli/{transport,transformation,streaming,tools,client_tools_server,usage,errors,sessions}.py` into the matching modules under `src/claude_subscription_adapter/`. Copied `handler.py` to `service.py`. Replaced package `__init__.py` with version information. Copied all five NDJSON fixtures into `tests/fixtures/` and provider tests into `tests/unit/`.

Intentional differences:

- LiteLLM imports replaced by standard logging and small local models/dictionaries in schemas.py. CompletionService replaces CustomLLM inheritance; removed provider registration and synchronous rejection stubs. HTTP clients can use synchronous or asynchronous requests.
- Dispatcher-specific tests removed because the dispatcher is no longer part of this package. Kept provider behavior tests with imports, fixture paths and public cache-counter fields adapted. Package export test now checks the local service import.
- HTTP statuses now follow the architecture: missing CLI/login 503, unexpected exit 502, queue refusal 429 with Retry-After. Classification logic preserved.
- Added API request/history validation, response identities, SSE serialization, bearer authentication, model aliases, readiness, configuration and console command.
- POSIX children start in a new session; cleanup kills the process group to include MCP descendants. Windows behavior is not certified.
- Local finish mapping covers Anthropic and standard OpenAI reasons including compaction; unrelated provider mappings were excluded.

License: retained the source MIT copyright notice in LICENSE; no enterprise modules copied.

Evidence: installed Claude Code `2.1.316`; help confirms `--autocompact` and `auth status --json`. Resolved MCP `2.3.0`; real stdio catalog test passes. Python used for development: 3.13.0. Wheel and source distribution build successfully. Provider regression run: 405 passed; with HTTP tests: 408 passed. No LiteLLM runtime dependency.

Remaining release gates: live CLI inference and complete tool loops; sync/async OpenAI SDK, LangChain stream/invoke variants and LangGraph checkpoint parity; process-tree leak tests for cancellation/disconnect/shutdown; installed-artifact MCP and clean uv tool/project checks; malformed/truncated and oversized output behavior; alternate credential/routing environment audit; dependency size/startup measurement; Python 3.11 verification; final publication metadata and name availability. Existing source malformed-line skipping, stderr read order and session compatibility behavior remain unchanged. Do not claim full release parity until these checks are recorded.
