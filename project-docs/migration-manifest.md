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

Evidence: installed Claude Code `2.1.296`; `auth status --json` confirms an authenticated local subscription. Resolved MCP `2.3.0`; real stdio catalog test passes. The complete suite has 414 passing tests on Python 3.13.0 and the declared minimum Python 3.11.16; lint and format are clean. Wheel and source distribution build successfully, and the wheel was installed into a fresh Python 3.11 environment where its console command and MCP helper imported successfully. Live server checks passed for OpenAI sync completion and async SSE, complete function-tool loops, runnable Python/LangChain/LangGraph examples, LangChain invoke/stream/ainvoke/astream, and LangGraph tool execution plus checkpoint continuation. Alternate Anthropic, Bedrock, Vertex, and Foundry credential/routing environment variables are stripped from the child process. PyPI returned 404 for `claude-subscription-adapter` on 12 November 2026. No LiteLLM runtime dependency.

Existing source malformed-line skipping, stderr read order and session compatibility behavior remain unchanged. The package is ready for publication; publication is an owner-controlled external action. Do not claim guarantees outside the documented compatibility boundaries or Windows certification.
