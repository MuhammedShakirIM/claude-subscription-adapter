# Claude Subscription Adapter

A small local OpenAI-compatible Chat Completions server around the existing Claude Code CLI provider. Python 3.11+ is required. Install and authenticate Claude Code separately as the user running the server.

## Run from this checkout

```bash
uv sync
uv run claude-subscription-adapter serve
# Optional port override
uv run claude-subscription-adapter serve --port 9100
# Install the local package as a standalone tool
uv tool install .
```

The foreground server defaults to `http://localhost:9001`; clients use `http://localhost:9001/v1`. This package has not been published to PyPI.

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:9001/v1", api_key="local")
response = client.chat.completions.create(
    model="sonnet", messages=[{"role": "user", "content": "Hello"}]
)
print(response.choices[0].message.content)
```

Use `AsyncOpenAI` for asynchronous calls and `stream=True` for SSE. The server accepts text messages, system/developer prompts, assistant function calls and correlated tool results. Applications execute their own tools. Runnable complete tool-loop examples are in `examples/python_tools.py`, `examples/langchain_tools.py`, and `examples/langgraph_tools.py` (development dependencies include these clients).

## Run the examples

The three scripts in `examples/` are runnable end-to-end function-tool loops. They use the local adapter, ask Claude to call a `multiply` function, execute that function in the client process, send its result back, and print the final answer.

Start the adapter in one terminal:

```bash
uv run claude-subscription-adapter serve
```

Then, from a second terminal in this checkout, run one of the examples:

```bash
uv run examples/python_tools.py
uv run examples/langchain_tools.py
uv run examples/langgraph_tools.py
```

All examples default to model `sonnet` and `http://localhost:9001/v1`. Override either when needed:

```bash
CSA_MODEL=your-model CSA_BASE_URL=http://localhost:9100/v1 uv run examples/python_tools.py
```

When server authentication is enabled, also supply the configured token as `CSA_API_KEY`:

```bash
CSA_API_KEY=your-token uv run examples/langchain_tools.py
```

## Configuration

Flags override environment settings. `CSA_HOST` defaults to `127.0.0.1`; `CSA_PORT` to `9001`. Set `CSA_API_KEY` to require a bearer token on model routes and readiness; it is required for non-loopback binding. Health and OpenAPI remain public. `CSA_MODELS` optionally contains a JSON object mapping advertised model aliases to CLI model names; unlisted request models are passed through. An empty configured catalogue returns an empty `/v1/models` list.

Copied settings: `CLAUDE_CODE_PATH` (default `claude`), `CLAUDE_CODE_CWD` (neutral temporary directory), `CLAUDE_CODE_TIMEOUT_SECONDS` (600), `CLAUDE_CODE_MAX_OUTPUT_TOKENS` (32000), `CLAUDE_CODE_MAX_CONCURRENCY` (4), `CLAUDE_CODE_MAX_QUEUE_LENGTH` (20), and `CLAUDE_CODE_SESSION_INACTIVITY_WINDOW_SECONDS` (14400). Run one worker.

Endpoints: `POST /v1/chat/completions`, `GET /v1/models`, `/health`, `/ready`, and `/openapi.json`. Readiness checks CLI authentication without inference. Streaming includes stable response IDs, tool indexes, terminal finish reason and `[DONE]`; `stream_options.include_usage` enables a usage-only chunk. Errors use an `error` object. After SSE begins, failures are sent as `data: {"error": ...}` and the stream closes without a successful terminal marker.

## Compatibility boundaries

This is a faithful extraction verified with fixture, subprocess, installed-wheel, and authenticated local-CLI checks. Live validation covers OpenAI sync/async streaming, complete function-tool loops, LangChain invoke/stream/async variants, and LangGraph tool execution with checkpoint continuation.

The inherited provider silently ignores sampling/structured-output controls. Token limits can raise the internal configured cap, but cannot lower it. `tool_choice="none"` suppresses tools; required/named choices and `parallel_tool_calls=false` are not enforced. Invalid tool definitions retain the source's skip behavior. Strict tool schemas are not guaranteed. WebSearch stays enabled; built-in file and shell tools are excluded.

Requests are stateless by default. `prompt_cache_key` opts into copied session behavior; `metadata.user_id` containing a JSON string with `session_id` takes precedence. A LangGraph thread ID alone does not enable resume. Restart loses the in-memory mapping. The transport removes alternate Anthropic, Bedrock, Vertex, and Foundry credentials and routing configuration from the CLI child so it uses the local Claude subscription login.

## Development

```bash
uv run pytest -q
uv run ruff check src
uv build
```

See `project-docs/migration-manifest.md` for provenance and intentional differences, and `project-docs/review-notes.md` for release evidence.
