# Claude Subscription Adapter

Run applications built for the OpenAI Chat Completions API against your local
Claude Code subscription. The adapter starts Claude Code on the machine where
the server runs and exposes a small, OpenAI-compatible HTTP API for chat,
streaming, and application-defined function tools.

It is intended for local development and self-hosted use. It is not an
Anthropic API proxy and does not use an Anthropic API key: Claude Code must
already be installed and authenticated for the operating-system user running
the adapter.

## What you get

- `POST /v1/chat/completions` compatible with the OpenAI Python SDK and
  `langchain-openai`'s `ChatOpenAI`.
- Standard application function tools, including multi-round tool loops.
- Claude Code's built-in `WebSearch`, with no search API key or search client
  required in your application.
- Optional Server-Sent Events streaming, model aliases, authentication,
  concurrency limits, and opt-in conversation continuation.

The adapter deliberately exposes a narrow capability set. Claude Code's
built-in file, shell, and other host tools are not available through it.

## Prerequisites

- Python 3.11 or later.
- [Claude Code](https://docs.anthropic.com/en/docs/claude-code) installed on
  the computer that will run the adapter.
- An active Claude Code login for the same OS user that starts the adapter.
- [`uv`](https://docs.astral.sh/uv/) (recommended) or another Python 3.11+
  environment manager.

Before starting the server, verify Claude Code can authenticate in that shell:

```bash
claude auth status --json
```

If it is not authenticated, run `claude` and complete its login flow first.

## Quick start

Clone the repository and install its dependencies:

```bash
git clone <repository-url>
cd claude-subscription-adapter
uv sync
```

Start the server in one terminal:

```bash
uv run claude-subscription-adapter serve
```

It listens on `http://127.0.0.1:9001` by default. Confirm that both the HTTP
server and Claude Code are ready:

```bash
curl http://127.0.0.1:9001/health
curl http://127.0.0.1:9001/ready
```

In a second terminal, make a request with the OpenAI SDK:

```bash
uv add openai
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:9001/v1", api_key="local")

response = client.chat.completions.create(
    model="sonnet",
    messages=[{"role": "user", "content": "Explain why tests are useful in one sentence."}],
)
print(response.choices[0].message.content)
```

`api_key="local"` is a placeholder while server authentication is disabled.
If `CSA_API_KEY` is configured, supply that exact value instead.

### Choose models explicitly

Model names are passed through to Claude Code unless you configure aliases.
For a stable application-facing model name, copy the example environment file:

```bash
cp .env.example .env
```

Then set `CSA_MODELS` to models available to your Claude Code account, for
example:

```dotenv
CSA_MODELS='{"sonnet":"claude-sonnet-5-5","opus":"claude-opus-5-5"}'
```

With that configuration, `model="sonnet"` is sent to Claude Code as
`claude-sonnet-5-5`. Shell environment variables override `.env`; `--host`
and `--port` override their corresponding environment variables.

## Web search

Claude Code's own `WebSearch` tool is enabled for every request. Ask for a
search plainly in the prompt; do **not** register DuckDuckGo, Tavily, or any
other search API merely to use this feature.

```python
response = client.chat.completions.create(
    model="sonnet",
    messages=[
        {
            "role": "user",
            "content": (
                "Use web search to find the current Python release documented "
                "on python.org. Give the version and link to the source."
            ),
        }
    ],
)
print(response.choices[0].message.content)
```

WebSearch runs inside the Claude Code process. It is not returned as an
OpenAI/LangChain function call, and your application must not execute or send
back a tool result for it. Claude Code completes the search and returns its
normal assistant answer. In streaming responses, search progress is carried in
the nonstandard `provider_specific_fields.claude_code_cli_web_search` field;
standard text deltas continue to work normally.

For a runnable LangChain version that saves the resulting answer, run:

```bash
uv run examples/langchain_claude_code_web_search.py
```

It writes `claude_code_web_search_response.md` in the directory from which it
is run.

## Application-defined tools

Use function tools when your application must access its own files, database,
internal services, or business logic. The adapter presents their schemas to
Claude Code, returns requested calls in the standard `tool_calls` field, and
then stops that response. **Your application executes every application tool**
and submits the result in a new completion request.

The following complete loop is the essential pattern:

```python
import json

from openai import OpenAI

client = OpenAI(base_url="http://localhost:9001/v1", api_key="local")


def lookup_customer(customer_id: str) -> dict[str, str]:
    # Replace with your own authorized database or service call.
    return {"customer_id": customer_id, "plan": "team", "status": "active"}


tools = [
    {
        "type": "function",
        "function": {
            "name": "lookup_customer",
            "description": "Look up a customer after authorization has been checked.",
            "parameters": {
                "type": "object",
                "properties": {"customer_id": {"type": "string"}},
                "required": ["customer_id"],
                "additionalProperties": False,
            },
        },
    }
]
functions = {"lookup_customer": lookup_customer}
messages = [{"role": "user", "content": "What plan does customer cust_123 have?"}]

for _ in range(10):
    answer = client.chat.completions.create(model="sonnet", messages=messages, tools=tools).choices[0].message
    messages.append(answer.model_dump(exclude_none=True))

    if not answer.tool_calls:
        print(answer.content)
        break

    for call in answer.tool_calls:
        function = functions.get(call.function.name)
        if function is None:
            raise ValueError(f"Unexpected tool requested: {call.function.name}")
        result = function(**json.loads(call.function.arguments))
        messages.append(
            {"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)}
        )
else:
    raise RuntimeError("Tool loop exceeded ten rounds")
```

Important tool-handling rules:

- Treat tool arguments as untrusted input. Authenticate the end user,
  authorize the operation, validate arguments, set timeouts, and constrain
  file paths or database queries before running anything.
- Preserve each assistant message and return each result with the original
  `tool_call_id`. Submit results for all calls before asking for the next turn.
- Dispatch only tools you registered. Reject unknown tool names rather than
  dynamically importing or executing a model-provided command.
- A tool can return an error payload instead of raising if you want the model
  to recover and choose another approach.
- `tool_choice="none"` prevents application tools from being offered. It does
  not disable Claude Code's built-in WebSearch.

The adapter does not run application functions on its server. This keeps your
credentials and data-access decisions in your application process.

### LangChain tools

Install the development dependencies in this checkout with `uv sync`, then use
the same Chat Completions endpoint with `ChatOpenAI`:

```python
import os

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI


@tool
def customer_plan(customer_id: str) -> str:
    """Return the authorized customer's subscription plan."""
    return f"Customer {customer_id} has the team plan."


model = ChatOpenAI(
    model=os.getenv("CSA_MODEL", "sonnet"),
    base_url=os.getenv("CSA_BASE_URL", "http://localhost:9001/v1"),
    api_key=os.getenv("CSA_API_KEY", "local"),
    use_responses_api=False,
).bind_tools([customer_plan])

messages = [("user", "Use customer_plan for cust_123.")]
for _ in range(10):
    answer = model.invoke(messages)
    messages.append(answer)
    if not answer.tool_calls:
        print(answer.content)
        break
    for call in answer.tool_calls:
        messages.append(customer_plan.invoke(call))
else:
    raise RuntimeError("Tool loop exceeded ten rounds")
```

See [examples/langchain_tools.py](examples/langchain_tools.py) for the small
example, [examples/langgraph_tools.py](examples/langgraph_tools.py) for a
LangGraph loop, and
[examples/langchain_web_research_agent.py](examples/langchain_web_research_agent.py)
for a multi-tool LangChain research workflow that writes a file locally.

## Streaming

Set `stream=True` to receive standard Chat Completions Server-Sent Events.
The stream ends with `[DONE]`. Set `stream_options={"include_usage": True}` to
receive a final usage-only chunk.

```python
stream = client.chat.completions.create(
    model="sonnet",
    messages=[{"role": "user", "content": "Write a haiku about local APIs."}],
    stream=True,
    stream_options={"include_usage": True},
)

for chunk in stream:
    if chunk.choices and (text := chunk.choices[0].delta.content):
        print(text, end="", flush=True)
```

Function-call chunks use stable IDs and indexes. As with non-streaming calls,
execute application tools in the caller and start the next request with the
assistant message plus its tool results.

## Server configuration

The server loads `.env` from its current working directory at startup. Copy
`.env.example` for all available settings. Keep `.env` out of source control
when it contains `CSA_API_KEY`.

| Variable | Default | Description |
| --- | --- | --- |
| `CSA_HOST` | `127.0.0.1` | Bind address. A non-loopback address requires `CSA_API_KEY`. |
| `CSA_PORT` | `9001` | Listening port. `serve --port` takes precedence. |
| `CSA_API_KEY` | unset | Bearer token required by chat, models, and ready endpoints when set. |
| `CSA_MODELS` | `{}` | JSON object mapping public aliases to Claude Code model names. |
| `CLAUDE_CODE_PATH` | `claude` | Claude Code executable path or command name. |
| `CLAUDE_CODE_CWD` | neutral temporary directory | Working directory given to Claude Code. Leave empty unless a specific directory is required. |
| `CLAUDE_CODE_TIMEOUT_SECONDS` | `600` | Per-request CLI timeout, excluding time spent queued. |
| `CLAUDE_CODE_MAX_OUTPUT_TOKENS` | `32000` | Maximum output-token cap sent to Claude Code. |
| `CLAUDE_CODE_MAX_CONCURRENCY` | `4` | Maximum concurrent Claude Code processes. |
| `CLAUDE_CODE_MAX_QUEUE_LENGTH` | `20` | Maximum waiting requests; additional requests receive `429`. |
| `CLAUDE_CODE_SESSION_INACTIVITY_WINDOW_SECONDS` | `14400` | Cleanup window for opt-in resumed sessions. |

### Authentication and network exposure

The default loopback binding is appropriate for one machine. To expose the
server beyond loopback, set a long random API key and terminate TLS at a
trusted reverse proxy; this adapter itself serves HTTP only.

```bash
CSA_HOST=0.0.0.0 \
CSA_API_KEY='replace-with-a-long-random-secret' \
uv run claude-subscription-adapter serve
```

Clients then send it as a bearer token, which the OpenAI SDK does when passed
as `api_key`.

### Model discovery and readiness

```bash
curl -s http://localhost:9001/v1/models | jq
curl -s http://localhost:9001/ready | jq
```

When `CSA_API_KEY` is set, add:

```bash
-H "Authorization: Bearer $CSA_API_KEY"
```

Available routes:

| Method | Route | Purpose |
| --- | --- | --- |
| `POST` | `/v1/chat/completions` | Chat completions, tools, and optional SSE streaming. |
| `GET` | `/v1/models` | Configured public model aliases and their Claude Code targets. |
| `GET` | `/health` | Process liveness; never authenticated. |
| `GET` | `/ready` | Verifies that Claude Code is installed and logged in. |
| `GET` | `/openapi.json` | OpenAPI schema; never authenticated. |

## Conversations and sessions

Requests are stateless by default: send the history you want the model to use.
To opt into Claude Code session continuation, provide a stable
`prompt_cache_key` on related requests. The adapter's session mapping is
in-memory, so restarting the adapter loses it. A JSON value in
`metadata.user_id` with a string `session_id` takes precedence over
`prompt_cache_key`; a LangGraph `thread_id` alone does not enable continuation.

Use session continuation only when the same server process owns the sequence;
run one server worker so its session map and request queue remain consistent.

## Compatibility and limits

The supported surface is intentionally focused:

- Supported message roles are `system`, `developer`, `user`, `assistant`, and
  `tool`; text content and text parts are accepted.
- `n` must be `1`.
- Sampling and structured-output controls accepted by some OpenAI clients are
  forwarded for compatibility but are not enforced by the underlying provider.
- A requested token limit can raise the configured cap but cannot lower it.
- Required/named `tool_choice`, `parallel_tool_calls=false`, and strict JSON
  schemas are not guaranteed. Design tool dispatchers to validate all input.
- Invalid tool definitions may be skipped rather than rejected.
- WebSearch remains enabled even when no application tools are registered.

The adapter strips alternate Anthropic, Bedrock, Vertex, and Foundry routing
credentials before starting Claude Code so requests use the local Claude Code
subscription login.

## Examples

Start the adapter, then run any example from the repository root:

```bash
uv run examples/python_tools.py
uv run examples/langchain_tools.py
uv run examples/langgraph_tools.py
uv run examples/langchain_web_research_agent.py
uv run examples/langchain_claude_code_web_search.py
```

All examples default to `sonnet` and `http://localhost:9001/v1`. Override them
without changing source code:

```bash
CSA_MODEL=your-model CSA_BASE_URL=http://localhost:9100/v1 \
uv run examples/langchain_claude_code_web_search.py
```

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| `/ready` returns `503` | Run `claude auth status --json`; install Claude Code or set `CLAUDE_CODE_PATH` if it is outside `PATH`. |
| Client gets `401` | Set the client's `api_key` to the same value as `CSA_API_KEY`. |
| Client gets `429` | The Claude Code account may be rate limited, or the adapter queue is full. Reduce traffic, wait, or tune concurrency and queue limits. |
| Client gets `504` | Increase `CLAUDE_CODE_TIMEOUT_SECONDS` only after checking the prompt size, model, and tool workflow. |
| A local tool never runs | This is expected: the adapter returns a tool call. Your application must execute it and send a `role="tool"` result with the matching ID. |
| Claude does not search | Explicitly ask it to use WebSearch and ask for sources; make sure the Claude Code account can use web search. |

## Development

```bash
uv run pytest -q
uv run ruff check src examples
uv build
```

See [project-docs/migration-manifest.md](project-docs/migration-manifest.md)
for provenance and
[project-docs/review-notes.md](project-docs/review-notes.md) for release
evidence.
