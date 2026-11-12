import json

import pytest
from httpx import ASGITransport, AsyncClient

from claude_subscription_adapter.api import create_app
from claude_subscription_adapter.config import ServerConfig
from claude_subscription_adapter.service import CompletionService


class Process:
    returncode = None

    def __init__(self, events):
        self.lines = [json.dumps(e).encode() + b"\n" for e in events]

    async def write_stdin(self, data):
        self.input = data

    async def read_stdout_line(self):
        return self.lines.pop(0) if self.lines else b""

    async def read_stderr(self):
        return b""

    def kill(self):
        self.returncode = -9

    async def wait(self):
        if self.returncode is None:
            self.returncode = 0
        return self.returncode


@pytest.mark.parametrize("stream", [False, True])
async def test_http_tool_round_trip(stream):
    runs = [
        [
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {"type": "tool_use", "id": "call-1", "name": "mcp__c__lookup", "input": {"name": "日本"}}
                    ]
                },
            },
            {"type": "stream_event", "event": {"type": "message_delta", "delta": {"stop_reason": "tool_use"}}},
        ],
        [
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "Found it"}]}},
            {"type": "result", "usage": {"input_tokens": 5, "output_tokens": 2}, "stop_reason": "end_turn"},
        ],
    ]
    processes = []

    async def spawn(**kwargs):
        proc = Process(runs.pop(0))
        processes.append(proc)
        return proc

    app = create_app(ServerConfig(), CompletionService(spawner=spawn, environment={}))
    messages = [{"role": "user", "content": "Find 日本"}]
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": messages,
                "tools": [{"type": "function", "function": {"name": "lookup", "parameters": {}}}],
            },
        )
        assert response.status_code == 200, response.text
        answer = response.json()["choices"][0]["message"]
        assert json.loads(answer["tool_calls"][0]["function"]["arguments"]) == {"name": "日本"}
        messages += [answer, {"role": "tool", "tool_call_id": "call-1", "content": "Found"}]
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": messages,
                "stream": stream,
                "stream_options": {"include_usage": True},
            },
        )
        assert response.status_code == 200, response.text
        assert "Found it" in response.text
        assert b"Tool result from" in processes[1].input
        if stream:
            chunks = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: {")]
            assert len({c["id"] for c in chunks}) == 1
            assert chunks[-1]["choices"] == []
            assert chunks[-1]["usage"]["total_tokens"] == 7
            assert response.text.endswith("data: [DONE]\n\n")


async def test_validation_auth_and_schema():
    app = create_app(ServerConfig(api_key="secret", models={"sonnet": "claude-sonnet-5"}))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        assert (await c.get("/health")).status_code == 200
        assert (await c.get("/v1/models")).status_code == 401
        models = await c.get("/v1/models", headers={"Authorization": "Bearer secret"})
        assert models.json()["data"] == [
            {
                "id": "sonnet",
                "object": "model",
                "created": 0,
                "owned_by": "local",
                "claude_code_model": "claude-sonnet-5",
            }
        ]
        assert (await c.get("/openapi.json")).json()["paths"]["/v1/chat/completions"]
        r = await c.post(
            "/v1/chat/completions", headers={"Authorization": "Bearer secret"}, json={"model": "x", "messages": []}
        )
        assert r.status_code == 400
        assert "error" in r.json()


@pytest.mark.parametrize("models", [[], {"": "sonnet"}, {"sonnet": ""}, {"sonnet": 1}])
def test_invalid_model_aliases_are_rejected(models):
    with pytest.raises(ValueError, match="CSA_MODELS"):
        ServerConfig(models=models).validate()


def test_invalid_model_alias_json_is_rejected_at_startup(monkeypatch):
    monkeypatch.setenv("CSA_MODELS", "[]")
    with pytest.raises(ValueError, match="CSA_MODELS"):
        ServerConfig.from_env()


def test_dotenv_is_loaded_without_overwriting_environment(monkeypatch, tmp_path):
    (tmp_path / ".env").write_text('CSA_PORT=9102\nCSA_MODELS={"alias":"from-dotenv"}\n', encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CSA_MODELS", raising=False)
    monkeypatch.setenv("CSA_PORT", "9103")

    config = ServerConfig.from_env()

    assert config.port == 9103
    assert config.models == {"alias": "from-dotenv"}


async def test_unexpected_streaming_failure_emits_an_error_event_without_done(caplog):
    caplog.set_level("INFO")

    class FailingService:
        async def astreaming(self, **kwargs):
            kwargs["on_started"](0.01)
            yield {"text": "partial", "tool_use": None, "is_finished": False, "finish_reason": "", "usage": None}
            raise RuntimeError("private failure details")

    app = create_app(ServerConfig(), FailingService())
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "test-model", "messages": [{"role": "user", "content": "hello"}], "stream": True},
        )

    events = [line[6:] for line in response.text.splitlines() if line.startswith("data: ")]
    assert response.status_code == 200
    assert "partial" in response.text
    assert events[-1] == json.dumps(
        {
            "error": {
                "message": "Unexpected adapter failure",
                "type": "server_error",
                "param": None,
                "code": "internal_error",
            }
        }
    )
    assert "[DONE]" not in response.text
    record = next(record for record in caplog.records if record.message == "adapter request completed")
    assert record.request_id.startswith("chatcmpl-")
    assert record.model == "test-model"
    assert record.queue_wait_seconds == 0.01
    assert record.execution_seconds is not None
    assert record.outcome == "unexpected_error"
