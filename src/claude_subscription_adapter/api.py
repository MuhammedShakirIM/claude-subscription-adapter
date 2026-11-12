"""HTTP and SSE serialization around the extracted completion service."""

import asyncio
import json
import logging
import secrets
from contextlib import asynccontextmanager, suppress
from datetime import timedelta
from time import monotonic, time
from uuid import uuid4

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.exceptions import HTTPException

from .config import ServerConfig
from .schemas import AdapterError, ChatRequest, ModelResponse
from .service import CompletionService
from .sessions import run_sweep_forever
from .transport import build_process_env, read_settings, transcripts_dir

logger = logging.getLogger(__name__)


def create_app(config=None, service=None):
    config = config or ServerConfig.from_env()
    config.validate()
    service = service or CompletionService()

    @asynccontextmanager
    async def lifespan(app):
        settings = read_settings(service._environment)
        sweep = asyncio.create_task(
            run_sweep_forever(
                service.sessions,
                interval=timedelta(minutes=5),
                inactivity_window=timedelta(seconds=settings.session_inactivity_window_seconds),
                transcripts_dir=transcripts_dir(settings.working_directory),
            )
        )
        try:
            yield
        finally:
            sweep.cancel()
            with suppress(asyncio.CancelledError):
                await sweep

    app = FastAPI(title="Claude Subscription Adapter", version="0.1.0", lifespan=lifespan)
    app.state.service = service

    async def authorize(request: Request):
        if config.api_key and not secrets.compare_digest(
            request.headers.get("authorization", ""), "Bearer " + config.api_key
        ):
            raise AdapterError(status_code=401, message="Invalid bearer token", code="invalid_api_key")

    @app.exception_handler(AdapterError)
    async def adapter_error(request, exc):
        return JSONResponse(
            exc.body(),
            status_code=exc.status_code,
            headers={"Retry-After": str(exc.retry_after)} if exc.retry_after else None,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        return JSONResponse(
            AdapterError(
                status_code=400,
                message="Invalid request: " + "; ".join(str(e["msg"]) for e in exc.errors()),
                code="invalid_request",
            ).body(),
            status_code=400,
        )

    @app.exception_handler(HTTPException)
    async def http_error(request, exc):
        return JSONResponse(
            AdapterError(status_code=exc.status_code, message=str(exc.detail), code="http_error").body(),
            status_code=exc.status_code,
        )

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/ready", dependencies=[Depends(authorize)])
    async def ready():
        settings = read_settings(service._environment)
        try:
            proc = await asyncio.create_subprocess_exec(
                settings.executable,
                "auth",
                "status",
                "--json",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=dict(build_process_env(service._environment, settings)),
            )
            try:
                out, _ = await asyncio.wait_for(proc.communicate(), 10)
            finally:
                if proc.returncode is None:
                    proc.kill()
                    await proc.wait()
            logged_in = json.loads(out).get("loggedIn", False)
            if proc.returncode != 0 or not logged_in:
                raise ValueError("login unavailable")
        except (TimeoutError, OSError, ValueError):
            raise AdapterError(status_code=503, message="CLI unavailable or not authenticated", code="cli_not_ready")
        return {"status": "ready"}

    @app.get("/v1/models", dependencies=[Depends(authorize)])
    async def models():
        return {
            "object": "list",
            "data": [
                {
                    "id": alias,
                    "object": "model",
                    "created": 0,
                    "owned_by": "local",
                    "claude_code_model": model,
                }
                for alias, model in config.models.items()
            ],
        }

    @app.post("/v1/chat/completions", response_model=ModelResponse, dependencies=[Depends(authorize)])
    async def completions(body: ChatRequest):
        request_id = "chatcmpl-" + uuid4().hex
        requested_at = monotonic()
        started_at: float | None = None
        queue_wait_seconds: float | None = None
        outcome = "error"

        def on_started(wait_seconds: float) -> None:
            nonlocal queue_wait_seconds, started_at
            queue_wait_seconds = wait_seconds
            started_at = monotonic()

        def log_completion() -> None:
            logger.info(
                "adapter request completed",
                extra={
                    "request_id": request_id,
                    "model": body.model,
                    "queue_wait_seconds": queue_wait_seconds,
                    "execution_seconds": None if started_at is None else monotonic() - started_at,
                    "request_seconds": monotonic() - requested_at,
                    "outcome": outcome,
                },
            )

        data = body.model_dump(exclude_none=True)
        model = config.models.get(body.model, body.model)
        options = {k: v for k, v in data.items() if k not in {"model", "messages", "stream", "metadata"}}
        kwargs = {
            "model": model,
            "messages": data["messages"],
            "optional_params": options,
            "request_metadata": {"metadata": body.metadata or {}},
        }
        if not body.stream:
            try:
                response = await service.acompletion(**kwargs, on_started=on_started)
                response.model = body.model
                outcome = "success"
                return response
            except asyncio.CancelledError:
                outcome = "cancelled"
                raise
            except AdapterError:
                outcome = "adapter_error"
                raise
            except Exception:
                outcome = "unexpected_error"
                logger.exception("adapter request failed", extra={"request_id": request_id, "model": body.model})
                raise
            finally:
                log_completion()
        stream = service.astreaming(**kwargs, on_started=on_started)
        try:
            first = await anext(stream)
        except asyncio.CancelledError:
            await stream.aclose()
            outcome = "cancelled"
            log_completion()
            raise
        except AdapterError:
            await stream.aclose()
            outcome = "adapter_error"
            log_completion()
            raise
        except Exception:
            await stream.aclose()
            outcome = "unexpected_error"
            logger.exception(
                "adapter stream failed before response", extra={"request_id": request_id, "model": body.model}
            )
            log_completion()
            raise
        identity = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": int(time()),
            "model": body.model,
        }

        def event(delta, finish=None, choices=True, usage=None):
            payload = {
                **identity,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}] if choices else [],
            }
            if usage is not None:
                payload["usage"] = usage
            return "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"

        async def generate():
            nonlocal outcome
            try:
                yield event({"role": "assistant", "content": ""})
                chunk = first
                while True:
                    delta = {}
                    if chunk.get("text"):
                        delta["content"] = chunk["text"]
                    if chunk.get("tool_use"):
                        delta["tool_calls"] = [chunk["tool_use"]]
                    if delta or chunk["is_finished"]:
                        yield event(delta, chunk["finish_reason"] if chunk["is_finished"] else None)
                    if chunk["is_finished"] and body.stream_options and body.stream_options.include_usage:
                        yield event({}, choices=False, usage=chunk["usage"])
                    try:
                        chunk = await anext(stream)
                    except StopAsyncIteration:
                        break
                yield "data: [DONE]\n\n"
                outcome = "success"
            except asyncio.CancelledError:
                outcome = "cancelled"
                raise
            except AdapterError as exc:
                outcome = "adapter_error"
                yield "data: " + json.dumps(exc.body()) + "\n\n"
            except Exception:
                outcome = "unexpected_error"
                logger.exception("adapter stream failed", extra={"request_id": request_id, "model": body.model})
                error = AdapterError(status_code=500, message="Unexpected adapter failure", code="internal_error")
                yield "data: " + json.dumps(error.body()) + "\n\n"
            finally:
                try:
                    await stream.aclose()
                finally:
                    log_completion()

        return StreamingResponse(generate(), media_type="text/event-stream")

    return app
