"""Local protocol types; no provider framework dependency."""

from time import time
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

AllMessageValues = dict[str, Any]
GenericStreamingChunk = dict[str, Any]
ChatCompletionToolCallChunk = dict[str, Any]
ChatCompletionToolCallFunctionChunk = dict[str, Any]
ChatCompletionUsageBlock = dict[str, Any]


class AdapterError(Exception):
    def __init__(
        self,
        *,
        status_code: int,
        message: str,
        code: str = "adapter_error",
        num_retries: int = 0,
        retry_after: int | None = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.code = code
        self.num_retries = num_retries
        self.retry_after = retry_after

    def body(self):
        return {
            "error": {
                "message": self.message,
                "type": "invalid_request_error" if self.status_code == 400 else "server_error",
                "param": None,
                "code": self.code,
            }
        }


class PromptTokensDetailsWrapper(BaseModel):
    text_tokens: int = 0
    cached_tokens: int = 0
    cache_write_tokens: int = 0
    cache_creation_tokens: int = 0


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    prompt_tokens_details: PromptTokensDetailsWrapper = Field(default_factory=PromptTokensDetailsWrapper)
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

    @model_validator(mode="after")
    def cache_details(self):
        self.prompt_tokens_details.cached_tokens = self.cache_read_input_tokens
        self.prompt_tokens_details.cache_write_tokens = self.cache_creation_input_tokens
        self.prompt_tokens_details.cache_creation_tokens = self.cache_creation_input_tokens
        return self


class FunctionCall(BaseModel):
    name: str
    arguments: str


class ToolCall(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: FunctionCall


class Message(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str | None = None
    tool_calls: list[ToolCall] | None = None


class Choices(BaseModel):
    index: int = 0
    message: Message
    finish_reason: str


class ModelResponse(BaseModel):
    id: str = Field(default_factory=lambda: "chatcmpl-" + uuid4().hex)
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time()))
    model: str
    choices: list[Choices]
    usage: Usage


class TextPart(BaseModel):
    type: Literal["text"]
    text: str


class InputMessage(BaseModel):
    role: Literal["system", "developer", "user", "assistant", "tool"]
    content: str | list[TextPart] | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] | None = None


class StreamOptions(BaseModel):
    include_usage: bool = False


class ChatRequest(BaseModel):
    # Preserve the source's acceptance of optional controls (see README for limits).
    model_config = ConfigDict(extra="allow")
    model: str = Field(min_length=1)
    messages: list[InputMessage] = Field(min_length=1)
    stream: bool = False
    stream_options: StreamOptions | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    n: Literal[1] = 1
    metadata: dict[str, Any] | None = None

    @model_validator(mode="after")
    def validate_history(self):
        if not self.model.strip():
            raise ValueError("model must not be blank")
        pending: set[str] = set()
        seen: set[str] = set()
        for message in self.messages:
            if message.role == "tool":
                if message.tool_call_id not in pending:
                    raise ValueError("tool result must reference an outstanding assistant tool_call_id")
                pending.remove(message.tool_call_id)
            else:
                if pending:
                    raise ValueError("provide results for all tool calls before continuing")
                if message.tool_calls and message.role != "assistant":
                    raise ValueError("only assistant messages can contain tool_calls")
                for call in message.tool_calls or []:
                    if not call.id or call.id in seen:
                        raise ValueError("tool call IDs must be nonempty and unique")
                    seen.add(call.id)
                    pending.add(call.id)
        if pending:
            raise ValueError("provide results for all outstanding tool calls")
        return self


def map_finish_reason(reason: str) -> str:
    return {
        "compaction": "length",
        "end_turn": "stop",
        "stop_sequence": "stop",
        "max_tokens": "length",
        "tool_use": "tool_calls",
        "stop": "stop",
        "length": "length",
        "tool_calls": "tool_calls",
        "content_filter": "content_filter",
        "content_filtered": "content_filter",
        "refusal": "content_filter",
    }.get(reason, "stop")
