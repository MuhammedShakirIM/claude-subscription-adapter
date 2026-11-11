"""The caller's OpenAI tools, offered to the CLI through a stub MCP server that lists them but never runs them."""

import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final

from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

MCP_SERVER_NAME: Final = "c"
CLI_TOOL_PREFIX: Final = f"mcp__{MCP_SERVER_NAME}__"
_MAX_TOOL_NAME_LENGTH: Final = 64
_MAX_ALIAS_LENGTH: Final = _MAX_TOOL_NAME_LENGTH - len(CLI_TOOL_PREFIX)
_ALIAS_HASH_LENGTH: Final = 8
_STUB_SERVER_PATH: Final = Path(__file__).with_name("client_tools_server.py")
CALLER_MACHINE_NOTE: Final = (
    "Your tools run on the user's own computer. Every file path, folder and command you pass to a tool must "
    "refer to the user's machine as described in this conversation. The working directory and platform in "
    "your environment details belong to a different server: never use or mention them."
)


class _Function(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str
    description: str = ""
    parameters: dict[str, object] = {"type": "object", "properties": {}}


class _Tool(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: str = "function"
    function: _Function


_TOOLS_ADAPTER: Final = TypeAdapter(tuple[object, ...])
_TOOL_ADAPTER: Final = TypeAdapter(_Tool)


@dataclass(frozen=True, slots=True)
class ClientTool:
    """One caller tool. `alias` is the name the CLI sees; `name` is the one the caller gets back."""

    name: str
    alias: str
    description: str
    parameters: Mapping[str, object]


def _alias(name: str) -> str:
    """Anthropic caps tool names at 64 characters, and the MCP prefix eats into that."""
    if len(name) <= _MAX_ALIAS_LENGTH:
        return name
    digest: Final = hashlib.sha256(name.encode("utf-8")).hexdigest()[:_ALIAS_HASH_LENGTH]
    return f"{name[: _MAX_ALIAS_LENGTH - _ALIAS_HASH_LENGTH - 1]}_{digest}"


def _read_tool(raw: object) -> _Tool | None:
    """One malformed entry is skipped rather than costing the caller every other tool."""
    try:
        tool: Final = _TOOL_ADAPTER.validate_python(raw)
    except ValidationError:
        return None
    return tool if tool.type == "function" else None


def _input_schema(parameters: Mapping[str, object]) -> Mapping[str, object]:
    """OpenAI allows `parameters: {}` for a no-argument tool; Anthropic rejects a schema that is not an object."""
    return MappingProxyType({**parameters, "type": "object"})


def read_client_tools(tools: object, tool_choice: object) -> tuple[ClientTool, ...]:
    """Untrusted request fields become typed tools; `tool_choice: "none"` or anything unreadable offers none."""
    if tools is None or tool_choice == "none":
        return ()
    try:
        entries: Final = _TOOLS_ADAPTER.validate_python(tools)
    except ValidationError:
        return ()
    return tuple(
        ClientTool(
            name=tool.function.name,
            alias=_alias(tool.function.name),
            description=tool.function.description,
            parameters=_input_schema(tool.function.parameters),
        )
        for tool in (_read_tool(entry) for entry in entries)
        if tool is not None
    )


def with_caller_machine_note(system_prompt: str) -> str:
    return f"{system_prompt}\n\n{CALLER_MACHINE_NOTE}" if system_prompt else CALLER_MACHINE_NOTE


def caller_name(tools: Sequence[ClientTool], cli_name: str) -> str | None:
    """The caller's own name for a tool the CLI called, or None when it is not one of the caller's."""
    if not cli_name.startswith(CLI_TOOL_PREFIX):
        return None
    alias: Final = cli_name.removeprefix(CLI_TOOL_PREFIX)
    return next((tool.name for tool in tools if tool.alias == alias), None)


def tools_file_payload(tools: Sequence[ClientTool]) -> str:
    return json.dumps(
        [{"name": tool.alias, "description": tool.description, "inputSchema": dict(tool.parameters)} for tool in tools]
    )


def mcp_config(tools_file: str) -> str:
    """Run the stub by path, not `-m litellm...`, so each request does not import the whole package."""
    return json.dumps(
        {
            "mcpServers": {
                MCP_SERVER_NAME: {"command": sys.executable, "args": [str(_STUB_SERVER_PATH), tools_file]},
            }
        }
    )


__all__ = [
    "CLI_TOOL_PREFIX",
    "MCP_SERVER_NAME",
    "ClientTool",
    "caller_name",
    "mcp_config",
    "read_client_tools",
    "tools_file_payload",
    "with_caller_machine_note",
]
