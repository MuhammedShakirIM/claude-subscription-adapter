"""Stub MCP server the CLI launches per request: lists the caller's tools, never runs them.

Run by file path with a tools file argument. Imports only `mcp` and the standard library, because
importing `litellm` here would add seconds to every request.
"""

import asyncio
import sys
from pathlib import Path
from typing import Final

import mcp.types as mcp_types
from mcp.server.context import ServerRequestContext
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server
from pydantic import BaseModel, TypeAdapter


class _ListedTool(BaseModel):
    name: str
    description: str = ""
    inputSchema: dict[str, object]


_LISTED_ADAPTER: Final = TypeAdapter(tuple[_ListedTool, ...])


def load_tools(path: str) -> tuple[mcp_types.Tool, ...]:
    listed: Final = _LISTED_ADAPTER.validate_json(Path(path).read_text(encoding="utf-8"))
    return tuple(
        mcp_types.Tool(name=tool.name, description=tool.description, inputSchema=tool.inputSchema) for tool in listed
    )


def build_server(tools: tuple[mcp_types.Tool, ...]) -> Server:
    async def list_tools(
        ctx: ServerRequestContext[dict[str, object]], params: mcp_types.PaginatedRequestParams | None
    ) -> mcp_types.ListToolsResult:
        return mcp_types.ListToolsResult(tools=list(tools))

    async def call_tool(
        ctx: ServerRequestContext[dict[str, object]], params: mcp_types.CallToolRequestParams
    ) -> mcp_types.CallToolResult:
        """The provider stops the CLI before this could matter; the caller runs the tool, not this box."""
        await asyncio.Event().wait()
        return mcp_types.CallToolResult(content=[])

    return Server("client-tools", on_list_tools=list_tools, on_call_tool=call_tool)


async def _serve(tools_path: str) -> None:
    server: Final = build_server(load_tools(tools_path))
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(_serve(sys.argv[1]))
