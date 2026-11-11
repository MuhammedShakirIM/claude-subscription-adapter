import sys

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from claude_subscription_adapter.tools import _STUB_SERVER_PATH, read_client_tools, tools_file_payload


@pytest.mark.asyncio
async def test_the_stub_lists_the_callers_tools_over_stdio(tmp_path):
    """Runs the real stub process the CLI would start, so a broken server fails here, not in production."""
    schema = {"type": "object", "properties": {"filePath": {"type": "string"}}, "required": ["filePath"]}
    tools_file = tmp_path / "tools.json"
    tools_file.write_text(
        tools_file_payload(
            read_client_tools(
                [{"type": "function", "function": {"name": "create_file", "description": "d", "parameters": schema}}],
                None,
            )
        ),
        encoding="utf-8",
    )
    server = StdioServerParameters(command=sys.executable, args=[str(_STUB_SERVER_PATH), str(tools_file)])

    async with stdio_client(server) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        listed = (await session.list_tools()).tools

    assert [(tool.name, tool.description, tool.input_schema) for tool in listed] == [("create_file", "d", schema)]
