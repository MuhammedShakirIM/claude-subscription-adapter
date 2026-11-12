"""Use Claude Code's built-in WebSearch through the adapter.

Run the adapter in another terminal, then run this from the repository root:

    uv run examples/langchain_claude_code_web_search.py

This deliberately registers no LangChain search tool and calls no third-party
search API. The adapter launches Claude Code with its built-in ``WebSearch``
tool enabled. Claude Code performs the search internally, and this client saves
Claude's final research response in the current directory.
"""

import os
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model=os.getenv("CSA_MODEL", "sonnet"),
    base_url=os.getenv("CSA_BASE_URL", "http://localhost:9001/v1"),
    api_key=os.getenv("CSA_API_KEY", "local"),
    use_responses_api=False,
)

answer = model.invoke(
    [
        SystemMessage(
            "You are a concise research assistant. Use your built-in WebSearch tool before answering. "
            "Base the answer on the search results and include the most useful source URLs."
        ),
        HumanMessage(
            "Find the current Python release supported by the official Python documentation. "
            "Give a short explanation of why a team might choose it for a new service."
        ),
    ]
)

response = answer.text if isinstance(answer.text, str) else str(answer.content)
destination = Path.cwd() / "claude_code_web_search_response.md"
destination.write_text(response + "\n", encoding="utf-8")

print(response)
print(f"\nSaved Claude Code's web-researched response to {destination}")
