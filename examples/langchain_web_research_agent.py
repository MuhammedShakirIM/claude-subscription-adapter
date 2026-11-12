"""Research with LangChain tools, then persist the web-search response.

Run the adapter in another terminal, then run this from the repository root:

    uv run examples/langchain_web_research_agent.py

The agent must call ``web_search`` followed by ``save_search_response``. The
second tool writes ``web_search_response.txt`` in the directory from which the
script is run (normally this repository), never in a model-provided directory.
"""

import json
import os
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

LAST_SEARCH_RESPONSE = ""


@tool
def web_search(query: str) -> str:
    """Search the public web and return the response for a focused research query."""
    global LAST_SEARCH_RESPONSE

    url = "https://api.duckduckgo.com/?" + urlencode(
        {"q": query, "format": "json", "no_html": "1", "skip_disambig": "1"}
    )
    request = Request(url, headers={"User-Agent": "claude-subscription-adapter-example/1.0"})
    with urlopen(request, timeout=20) as response:
        payload = json.load(response)

    snippets = [payload.get("AbstractText", "")]
    snippets.extend(topic.get("Text", "") for topic in payload.get("RelatedTopics", []) if "Text" in topic)
    result = "\n".join(snippet for snippet in snippets if snippet).strip()
    LAST_SEARCH_RESPONSE = result or json.dumps(payload, ensure_ascii=False)
    return LAST_SEARCH_RESPONSE


@tool
def save_search_response(filename: str = "web_search_response.txt") -> str:
    """Save the exact most recent web-search response to a text file in the current directory."""
    if not LAST_SEARCH_RESPONSE:
        raise ValueError("Call web_search before saving its response.")
    if Path(filename).name != filename:
        raise ValueError("filename must be a simple filename, not a path.")

    destination = Path.cwd() / filename
    destination.write_text(LAST_SEARCH_RESPONSE + "\n", encoding="utf-8")
    return f"Saved the web-search response to {destination}"


model = ChatOpenAI(
    model=os.getenv("CSA_MODEL", "sonnet"),
    base_url=os.getenv("CSA_BASE_URL", "http://localhost:9001/v1"),
    api_key=os.getenv("CSA_API_KEY", "local"),
    use_responses_api=False,
).bind_tools([web_search, save_search_response])

messages = [
    SystemMessage(
        "You are a research agent. For every request, call web_search first, then call "
        "save_search_response to persist the exact returned search response. Only after both "
        "tools succeed, give a concise answer based on that response and mention the saved file."
    ),
    HumanMessage(
        "Research what the Python programming language is, using a web search. "
        "Save the returned search response as web_search_response.txt, then summarize it for me."
    ),
]

for _ in range(10):
    answer = model.invoke(messages)
    messages.append(answer)
    if not answer.tool_calls:
        print(answer.content)
        break
    for call in answer.tool_calls:
        if call["name"] == "web_search":
            messages.append(web_search.invoke(call))
        elif call["name"] == "save_search_response":
            messages.append(save_search_response.invoke(call))
        else:
            raise ValueError(f"Unexpected tool requested: {call['name']}")
else:
    raise RuntimeError("Agent loop exceeded ten rounds")
