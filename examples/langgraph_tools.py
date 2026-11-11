import os
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b


model = ChatOpenAI(
    model=os.getenv("CSA_MODEL", "sonnet"),
    base_url=os.getenv("CSA_BASE_URL", "http://localhost:9001/v1"),
    api_key=os.getenv("CSA_API_KEY", "local"),
    use_responses_api=False,
).bind_tools([multiply])


def chat(state):
    return {"messages": [model.invoke(state["messages"])]}


builder = StateGraph(MessagesState)
builder.add_node("chat", chat)
builder.add_node("tools", ToolNode([multiply]))
builder.add_edge(START, "chat")
builder.add_conditional_edges("chat", tools_condition, {"tools": "tools", "__end__": "__end__"})
builder.add_edge("tools", "chat")
graph = builder.compile(checkpointer=InMemorySaver())
result = graph.invoke(
    {"messages": [("user", "Use multiply to calculate 19 times 23.")]}, {"configurable": {"thread_id": "example"}}
)
print(result["messages"][-1].content)
