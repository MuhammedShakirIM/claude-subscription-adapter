import os
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool


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
messages = [("user", "Use multiply to calculate 19 times 23.")]
for _ in range(10):
    answer = model.invoke(messages)
    messages.append(answer)
    if not answer.tool_calls:
        print(answer.content)
        break
    for call in answer.tool_calls:
        messages.append(multiply.invoke(call))
else:
    raise RuntimeError("Tool loop exceeded ten rounds")
