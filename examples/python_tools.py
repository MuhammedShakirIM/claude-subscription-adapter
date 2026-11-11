"""Run against the local server: uv run examples/python_tools.py"""

import json
import os
from openai import OpenAI

client = OpenAI(
    base_url=os.getenv("CSA_BASE_URL", "http://localhost:9001/v1"), api_key=os.getenv("CSA_API_KEY", "local")
)
tools = [
    {
        "type": "function",
        "function": {
            "name": "multiply",
            "description": "Multiply two numbers",
            "parameters": {
                "type": "object",
                "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                "required": ["a", "b"],
            },
        },
    }
]
functions = {"multiply": lambda a, b: a * b}
messages = [{"role": "user", "content": "Use multiply to calculate 19 times 23."}]
for _ in range(10):
    answer = (
        client.chat.completions.create(model=os.getenv("CSA_MODEL", "sonnet"), messages=messages, tools=tools)
        .choices[0]
        .message
    )
    messages.append(answer.model_dump(exclude_none=True))
    if not answer.tool_calls:
        print(answer.content)
        break
    for call in answer.tool_calls:
        result = functions[call.function.name](**json.loads(call.function.arguments))
        messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)})
else:
    raise RuntimeError("Tool loop exceeded ten rounds")
