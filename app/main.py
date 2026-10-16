from ollama import chat

from app.tools import calculator, save_note, read_notes


MODEL = "qwen3:8b"

SYSTEM_PROMPT = """
You are a helpful local AI agent.

You have three tools:
1. calculator: perform arithmetic.
2. save_note: save a note requested by the user.
3. read_notes: read previously saved notes.

Use the appropriate tool when needed.
After receiving a tool result, explain the result to the user.
Do not claim a note was saved unless the tool confirms success.
"""


def run_agent(user_goal: str) -> None:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_goal},
    ]

    available_tools = {
        "calculator": calculator,
        "save_note": save_note,
        "read_notes": read_notes,
    }

    for step in range(5):
        response = chat(
            model=MODEL,
            messages=messages,
            tools=[calculator, save_note, read_notes],
        )

        messages.append(response.message)

        if not response.message.tool_calls:
            print("\nFinal answer:")
            print(response.message.content)
            return

        for tool_call in response.message.tool_calls:
            tool_name = tool_call.function.name
            arguments = tool_call.function.arguments

            print(f"\nTool requested: {tool_name}")

            function_to_call = available_tools.get(tool_name)

            if function_to_call is None:
                print("Unknown tool requested.")
                return

            try:
                result = function_to_call(**arguments)
            except (ValueError, TypeError, KeyError, OSError) as error:
                result = f"Tool error: {error}"

            print(f"Tool result: {result}")

            messages.append(
                {
                    "role": "tool",
                    "name": tool_name,
                    "content": str(result),
                }
            )

    print("Agent stopped: maximum number of steps reached.")


if __name__ == "__main__":
    goal = input("Enter your goal: ")
    run_agent(goal)