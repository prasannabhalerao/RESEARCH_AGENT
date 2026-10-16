from ollama import chat
from app.tools import calculator


MODEL = "qwen3:8b"

SYSTEM_PROMPT = """
You are a helpful AI assistant.

You have access to a calculator tool.
Use the calculator for arithmetic calculations.
After receiving the calculator result, explain the answer clearly.
Do not invent calculation results.
"""


def run_agent(user_goal: str) -> None:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_goal},
    ]

    available_tools = {
        "calculator": calculator,
    }

    for step in range(5):
        response = chat(
            model=MODEL,
            messages=messages,
            tools=[calculator],
        )

        # Save the model's response, including any tool calls.
        messages.append(response.message)

        # If the model did not request a tool, show its answer.
        if not response.message.tool_calls:
            print("\nFinal answer:")
            print(response.message.content)
            return

        # Execute each tool requested by the model.
        for tool_call in response.message.tool_calls:
            tool_name = tool_call.function.name
            arguments = tool_call.function.arguments

            print(f"Tool requested: {tool_name}")
            print(f"Arguments: {arguments}")

            function_to_call = available_tools.get(tool_name)

            if function_to_call is None:
                print("Unknown tool requested.")
                return

            try:
                result = function_to_call(**arguments)

            except (ValueError, TypeError, KeyError) as error:
                result = f"Tool error: {error}"

            print(f"Tool result: {result}")

            # Return the tool result to the model.
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