import json

from ollama import chat
from app.tools import calculator


MODEL = "qwen3:8b"

SYSTEM_PROMPT = """
You are an AI agent that can use a calculator.

You must respond with valid JSON only.

If a calculation is needed, return:
{
    "action": "calculator",
    "a": 125,
    "b": 48,
    "operation": "multiply"
}

Use only these operations:
add, subtract, multiply, divide.

If the calculation result has been provided and you can answer,
return:
{
    "action": "final",
    "answer": "Your final answer here"
}

Do not invent calculation results.
"""


def run_agent(user_goal):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_goal},
    ]

    # Limit the loop to avoid endless execution.
    for step in range(5):
        response = chat(
            model=MODEL,
            messages=messages,
            format="json",
            options={"temperature": 0},
        )

        content = response.message.content

        try:
            decision = json.loads(content)
        except json.JSONDecodeError:
            print("The model returned invalid JSON.")
            print(content)
            return

        action = decision.get("action")

        if action == "calculator":
            try:
                a = float(decision["a"])
                b = float(decision["b"])
                operation = decision["operation"]

                if operation not in {
                    "add", "subtract", "multiply", "divide"
                }:
                    print("Unsupported operation requested.")
                    return

                result = calculator(a, b, operation)

            except (KeyError, TypeError, ValueError) as error:
                print(f"Invalid calculator request: {error}")
                return

            print(f"Agent used calculator. Result: {result}")

            # Send the tool result back to the LLM.
            messages.append(
                {"role": "assistant", "content": content}
            )
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Calculator result: {result}. "
                        "Now provide the final answer as JSON."
                    ),
                }
            )

        elif action == "final":
            print("\nFinal answer:")
            print(decision.get("answer", "No answer provided."))
            return

        else:
            print(f"Unknown action: {action}")
            return

    print("Agent stopped: maximum number of steps reached.")


if __name__ == "__main__":
    goal = input("Enter your goal: ")
    run_agent(goal)