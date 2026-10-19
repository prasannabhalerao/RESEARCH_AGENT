
import json
from ollama import chat


MODEL = "qwen3:8b"

PLANNER_PROMPT = """
You are a task planner for a local AI research agent.

Break the user's goal into 2 to 5 small, actionable tasks.

Rules:
1. Put tasks in a logical order.
2. Each task must have one clear purpose.
3. Use the available project capabilities where relevant:
   - Search project documents.
   - Read or search saved notes.
   - Perform calculations.
   - Summarize retrieved information.
   - Prepare a report.
4. Do not claim that a task has already been completed.
5. Do not execute tasks. Only create the plan.
6. Return valid JSON with this structure:

{
  "goal": "The original user goal",
  "steps": [
    {
      "id": 1,
      "task": "Description of the first task"
    }
  ]
}
"""


def create_plan(goal: str) -> dict:
    """Ask the local LLM to create a structured task plan."""

    response = chat(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": PLANNER_PROMPT,
            },
            {
                "role": "user",
                "content": goal,
            },
        ],
        format="json",
    )

    plan = json.loads(response.message.content)

    if not isinstance(plan, dict):
        raise ValueError("The planner must return a JSON object.")

    steps = plan.get("steps")

    if not isinstance(steps, list) or not steps:
        raise ValueError("The planner returned no valid tasks.")

    if len(steps) > 5:
        raise ValueError("The planner returned more than 5 tasks.")

    for index, step in enumerate(steps, start=1):
        if not isinstance(step, dict):
            raise ValueError("Each task must be a JSON object.")

        if not isinstance(step.get("task"), str):
            raise ValueError("Each task must have a text description.")

        if not step["task"].strip():
            raise ValueError("Task descriptions cannot be empty.")

        step["id"] = index

    plan["goal"] = goal

    return plan


def display_plan(plan: dict) -> None:
    """Print the plan in a readable format."""

    print("\nGOAL")
    print(plan["goal"])

    print("\nGENERATED PLAN")

    for step in plan["steps"]:
        print(f'{step["id"]}. {step["task"]}')


if __name__ == "__main__":
    user_goal = input("Enter your goal: ").strip()

    if not user_goal:
        print("Please enter a goal.")
    else:
        try:
            generated_plan = create_plan(user_goal)
            display_plan(generated_plan)
        except (ValueError, json.JSONDecodeError) as error:
            print(f"Planner error: {error}")
