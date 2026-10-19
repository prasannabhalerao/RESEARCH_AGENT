
from ollama import chat

from app.planner import create_plan
from app.state import initialize_state, display_state
from app.tools import (
    calculator,
    save_note,
    read_notes,
    search_notes,
    search_documents,
)


MODEL = "qwen3:8b"

AVAILABLE_TOOLS = {
    "calculator": calculator,
    "save_note": save_note,
    "read_notes": read_notes,
    "search_notes": search_notes,
    "search_documents": search_documents,
}

SYSTEM_PROMPT = """
You are the task executor for a local AI research agent.

Execute the current task using the available tools when needed.

Rules:
1. Use search_documents for questions about project documents.
2. Use search_notes or read_notes for saved notes.
3. Use calculator for arithmetic when appropriate.
4. Use save_note only when saving information is explicitly requested.
5. Base factual claims about retrieved documents on tool results.
6. Do not claim a tool succeeded unless it returned a result.
7. If information is unavailable, say so honestly.
8. If the task is summarization or report writing, use the
   supplied previous task results as evidence.
"""


def execute_task(
    goal: str,
    task: str,
    previous_results: list[str],
) -> str:
    """Execute one planned task using the available tools."""

    context = "\n\n".join(previous_results)

    user_message = f"""
Original goal:
{goal}

Current task:
{task}

Results from previous tasks:
{context if context else "No previous task results."}

Complete the current task. Use tools when needed.
"""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    for _ in range(5):
        response = chat(
            model=MODEL,
            messages=messages,
            tools=[
                calculator,
                save_note,
                read_notes,
                search_notes,
                search_documents,
            ],
        )

        messages.append(response.message)

        if not response.message.tool_calls:
            return response.message.content or "No answer was returned."

        for tool_call in response.message.tool_calls:
            tool_name = tool_call.function.name
            arguments = tool_call.function.arguments

            print(f"    Tool requested: {tool_name}")

            function_to_call = AVAILABLE_TOOLS.get(tool_name)

            if function_to_call is None:
                result = f"Tool error: Unknown tool '{tool_name}'."
            else:
                try:
                    result = function_to_call(**arguments)
                except (ValueError, TypeError, KeyError, OSError) as error:
                    result = f"Tool error: {error}"

            print(f"    Tool result: {result}")

            messages.append(
                {
                    "role": "tool",
                    "name": tool_name,
                    "content": str(result),
                }
            )

    return "Task stopped because the tool-call limit was reached."


def run_plan(goal: str) -> None:
    """Create a plan, execute its tasks, and track their state."""

    plan = create_plan(goal)
    state = initialize_state(goal, plan)

    print("\nGenerated plan:")
    display_state(state)

    for index, task in enumerate(state.plan):
        state.current_task_index = index
        task["status"] = "running"
        state.steps += 1

        print(f"\nExecuting task {task['id']}: {task['task']}")

        try:
            result = execute_task(
                goal=state.goal,
                task=task["task"],
                previous_results=state.tool_results,
            )

            task["result"] = result
            task["status"] = "completed"
            state.tool_results.append(
                f"Task {task['id']}: {task['task']}\n"
                f"Result: {result}"
            )

            print(f"Task status: {task['status']}")
            print(f"Task result: {result}")

        except Exception as error:
            task["status"] = "failed"
            task["result"] = str(error)

            print(f"Task status: failed")
            print(f"Task error: {error}")

            # Stop rather than pretending later tasks succeeded.
            break

    state.completed = all(
        task["status"] == "completed"
        for task in state.plan
    )

    display_state(state)

    if state.tool_results:
        print("\nFINAL REPORT")

        report_response = chat(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Prepare a concise final report for the user's "
                        "original goal using the task results provided. "
                        "Do not invent facts or claim failed tasks succeeded. "
                        "Identify missing information when relevant."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Original goal:\n{goal}\n\n"
                        f"Task results:\n"
                        + "\n\n".join(state.tool_results)
                        + "\n\n"
                        + (
                            "All tasks completed."
                            if state.completed
                            else "The plan did not complete successfully. "
                            "Clearly disclose any incomplete tasks."
                        )
                    ),
                },
            ],
        )

        print(report_response.message.content or "No final report returned.")

    if state.completed:
        print("\nPlan completed successfully.")
    else:
        print("\nPlan incomplete. Review the task statuses above.")


if __name__ == "__main__":
    goal = input("Enter your goal: ").strip()

    if not goal:
        print("Please enter a goal.")
    else:
        run_plan(goal)
