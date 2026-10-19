
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentState:
    goal: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    tool_results: list[str] = field(default_factory=list)
    steps: int = 0
    completed: bool = False
    plan: list[dict[str, Any]] = field(default_factory=list)
    current_task_index: int = 0


def initialize_state(goal: str, plan: dict) -> AgentState:
    """Create an agent state from a goal and its generated plan."""

    tasks = []

    for step in plan["steps"]:
        tasks.append({
            "id": step["id"],
            "task": step["task"],
            "status": "pending",
            "result": None,
        })

    return AgentState(
        goal=goal,
        plan=tasks,
    )


def display_state(state: AgentState) -> None:
    """Display the goal and current task progress."""

    print("\nAGENT STATE")
    print(f"Goal: {state.goal}")
    print(f"Completed tasks: {sum(task['status'] == 'completed' for task in state.plan)} / {len(state.plan)}")

    for task in state.plan:
        print(f"{task['id']}. [{task['status']}] {task['task']}")
