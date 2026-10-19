
from app.planner import create_plan
from app.state import initialize_state, display_state


goal = "Analyze my project documents and explain the technologies used."

plan = create_plan(goal)
state = initialize_state(goal, plan)

display_state(state)

assert state.goal == goal
assert len(state.plan) > 0
assert all(task["status"] == "pending" for task in state.plan)

print("\nState initialization test: PASS")
