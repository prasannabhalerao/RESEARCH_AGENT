from app.state import AgentState

state = AgentState(
    goal="Calculate 500 + 800 + 200"
)

print("Goal:", state.goal)
print("Steps:", state.steps)
print("Completed:", state.completed)

state.steps += 1
state.tool_results.append("Total: 1500")
state.completed = True

print("Updated steps:", state.steps)
print("Tool results:", state.tool_results)
print("Completed:", state.completed)