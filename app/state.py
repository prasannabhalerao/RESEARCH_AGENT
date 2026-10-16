from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentState:
    goal: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    tool_results: list[str] = field(default_factory=list)
    steps: int = 0
    completed: bool = False