"""Agent step tracking and the policy protocol every agent's "brain"
implements."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from core.tools import Tool


@dataclass
class AgentStep:
    """One Thought -> Action -> Observation cycle in the scratchpad."""

    step_number: int
    thought: str
    action_name: str | None
    action_input: dict[str, Any] | None
    observation: str | None = None
    observation_data: dict[str, Any] | None = None


@dataclass
class Decision:
    """What a policy returns: either a tool call (action_name + input) or
    a final answer, never both."""

    thought: str
    action_name: str | None = None
    action_input: dict[str, Any] = field(default_factory=dict)
    final_answer: str | None = None

    def __post_init__(self) -> None:
        if (self.action_name is None) == (self.final_answer is None):
            raise ValueError("Decision must set exactly one of action_name or final_answer")


class LLMPolicyProtocol(Protocol):
    """The "brain" of an agent. Implement this to plug in a real LLM call
    (Claude, GPT-4.1, etc.) — the orchestrator and tools never change,
    only what decides the next action. Every policy in this repo is a
    deterministic, dependency-free stand-in so the demos run with no API
    keys."""

    def decide(self, question: str, scratchpad: list[AgentStep], tools: dict[str, Tool]) -> Decision: ...
