"""ReAct-style (Reason + Act) orchestrator: repeatedly asks the policy for
a Thought and either an Action (tool call) or a Final Answer, executes the
tool, feeds the Observation back in, and stops on a final answer or a
step budget."""

from __future__ import annotations

from dataclasses import dataclass, field

from core.agent import AgentStep, Decision, LLMPolicyProtocol
from core.tools import ToolRegistry


@dataclass
class AgentRunResult:
    question: str
    final_answer: str
    steps: list[AgentStep] = field(default_factory=list)
    resolved: bool = True


class ReActOrchestrator:
    def __init__(self, policy: LLMPolicyProtocol, tools: ToolRegistry, max_steps: int = 5):
        self.policy = policy
        self.tools = tools
        self.max_steps = max_steps

    def run(self, question: str) -> AgentRunResult:
        scratchpad: list[AgentStep] = []

        for step_number in range(1, self.max_steps + 1):
            decision: Decision = self.policy.decide(question, scratchpad, self.tools.as_dict())

            if decision.final_answer is not None:
                return AgentRunResult(question=question, final_answer=decision.final_answer, steps=scratchpad, resolved=True)

            tool = self.tools.get(decision.action_name)
            if tool is None:
                observation = f"error: unknown tool '{decision.action_name}'"
                observation_data = None
            else:
                result = tool.run(**decision.action_input)
                observation = result.output
                observation_data = result.data

            scratchpad.append(
                AgentStep(
                    step_number=step_number,
                    thought=decision.thought,
                    action_name=decision.action_name,
                    action_input=decision.action_input,
                    observation=observation,
                    observation_data=observation_data,
                )
            )

        return AgentRunResult(
            question=question,
            final_answer="I could not resolve this within the allotted number of steps.",
            steps=scratchpad,
            resolved=False,
        )
