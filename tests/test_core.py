from core.agent import AgentStep, Decision
from core.orchestrator import ReActOrchestrator
from core.tools import CalculatorTool, Tool, ToolRegistry, ToolResult


def test_calculator_evaluates_arithmetic():
    result = CalculatorTool().run(expression="12 * (7 + 3)")
    assert result.output == "120"
    assert result.data["value"] == 120


def test_calculator_rejects_unsafe_input():
    result = CalculatorTool().run(expression="__import__('os').system('echo hi')")
    assert result.output.startswith("error:")


def test_calculator_rejects_names():
    result = CalculatorTool().run(expression="os.system('ls')")
    assert result.output.startswith("error:")


def test_tool_registry_lookup_and_describe():
    registry = ToolRegistry([CalculatorTool()])
    assert "calculator" in registry
    assert registry.get("calculator") is not None
    assert registry.get("missing") is None
    assert registry.describe() == {"calculator": CalculatorTool.description}


def test_decision_requires_exactly_one_of_action_or_final_answer():
    Decision(thought="ok", action_name="calculator", action_input={})
    Decision(thought="ok", final_answer="done")
    try:
        Decision(thought="bad", action_name="calculator", final_answer="done")
        assert False, "expected ValueError"
    except ValueError:
        pass
    try:
        Decision(thought="bad")
        assert False, "expected ValueError"
    except ValueError:
        pass


class _EchoPolicy:
    """Trivial policy for testing the orchestrator mechanics in isolation
    from any specific project's decision logic: calls the calculator once,
    then answers with the observation."""

    def decide(self, question: str, scratchpad: list[AgentStep], tools: dict[str, Tool]) -> Decision:
        if not scratchpad:
            return Decision(thought="call calculator", action_name="calculator", action_input={"expression": question})
        return Decision(thought="answer with observation", final_answer=scratchpad[-1].observation)


class _NeverFinishesPolicy:
    def decide(self, question, scratchpad, tools) -> Decision:
        return Decision(thought="keep going", action_name="calculator", action_input={"expression": "1+1"})


def test_orchestrator_runs_tool_then_answers():
    orchestrator = ReActOrchestrator(policy=_EchoPolicy(), tools=ToolRegistry([CalculatorTool()]), max_steps=5)
    result = orchestrator.run("3 + 4")
    assert result.resolved is True
    assert result.final_answer == "7"
    assert len(result.steps) == 1
    assert result.steps[0].action_name == "calculator"


def test_orchestrator_stops_at_max_steps_if_never_resolved():
    orchestrator = ReActOrchestrator(policy=_NeverFinishesPolicy(), tools=ToolRegistry([CalculatorTool()]), max_steps=3)
    result = orchestrator.run("anything")
    assert result.resolved is False
    assert len(result.steps) == 3
    assert "could not resolve" in result.final_answer.lower()


def test_orchestrator_handles_unknown_tool_gracefully():
    class _BadToolPolicy:
        def decide(self, question, scratchpad, tools) -> Decision:
            if not scratchpad:
                return Decision(thought="call a tool that doesn't exist", action_name="not_a_real_tool", action_input={})
            return Decision(thought="give up", final_answer=scratchpad[-1].observation)

    orchestrator = ReActOrchestrator(policy=_BadToolPolicy(), tools=ToolRegistry([CalculatorTool()]), max_steps=3)
    result = orchestrator.run("anything")
    assert "unknown tool" in result.final_answer.lower()


def test_tool_result_dataclass_defaults():
    result = ToolResult(output="ok")
    assert result.data is None
