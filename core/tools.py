"""Tool protocol shared by every agent in this repo, plus a couple of
generic tools (calculator) used across projects."""

from __future__ import annotations

import ast
import operator
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class ToolResult:
    """What a tool hands back to the orchestrator: a human-readable
    observation string plus optional structured data for the policy to
    inspect (e.g. retrieval scores, confidence)."""

    output: str
    data: dict[str, Any] | None = None


class Tool(Protocol):
    name: str
    description: str

    def run(self, **kwargs: Any) -> ToolResult: ...


_SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(node: ast.AST) -> float:
    """Evaluate a restricted arithmetic AST. Only numbers and +-*/() and
    unary +/- are allowed — no names, calls, attribute access, or
    subscripting reach this function, so it is safe to run on untrusted
    input, unlike a bare `eval()`."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPERATORS:
        return _SAFE_OPERATORS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPERATORS:
        return _SAFE_OPERATORS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"unsupported expression: {ast.dump(node)}")


class CalculatorTool:
    name = "calculator"
    description = "Evaluate a basic arithmetic expression, e.g. '12 * (7 + 3)'."

    def run(self, expression: str, **_: Any) -> ToolResult:
        try:
            tree = ast.parse(expression, mode="eval")
            value = _safe_eval(tree.body)
        except Exception as exc:  # noqa: BLE001 - deliberately broad, surfaced as a tool error
            return ToolResult(output=f"error: could not evaluate '{expression}' ({exc})", data={"error": str(exc)})
        return ToolResult(output=str(value), data={"value": value})


class ToolRegistry:
    """Simple name -> Tool lookup, shared by the orchestrator."""

    def __init__(self, tools: list[Tool]):
        self._tools = {t.name: t for t in tools}

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def describe(self) -> dict[str, str]:
        return {name: tool.description for name, tool in self._tools.items()}

    def as_dict(self) -> dict[str, Tool]:
        return dict(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools
