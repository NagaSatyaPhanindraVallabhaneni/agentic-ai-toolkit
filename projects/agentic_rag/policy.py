"""RagAgentPolicy: the deterministic "brain" driving the agentic RAG demo.

This is a stand-in for a real LLM call — it implements `LLMPolicyProtocol`
from `core.agent`, so swapping it for a GPT-4.1 / Claude-backed policy that
reasons freely about which tool to call requires no change anywhere else
(orchestrator, tools, API layer). What it demonstrates is the *shape* of
agentic decision-making that a real model would be doing:

  1. Route simple arithmetic to the calculator instead of the retriever.
  2. Retrieve, and inspect retrieval *confidence* (the fused RRF score)
     rather than blindly trusting the top hit.
  3. If confidence is low, reformulate the query (strip low-signal words)
     and retry once — a real agentic behavior, not a single-shot RAG call.
  4. Answer with a citation, or admit defeat gracefully after the retry
     budget is spent, rather than hallucinating.
"""

from __future__ import annotations

import re

from core.agent import AgentStep, Decision, LLMPolicyProtocol
from core.tools import Tool

_ARITHMETIC_RE = re.compile(r"[\s\d+\-*/().]+")
_ARITHMETIC_PREFIX_RE = re.compile(r"^\s*(what'?s|what is|calculate|compute|solve)\s*:?\s*", re.IGNORECASE)
_CONFIDENCE_THRESHOLD = 0.3  # top_score (raw dense/bm25 signal, roughly 0-1) below this triggers a retry
_STOPWORDS = {"the", "a", "an", "is", "are", "of", "for", "to", "what", "how", "do", "i", "does", "my"}


def _extract_arithmetic_expression(question: str) -> str | None:
    """Strip a leading 'what is' / 'calculate' / etc. and a trailing '?',
    then check whether what remains is a pure arithmetic expression. Returns
    the cleaned expression if so, else None."""
    stripped = question.strip().rstrip("?").strip()
    stripped = _ARITHMETIC_PREFIX_RE.sub("", stripped).strip()
    if stripped and _ARITHMETIC_RE.fullmatch(stripped) and any(ch.isdigit() for ch in stripped):
        return stripped
    return None


def _reformulate(question: str) -> str:
    """Strip stopwords to sharpen a query that scored low on the first
    retrieval attempt."""
    words = re.findall(r"[a-zA-Z0-9]+", question.lower())
    kept = [w for w in words if w not in _STOPWORDS]
    return " ".join(kept) if kept else question


class RagAgentPolicy(LLMPolicyProtocol):
    def decide(self, question: str, scratchpad: list[AgentStep], tools: dict[str, Tool]) -> Decision:
        if not scratchpad:
            expression = _extract_arithmetic_expression(question)
            if expression is not None:
                return Decision(
                    thought="This looks like an arithmetic question, not a knowledge-base lookup — use the calculator.",
                    action_name="calculator",
                    action_input={"expression": expression},
                )
            return Decision(
                thought="This looks like a knowledge question — retrieve supporting context first.",
                action_name="retrieve_docs",
                action_input={"query": question, "k": 3},
            )

        last = scratchpad[-1]

        if last.action_name == "calculator":
            return Decision(thought="Calculator returned a value — answer directly.", final_answer=last.observation)

        if last.action_name == "retrieve_docs":
            top_score = (last.observation_data or {}).get("top_score", 0.0)
            results = (last.observation_data or {}).get("results", [])

            already_retried = any(s.action_name == "retrieve_docs" for s in scratchpad[:-1])

            if results and (top_score >= _CONFIDENCE_THRESHOLD or already_retried):
                top = results[0]
                if top_score >= _CONFIDENCE_THRESHOLD:
                    answer = f"{top['text']} (source: {top['title']})"
                else:
                    answer = (
                        f"I'm not fully confident, but the closest match I found is: "
                        f"{top['text']} (source: {top['title']})"
                    )
                return Decision(thought="Retrieval confidence is sufficient (or retry budget spent) — answer now.", final_answer=answer)

            reformulated = _reformulate(question)
            return Decision(
                thought=(
                    f"Retrieval confidence was low (score={top_score:.4f}) — reformulating the query "
                    f"from '{question}' to '{reformulated}' and retrying once."
                ),
                action_name="retrieve_docs",
                action_input={"query": reformulated, "k": 3},
            )

        return Decision(thought="Unrecognized tool result — giving up gracefully.", final_answer="I was unable to resolve this question.")
