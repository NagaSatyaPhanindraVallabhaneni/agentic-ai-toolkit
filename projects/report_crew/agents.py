"""Planner -> Writer -> Critic agents for the report-crew demo.

Every agent here is a deterministic, dependency-free stand-in — exactly
the same design principle as the RAG project's policy: implement the
matching protocol with a real LLM call (one prompt for planning, one for
drafting, one for critiquing) and the orchestration in `crew.py` does not
change at all.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class CritiqueResult:
    approved: bool
    issues: list[str] = field(default_factory=list)


Outline = list[str]
Report = dict[str, str]  # section title -> section body


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


class PlannerProtocol(Protocol):
    def plan(self, topic: str) -> Outline: ...


class WriterProtocol(Protocol):
    def draft(self, outline: Outline, topic: str, revision_sections: set[str] | None = None) -> Report: ...


class CriticProtocol(Protocol):
    def review(self, report: Report, outline: Outline) -> CritiqueResult: ...


# ---------------------------------------------------------------------------
# Deterministic default implementations
# ---------------------------------------------------------------------------


class PlannerAgent(PlannerProtocol):
    """Breaks a topic into a fixed, sensible outline. A real deployment
    would have an LLM propose the outline based on the topic instead of
    using a fixed template."""

    def plan(self, topic: str) -> Outline:
        return [
            "Introduction",
            f"Why {topic} Matters",
            f"Key Challenges in {topic}",
            f"Practical Recommendations for {topic}",
            "Conclusion",
        ]


class WriterAgent(WriterProtocol):
    """Drafts each section from short templated sentences. Deliberately
    writes a thin (2-sentence) first pass for every section; when the
    critic flags a section as too short, the next draft expands exactly
    those sections — a real (if simple) revision loop, not a scripted
    pass/fail."""

    BASE_SENTENCES = 2
    EXPANDED_SENTENCES = 4

    def draft(self, outline: Outline, topic: str, revision_sections: set[str] | None = None) -> Report:
        revision_sections = revision_sections or set()
        report: Report = {}
        for section in outline:
            n = self.EXPANDED_SENTENCES if section in revision_sections else self.BASE_SENTENCES
            report[section] = self._write_section(section, topic, n)
        return report

    @staticmethod
    def _write_section(section: str, topic: str, n_sentences: int) -> str:
        pool = [
            f"{section} is a critical part of understanding {topic}.",
            f"In practice, teams evaluating {topic} should pay close attention to {section.lower()}.",
            f"A concrete example illustrates how {section.lower()} shapes outcomes for {topic}.",
            f"Overall, {section.lower()} should be revisited regularly as {topic} evolves.",
            f"Stakeholders working on {topic} often cite {section.lower()} as a deciding factor.",
            f"Ignoring {section.lower()} is one of the most common mistakes teams make with {topic}.",
        ]
        return " ".join(pool[: max(1, n_sentences)])


class CriticAgent(CriticProtocol):
    """Checks a draft against a simple, explicit rubric and returns
    concrete, section-addressable issues the writer can act on."""

    def __init__(self, min_sentences: int = 3):
        self.min_sentences = min_sentences

    def review(self, report: Report, outline: Outline) -> CritiqueResult:
        issues: list[str] = []

        for section in outline:
            body = report.get(section, "")
            if not body.strip():
                issues.append(f"Section '{section}' is missing or empty.")
                continue
            sentence_count = len([s for s in re.split(r"(?<=[.!?])\s+", body.strip()) if s])
            if sentence_count < self.min_sentences:
                issues.append(f"Section '{section}' has only {sentence_count} sentence(s) (minimum {self.min_sentences}).")

        return CritiqueResult(approved=not issues, issues=issues)
