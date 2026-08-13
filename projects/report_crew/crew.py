"""ReportCrew: orchestrates Planner -> Writer -> Critic in an iterative
plan -> draft -> critique -> revise loop, stopping when the critic
approves or a max-iteration budget is spent."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from projects.report_crew.agents import (
    CriticProtocol,
    CritiqueResult,
    Outline,
    PlannerProtocol,
    Report,
    WriterProtocol,
)


@dataclass
class IterationRecord:
    iteration: int
    report: Report
    critique: CritiqueResult


@dataclass
class CrewRunResult:
    topic: str
    outline: Outline
    final_report: Report
    iterations: list[IterationRecord] = field(default_factory=list)
    approved: bool = False


def _sections_with_issues(issues: list[str], outline: Outline) -> set[str]:
    """Map free-text critique issues back to the section names they refer
    to, so the writer knows exactly what to revise."""
    flagged: set[str] = set()
    for issue in issues:
        match = re.search(r"'([^']+)'", issue)
        if match and match.group(1) in outline:
            flagged.add(match.group(1))
    return flagged


class ReportCrew:
    def __init__(self, planner: PlannerProtocol, writer: WriterProtocol, critic: CriticProtocol):
        self.planner = planner
        self.writer = writer
        self.critic = critic

    def run(self, topic: str, max_iterations: int = 3) -> CrewRunResult:
        outline = self.planner.plan(topic)
        revision_sections: set[str] = set()
        iterations: list[IterationRecord] = []
        report: Report = {}

        for i in range(1, max_iterations + 1):
            report = self.writer.draft(outline, topic, revision_sections)
            critique = self.critic.review(report, outline)
            iterations.append(IterationRecord(iteration=i, report=dict(report), critique=critique))

            if critique.approved:
                return CrewRunResult(topic=topic, outline=outline, final_report=report, iterations=iterations, approved=True)

            revision_sections |= _sections_with_issues(critique.issues, outline)

        return CrewRunResult(topic=topic, outline=outline, final_report=report, iterations=iterations, approved=False)
