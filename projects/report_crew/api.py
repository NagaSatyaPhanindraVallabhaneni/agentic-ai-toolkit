"""FastAPI app exposing the report-crew demo: give it a topic, watch
Planner -> Writer -> Critic iterate until the draft passes review, with
the full per-iteration trace returned for transparency."""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

from projects.report_crew.agents import CriticAgent, PlannerAgent, WriterAgent
from projects.report_crew.crew import CrewRunResult, ReportCrew

app = FastAPI(
    title="report-crew",
    description="A Planner -> Writer -> Critic multi-agent loop that drafts and iteratively revises a structured report.",
    version="1.0.0",
)

crew = ReportCrew(planner=PlannerAgent(), writer=WriterAgent(), critic=CriticAgent())


class ReportRequest(BaseModel):
    topic: str = Field(..., min_length=1, examples=["zero-trust API security"])
    max_iterations: int = Field(3, ge=1, le=10)


class IterationView(BaseModel):
    iteration: int
    approved: bool
    issues: list[str]


class ReportResponse(BaseModel):
    topic: str
    outline: list[str]
    approved: bool
    iterations_taken: int
    final_report: dict[str, str]
    iterations: list[IterationView]


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/crew/report", response_model=ReportResponse)
def create_report(request: ReportRequest) -> ReportResponse:
    result: CrewRunResult = crew.run(request.topic, max_iterations=request.max_iterations)
    return ReportResponse(
        topic=result.topic,
        outline=result.outline,
        approved=result.approved,
        iterations_taken=len(result.iterations),
        final_report=result.final_report,
        iterations=[
            IterationView(iteration=rec.iteration, approved=rec.critique.approved, issues=rec.critique.issues)
            for rec in result.iterations
        ],
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("projects.report_crew.api:app", host="0.0.0.0", port=8002, reload=True)
