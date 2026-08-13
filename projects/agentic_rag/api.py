"""FastAPI app exposing the agentic RAG demo: ask a question, watch the
agent decide whether to retrieve, judge its own confidence, retry with a
reformulated query if needed, and answer with a citation — full step
trace included in the response for transparency."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field

from core.orchestrator import AgentRunResult, ReActOrchestrator
from core.tools import CalculatorTool, ToolRegistry
from projects.agentic_rag.corpus import SEED_DOCUMENTS
from projects.agentic_rag.policy import RagAgentPolicy
from projects.agentic_rag.retriever import build_index_from_documents
from projects.agentic_rag.tools import RetrieveDocsTool


class AppState:
    def __init__(self) -> None:
        self.orchestrator: ReActOrchestrator | None = None

    def reset(self, documents: list[dict]) -> None:
        retriever = build_index_from_documents(documents)
        tools = ToolRegistry([RetrieveDocsTool(retriever), CalculatorTool()])
        self.orchestrator = ReActOrchestrator(policy=RagAgentPolicy(), tools=tools, max_steps=4)


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    state.reset(list(SEED_DOCUMENTS))
    yield


app = FastAPI(
    title="agentic-rag",
    description="A tool-calling agent that decides when to retrieve, judges its own retrieval confidence, and retries before answering.",
    version="1.0.0",
    lifespan=lifespan,
)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, examples=["How often should I rotate my API keys?"])


class StepView(BaseModel):
    step_number: int
    thought: str
    action_name: str | None
    action_input: dict | None
    observation: str | None


class AskResponse(BaseModel):
    question: str
    answer: str
    resolved: bool
    steps: list[StepView]


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/agent/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    result: AgentRunResult = state.orchestrator.run(request.question)
    return AskResponse(
        question=result.question,
        answer=result.final_answer,
        resolved=result.resolved,
        steps=[
            StepView(
                step_number=s.step_number,
                thought=s.thought,
                action_name=s.action_name,
                action_input=s.action_input,
                observation=s.observation,
            )
            for s in result.steps
        ],
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("projects.agentic_rag.api:app", host="0.0.0.0", port=8001, reload=True)
