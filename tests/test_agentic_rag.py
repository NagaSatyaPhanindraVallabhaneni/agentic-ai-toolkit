import pytest
from fastapi.testclient import TestClient

from core.orchestrator import ReActOrchestrator
from core.tools import CalculatorTool, ToolRegistry
from projects.agentic_rag.api import app
from projects.agentic_rag.corpus import SEED_DOCUMENTS
from projects.agentic_rag.policy import RagAgentPolicy, _extract_arithmetic_expression, _reformulate
from projects.agentic_rag.retriever import build_index_from_documents
from projects.agentic_rag.tools import RetrieveDocsTool

@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def orchestrator():
    retriever = build_index_from_documents(list(SEED_DOCUMENTS))
    tools = ToolRegistry([RetrieveDocsTool(retriever), CalculatorTool()])
    return ReActOrchestrator(policy=RagAgentPolicy(), tools=tools, max_steps=4)


# --- API tests ---------------------------------------------------------------


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_ask_returns_answer_and_step_trace(client):
    resp = client.post("/agent/ask", json={"question": "How often should I rotate my API keys?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["resolved"] is True
    assert "90 days" in body["answer"]
    assert len(body["steps"]) == 1
    assert body["steps"][0]["action_name"] == "retrieve_docs"


def test_ask_rejects_empty_question(client):
    resp = client.post("/agent/ask", json={"question": ""})
    assert resp.status_code == 422


# --- End-to-end orchestrator behavior -----------------------------------------


def test_agent_routes_arithmetic_to_calculator(orchestrator):
    result = orchestrator.run("What is 12 * (7 + 3)?")
    assert result.final_answer == "120"
    assert len(result.steps) == 1
    assert result.steps[0].action_name == "calculator"


def test_agent_answers_confident_retrieval_in_one_step(orchestrator):
    result = orchestrator.run("Can EU customers request data erasure under GDPR?")
    assert result.resolved is True
    assert len(result.steps) == 1
    assert "erasure" in result.final_answer.lower()


def test_agent_retries_once_on_low_confidence_then_degrades_gracefully(orchestrator):
    result = orchestrator.run("asdf jkl random gibberish nonsense")
    assert result.resolved is True
    assert len(result.steps) == 2
    assert all(s.action_name == "retrieve_docs" for s in result.steps)
    assert "not fully confident" in result.final_answer.lower()


# --- Unit tests: policy helpers ------------------------------------------------


@pytest.mark.parametrize(
    "question,expected",
    [
        ("12 * (7 + 3)", "12 * (7 + 3)"),
        ("What is 12 * (7 + 3)?", "12 * (7 + 3)"),
        ("Calculate 45 / 9 + 3", "45 / 9 + 3"),
        ("How often should I rotate my API keys?", None),
    ],
)
def test_extract_arithmetic_expression(question, expected):
    assert _extract_arithmetic_expression(question) == expected


def test_reformulate_strips_stopwords():
    reformulated = _reformulate("What is the deal with my rate limit?")
    assert "what" not in reformulated.split()
    assert "rate" in reformulated
    assert "limit" in reformulated


def test_reformulate_falls_back_to_original_if_all_stopwords():
    assert _reformulate("what is the a") == "what is the a"
