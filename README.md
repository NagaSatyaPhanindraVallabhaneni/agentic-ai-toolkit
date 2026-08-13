<div align="center">

# agentic-ai-toolkit

### A small, dependency-light framework for building tool-using, multi-agent AI systems — plus two working reference implementations.

[![CI](https://github.com/NagaSatyaPhanindraVallabhaneni/agentic-ai-toolkit/actions/workflows/ci.yml/badge.svg)](https://github.com/NagaSatyaPhanindraVallabhaneni/agentic-ai-toolkit/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)

</div>

---

Why this exists
----------------

Agentic AI — systems that reason step by step, call tools, judge their own
output, and loop until a goal is met, instead of returning a single-shot
model response — is the fastest-growing category in AI engineering right
now, alongside retrieval-augmented generation. This repo is a from-scratch,
dependency-light implementation of the core patterns underneath both:

  * **ReAct-style tool calling** — Thought → Action → Observation, repeated
    until the agent has enough information to answer.
  * **Confidence-gated retrieval** — an agent that inspects *how confident*
    a retrieval was before trusting it, and retries with a reformulated
    query rather than blindly answering off the top hit.
  * **Multi-agent orchestration with reflection** — a Planner, a Writer,
    and a Critic that hand work off to each other and iterate until an
    explicit rubric passes, the same shape as a self-correcting agent
    swarm.

None of it depends on an LLM API key to run. Every agent's "brain" is a
`Protocol` — a deterministic, fully-tested implementation ships by default
so the whole repo runs and tests offline, and a production deployment
swaps in a real Claude / GPT-4.1-backed policy by implementing that same
protocol. The orchestration, tool-calling, and API layers never change.

Architecture
------------

```
core/  (shared framework)
├── tools.py          Tool protocol, ToolRegistry, CalculatorTool
├── agent.py          AgentStep, Decision, LLMPolicyProtocol
└── orchestrator.py   ReActOrchestrator: Thought -> Action -> Observation, looped

projects/agentic_rag/          built on core/
├── retriever.py       HybridRetriever: BM25 + FAISS dense vectors, fused with RRF
├── tools.py            RetrieveDocsTool (wraps the retriever as a tool)
├── policy.py            RagAgentPolicy: route -> retrieve -> check confidence
│                          -> retry once if low -> answer (or degrade gracefully)
└── api.py                 FastAPI: POST /agent/ask -> answer + full step trace

projects/report_crew/          built on core/'s design principles
├── agents.py           PlannerAgent, WriterAgent, CriticAgent (each a Protocol
│                        + a deterministic default implementation)
├── crew.py              ReportCrew: plan -> draft -> critique -> revise,
│                          looped until the critic approves or the budget runs out
└── api.py                 FastAPI: POST /crew/report -> report + per-iteration trace
```

Projects
--------

| Project | Demonstrates | Try it |
| --- | --- | --- |
| [`projects/agentic_rag`](projects/agentic_rag/) | Tool-calling agent, confidence-gated hybrid (BM25 + dense vector) retrieval, query reformulation and retry | `uvicorn projects.agentic_rag.api:app --port 8001` |
| [`projects/report_crew`](projects/report_crew/) | Multi-agent orchestration: Planner → Writer → Critic with an iterative revise loop | `uvicorn projects.report_crew.api:app --port 8002` |

Each project has its own README with the design rationale and example
requests.

Quick start
-----------

```bash
pip install -r requirements.txt
pytest -q          # 33 tests, all offline, no API keys required
```

```bash
# agentic-rag: ask a question, get back the answer AND the agent's full
# reasoning trace (what it tried, how confident it was, whether it retried)
uvicorn projects.agentic_rag.api:app --reload --port 8001
curl -s -X POST localhost:8001/agent/ask \
  -H 'Content-Type: application/json' \
  -d '{"question": "How often should I rotate my API keys?"}' | python -m json.tool
```

```bash
# report-crew: give it a topic, watch three agents draft and revise a
# report until an explicit rubric passes
uvicorn projects.report_crew.api:app --reload --port 8002
curl -s -X POST localhost:8002/crew/report \
  -H 'Content-Type: application/json' \
  -d '{"topic": "zero-trust API security"}' | python -m json.tool
```

Testing
-------

```bash
pytest -q
```

33 tests across three modules:

  * `tests/test_core.py` — the orchestrator and tool-calling machinery in
    isolation (a stub policy, not any project's real decision logic):
    tool execution, unknown-tool handling, max-step budgets, the safe
    arithmetic evaluator's sandboxing.
  * `tests/test_agentic_rag.py` — end-to-end agent runs (arithmetic
    routing, confident one-shot retrieval, the low-confidence retry-then-
    degrade path) plus unit tests for the policy's helper functions.
  * `tests/test_report_crew.py` — each agent in isolation (planner,
    writer's thin-then-expanded drafting, critic's rubric), the crew's
    convergence behavior, and the max-iteration budget.

Tech stack
----------

`FastAPI` · `Pydantic` · `rank_bm25` (lexical retrieval) · `scikit-learn`
(TF-IDF + SVD dense embeddings) · `FAISS` (vector index) · `NumPy` ·
`pytest` / `httpx`

Extending this toward a real deployment
-----------------------------------------

  * Implement `LLMPolicyProtocol` (agentic-rag) or `PlannerProtocol` /
    `WriterProtocol` / `CriticProtocol` (report-crew) with a real model
    call — Claude, GPT-4.1, or anything else with a chat completion API.
    Nothing in `core/`, the tools, or the FastAPI layer needs to change.
  * Swap `TfidfSvdEmbedder` for a real sentence-transformer model or an
    API-backed embedder (`EmbedderProtocol` in
    `projects/agentic_rag/retriever.py`) for stronger semantic recall.
  * Add more tools to the agentic-rag `ToolRegistry` — a real web search,
    a code executor, a second knowledge base — the ReAct loop already
    supports arbitrary multi-tool plans.
  * Persist the FAISS index and BM25 corpus to disk instead of rebuilding
    in memory on startup.
  * Add a fourth agent to report-crew (e.g. a fact-checker) between Writer
    and Critic.

License
-------

MIT — see [LICENSE](LICENSE).
