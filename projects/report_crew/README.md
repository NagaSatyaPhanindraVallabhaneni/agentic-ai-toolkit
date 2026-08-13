report-crew
============

A three-agent Planner → Writer → Critic loop, orchestrated by `ReportCrew`
(`crew.py`), that drafts a structured report and iteratively revises it
until an explicit rubric passes — or a max-iteration budget runs out.

The pattern
-----------

* **PlannerAgent** turns a topic into a section outline.
* **WriterAgent** drafts each section — deliberately thin on the first
  pass.
* **CriticAgent** checks the draft against an explicit, section-addressable
  rubric (minimum sentence count per section, no missing/empty sections)
  and returns concrete issues, not a vague thumbs-down.
* **ReportCrew** feeds the critic's issues back to the writer, which
  expands exactly the flagged sections, and loops until the critic
  approves or the iteration budget is spent.

On a typical topic this converges in exactly two iterations: the first
draft is flagged as too short, the second (targeted) revision passes. The
full per-iteration trace — what was flagged, what changed — is returned by
both the Python API and the HTTP endpoint, so the loop is inspectable, not
a black box.

Every agent implements a protocol (`PlannerProtocol`, `WriterProtocol`,
`CriticProtocol` in `agents.py`). Swap any one of them for a real
LLM-backed implementation — an LLM proposing the outline, drafting prose,
or critiquing against a richer rubric — without changing `ReportCrew`'s
orchestration logic at all.

Run it
------

```bash
uvicorn projects.report_crew.api:app --reload --port 8002
```

```bash
curl -s -X POST localhost:8002/crew/report \
  -H 'Content-Type: application/json' \
  -d '{"topic": "zero-trust API security"}' | python -m json.tool
```

See the top-level repo README for the overall architecture.
