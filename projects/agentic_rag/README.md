agentic-rag
============

A tool-calling agent built on the shared `core` ReAct orchestrator. Given a
question, it decides — step by step, with a visible scratchpad — whether to
retrieve, how confident the retrieval was, and whether to retry before
answering.

What makes this "agentic" rather than "RAG"
---------------------------------------------

A plain RAG pipeline retrieves once and answers. This agent:

1. **Routes by intent first.** Arithmetic questions go straight to a
   calculator tool instead of the knowledge base.
2. **Judges its own retrieval confidence** using the raw similarity signal
   (not just rank), instead of blindly trusting whatever came back top.
3. **Retries with a reformulated query once** if confidence is low —
   genuine multi-step tool use, not a single-shot call.
4. **Degrades gracefully** with an explicit low-confidence caveat if the
   retry still doesn't find anything, rather than hallucinating a
   confident-sounding answer.

Every decision above is made by `RagAgentPolicy` (see `policy.py`), which
implements `LLMPolicyProtocol` from `core.agent`. Swap it for a real
Claude/GPT-4.1-backed policy that reasons freely instead of following fixed
rules, and nothing else — orchestrator, tools, retriever, API — has to
change.

Run it
------

```bash
uvicorn projects.agentic_rag.api:app --reload --port 8001
```

```bash
curl -s -X POST localhost:8001/agent/ask \
  -H 'Content-Type: application/json' \
  -d '{"question": "How often should I rotate my API keys?"}' | python -m json.tool
```

The response includes the full step trace (`thought`, `action_name`,
`observation` per step) so you can see exactly how the agent got to its
answer.

See the top-level repo README for the overall architecture.
