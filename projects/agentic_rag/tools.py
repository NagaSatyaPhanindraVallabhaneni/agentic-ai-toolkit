"""Tools available to the agentic RAG agent."""

from __future__ import annotations

from typing import Any

from core.tools import ToolResult
from projects.agentic_rag.retriever import HybridRetriever


class RetrieveDocsTool:
    name = "retrieve_docs"
    description = "Retrieve the most relevant document chunks for a query from the indexed knowledge base."

    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever

    def run(self, query: str, k: int = 3, **_: Any) -> ToolResult:
        results = self.retriever.retrieve(query, k=k)
        if not results:
            return ToolResult(output="no results found", data={"results": [], "top_score": 0.0})

        formatted = "\n".join(f"[{r.title}] {r.text}" for r in results)
        # `top_score` is the raw dense cosine similarity of the best hit
        # (not the RRF fused rank score) — with a small demo corpus, RRF's
        # rank-based fusion barely separates a real match from an
        # accidental tie, whereas raw similarity cleanly distinguishes
        # "found something relevant" (~0.5-1.0) from "found nothing"
        # (~0.0), which is what confidence-gated retrieval needs.
        top_score = max(results[0].dense_score, min(results[0].bm25_score / 5.0, 1.0))
        return ToolResult(
            output=formatted,
            data={
                "results": [
                    {
                        "doc_id": r.doc_id,
                        "title": r.title,
                        "text": r.text,
                        "fused_score": r.fused_score,
                        "dense_score": r.dense_score,
                        "bm25_score": r.bm25_score,
                    }
                    for r in results
                ],
                "top_score": top_score,
            },
        )
