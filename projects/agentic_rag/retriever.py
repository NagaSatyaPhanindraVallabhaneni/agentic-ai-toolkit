"""Hybrid (BM25 + dense vector) retrieval with Reciprocal Rank Fusion.

This is the same retrieval core used in the standalone `rag-eval-api`
project, reused here as the tool an agent calls rather than as a
single-shot query endpoint. See that project for the full writeup on the
fusion strategy and the swappable-embedder design.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    import faiss

    _HAS_FAISS = True
except ImportError:  # pragma: no cover - faiss-cpu is a required dependency
    _HAS_FAISS = False


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    text: str


def chunk_text(doc_id: str, title: str, text: str, chunk_size: int = 60, overlap: int = 15) -> list[Chunk]:
    words = text.split()
    if not words:
        return []
    chunks: list[Chunk] = []
    start = 0
    idx = 0
    step = max(chunk_size - overlap, 1)
    while start < len(words):
        window = words[start : start + chunk_size]
        chunks.append(Chunk(chunk_id=f"{doc_id}::{idx}", doc_id=doc_id, title=title, text=" ".join(window)))
        idx += 1
        if start + chunk_size >= len(words):
            break
        start += step
    return chunks


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class EmbedderProtocol(Protocol):
    def fit(self, corpus: list[str]) -> None: ...
    def embed(self, texts: list[str]) -> np.ndarray: ...


class TfidfSvdEmbedder:
    """Local, dependency-light dense embedder (TF-IDF -> truncated SVD).
    No network access or model download required. Swap in a real
    sentence-transformer or API-backed embedder by implementing
    `EmbedderProtocol` — nothing downstream changes."""

    def __init__(self, n_components: int = 128, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state
        self._vectorizer: TfidfVectorizer | None = None
        self._svd: TruncatedSVD | None = None

    def fit(self, corpus: list[str]) -> None:
        self._vectorizer = TfidfVectorizer(stop_words="english")
        tfidf = self._vectorizer.fit_transform(corpus)
        n_components = max(2, min(self.n_components, tfidf.shape[1] - 1, tfidf.shape[0] - 1))
        self._svd = TruncatedSVD(n_components=n_components, random_state=self.random_state)
        self._svd.fit(tfidf)

    def embed(self, texts: list[str]) -> np.ndarray:
        if self._vectorizer is None or self._svd is None:
            raise RuntimeError("Embedder must be fit() before embed()")
        tfidf = self._vectorizer.transform(texts)
        dense = self._svd.transform(tfidf).astype("float32")
        norms = np.linalg.norm(dense, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return dense / norms


@dataclass
class RetrievalResult:
    chunk_id: str
    doc_id: str
    title: str
    text: str
    bm25_rank: int | None
    dense_rank: int | None
    fused_score: float
    dense_score: float = 0.0
    bm25_score: float = 0.0


class HybridRetriever:
    def __init__(self, embedder: EmbedderProtocol | None = None, rrf_k: int = 60):
        self.embedder = embedder or TfidfSvdEmbedder()
        self.rrf_k = rrf_k
        self.chunks: list[Chunk] = []
        self._bm25: BM25Okapi | None = None
        self._faiss_index: "faiss.IndexFlatIP | None" = None
        self._dense_vectors: np.ndarray | None = None

    def build(self, chunks: list[Chunk]) -> None:
        if not chunks:
            raise ValueError("cannot build a retriever over zero chunks")
        self.chunks = chunks
        texts = [c.text for c in chunks]

        tokenized = [_tokenize(t) for t in texts]
        self._bm25 = BM25Okapi(tokenized)

        self.embedder.fit(texts)
        vectors = self.embedder.embed(texts)
        self._dense_vectors = vectors

        if _HAS_FAISS:
            index = faiss.IndexFlatIP(vectors.shape[1])
            index.add(vectors)
            self._faiss_index = index
        else:  # pragma: no cover - faiss-cpu is a required dependency
            self._faiss_index = None

    def _dense_ranks(self, query_vec: np.ndarray, top_n: int) -> list[tuple[int, float]]:
        if self._faiss_index is not None:
            scores, indices = self._faiss_index.search(query_vec.reshape(1, -1), top_n)
            return [(int(i), float(s)) for i, s in zip(indices[0], scores[0]) if i != -1]
        sims = (self._dense_vectors @ query_vec.reshape(-1, 1)).ravel()
        order = np.argsort(-sims)[:top_n]
        return [(int(i), float(sims[i])) for i in order]

    def retrieve(self, query: str, k: int = 5, candidate_pool: int = 20) -> list[RetrievalResult]:
        if self._bm25 is None or self.embedder is None:
            raise RuntimeError("HybridRetriever.build() must be called before retrieve()")

        pool = min(candidate_pool, len(self.chunks))

        bm25_scores = self._bm25.get_scores(_tokenize(query))
        bm25_order = np.argsort(-bm25_scores)[:pool]
        bm25_rank_of = {int(idx): rank + 1 for rank, idx in enumerate(bm25_order)}
        bm25_score_of = {int(idx): float(bm25_scores[idx]) for idx in bm25_order}

        query_vec = self.embedder.embed([query])[0]
        dense_hits = self._dense_ranks(query_vec, pool)
        dense_rank_of = {idx: rank + 1 for rank, (idx, _score) in enumerate(dense_hits)}
        dense_score_of = {idx: score for idx, score in dense_hits}

        candidate_indices = set(bm25_rank_of) | set(dense_rank_of)

        fused: list[tuple[int, float]] = []
        for idx in candidate_indices:
            score = 0.0
            if idx in bm25_rank_of:
                score += 1.0 / (self.rrf_k + bm25_rank_of[idx])
            if idx in dense_rank_of:
                score += 1.0 / (self.rrf_k + dense_rank_of[idx])
            fused.append((idx, score))

        fused.sort(key=lambda pair: pair[1], reverse=True)

        results: list[RetrievalResult] = []
        for idx, score in fused[:k]:
            c = self.chunks[idx]
            results.append(
                RetrievalResult(
                    chunk_id=c.chunk_id,
                    doc_id=c.doc_id,
                    title=c.title,
                    text=c.text,
                    bm25_rank=bm25_rank_of.get(idx),
                    dense_rank=dense_rank_of.get(idx),
                    fused_score=score,
                    dense_score=dense_score_of.get(idx, 0.0),
                    bm25_score=bm25_score_of.get(idx, 0.0),
                )
            )
        return results


def build_index_from_documents(documents: list[dict]) -> HybridRetriever:
    chunks: list[Chunk] = []
    for doc in documents:
        chunks.extend(chunk_text(doc["id"], doc["title"], doc["text"]))
    retriever = HybridRetriever()
    retriever.build(chunks)
    return retriever
