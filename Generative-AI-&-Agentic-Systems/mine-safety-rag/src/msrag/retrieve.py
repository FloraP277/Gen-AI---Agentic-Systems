"""Retrievers: dense (meaning), BM25 (keywords), hybrid (both), plus reranking.

Week 6 of the course: dense retrieval matches meaning but can miss exact terms
like section numbers and jargon; BM25 matches exact terms but misses
paraphrases. Hybrid search combines both, and a cross-encoder reranker then
re-scores the top candidates by reading the query and chunk together.
"""
import re
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi

from .corpus import Corpus

# Keep section numbers like 75.400 or 56.14107(a) together as one token.
TOKEN = re.compile(r"\d+\.\d+(?:\([a-z0-9]+\))*|[a-z0-9]+")
STOP = set("a an and are as at be by for from has have in is it its of on or shall that the this to was were what when where which who will with".split())


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN.findall(text.lower()) if t not in STOP]


@dataclass
class Hit:
    chunk_id: str
    section_id: str
    text: str
    score: float


def _hits(corpus: Corpus, idx, scores) -> list[Hit]:
    rows = corpus.chunks.iloc[list(idx)]
    return [Hit(r.chunk_id, r.section_id, r.text, float(s)) for r, s in zip(rows.itertuples(), scores)]


class DenseRetriever:
    def __init__(self, corpus: Corpus, embedder):
        self.corpus, self.embedder = corpus, embedder

    def search(self, query: str, k: int) -> list[Hit]:
        if len(self.corpus) == 0:
            return []
        q = self.embedder.embed([query])[0]
        scores = self.corpus.embeddings @ q
        top = np.argsort(-scores)[:k]
        return _hits(self.corpus, top, scores[top])


class BM25Retriever:
    def __init__(self, corpus: Corpus):
        self.corpus = corpus
        self.bm25 = BM25Okapi([tokenize(t) for t in corpus.chunks["text"]]) if len(corpus) else None

    def search(self, query: str, k: int) -> list[Hit]:
        if self.bm25 is None:
            return []
        scores = self.bm25.get_scores(tokenize(query))
        top = np.argsort(-scores)[:k]
        return _hits(self.corpus, top, scores[top])


class HybridRetriever:
    """Reciprocal rank fusion: score = sum over retrievers of 1 / (k_rrf + rank).

    RRF only uses ranks, so it doesn't matter that BM25 and cosine scores are on
    completely different scales.
    """

    def __init__(self, retrievers, k_rrf: int = 60, candidates: int = 50):
        self.retrievers, self.k_rrf, self.candidates = retrievers, k_rrf, candidates

    def search(self, query: str, k: int) -> list[Hit]:
        fused, by_id = {}, {}
        for r in self.retrievers:
            for rank, h in enumerate(r.search(query, self.candidates)):
                fused[h.chunk_id] = fused.get(h.chunk_id, 0.0) + 1.0 / (self.k_rrf + rank + 1)
                by_id[h.chunk_id] = h
        ranked = sorted(fused, key=lambda c: -fused[c])[:k]
        return [Hit(c, by_id[c].section_id, by_id[c].text, fused[c]) for c in ranked]


class CrossEncoderReranker:
    """Open-source BGE reranker. Reads query and chunk together, so it is slower
    but more accurate than comparing two separate embeddings."""

    def __init__(self, model: str = "BAAI/bge-reranker-base"):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model)

    def score(self, query: str, texts: list[str]) -> list[float]:
        return [float(s) for s in self.model.predict([(query, t) for t in texts])]


class Reranked:
    """Retrieve `candidates` chunks with a base retriever, keep the best k after reranking."""

    def __init__(self, base, reranker, candidates: int = 20):
        self.base, self.reranker, self.candidates = base, reranker, candidates

    def search(self, query: str, k: int) -> list[Hit]:
        hits = self.base.search(query, self.candidates)
        if not hits:
            return []
        scores = self.reranker.score(query, [h.text for h in hits])
        order = sorted(range(len(hits)), key=lambda i: -scores[i])[:k]
        return [Hit(hits[i].chunk_id, hits[i].section_id, hits[i].text, scores[i]) for i in order]


def build(kind: str, corpus: Corpus, embedder, reranker=None):
    """kind: dense | bm25 | hybrid | hybrid_rerank"""
    if kind == "dense":
        return DenseRetriever(corpus, embedder)
    if kind == "bm25":
        return BM25Retriever(corpus)
    hybrid = HybridRetriever([DenseRetriever(corpus, embedder), BM25Retriever(corpus)])
    if kind == "hybrid":
        return hybrid
    if kind == "hybrid_rerank":
        if reranker is None:
            raise ValueError("hybrid_rerank needs a reranker")
        return Reranked(hybrid, reranker)
    raise ValueError(f"unknown retriever {kind!r}")
