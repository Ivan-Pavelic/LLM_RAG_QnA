from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol, Sequence, Tuple

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from .utils import Chunk


def _tokenize_for_bm25(text: str) -> List[str]:
    # Minimal tokenizer: lower + split. (Explainable, reproducible.)
    return text.lower().split()


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float


def _reciprocal_rank_fusion(
    ranked_lists: List[List[Tuple[Chunk, float]]],
    *,
    k: int = 60,
) -> List[RetrievedChunk]:
    """Merge multiple ranked lists using Reciprocal Rank Fusion. Each list is (chunk, rank_score)."""
    rrf_scores: dict = {}
    chunk_by_id = {}
    for rl in ranked_lists:
        for rank, (chunk, _) in enumerate(rl):
            cid = chunk.chunk_id
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
            chunk_by_id[cid] = chunk
    sorted_ids = sorted(rrf_scores.keys(), key=lambda x: -rrf_scores[x])
    return [
        RetrievedChunk(chunk=chunk_by_id[cid], score=rrf_scores[cid])
        for cid in sorted_ids
    ]


class BM25Retriever:
    def __init__(self, chunks: Sequence[Chunk]):
        self.chunks = list(chunks)
        self.corpus_tokens = [_tokenize_for_bm25(c.text) for c in self.chunks]
        self.bm25 = BM25Okapi(self.corpus_tokens)

    def retrieve(self, query: str, *, top_k: int = 5) -> List[RetrievedChunk]:
        q_tokens = _tokenize_for_bm25(query)
        scores = self.bm25.get_scores(q_tokens)
        idx = np.argsort(scores)[::-1][:top_k]
        return [RetrievedChunk(chunk=self.chunks[i], score=float(scores[i])) for i in idx]


class DenseRetriever:
    # Simple in-process cache so we don't reload the same SentenceTransformer
    # weights on every instantiation (important for per-article summarization).
    _MODEL_CACHE: dict[str, SentenceTransformer] = {}

    def __init__(
        self,
        chunks: Sequence[Chunk],
        *,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 64,
        normalize: bool = True,
        show_progress_bar: bool = False,
    ):
        self.chunks = list(chunks)
        if model_name in self._MODEL_CACHE:
            self.model = self._MODEL_CACHE[model_name]
        else:
            self.model = SentenceTransformer(model_name)
            self._MODEL_CACHE[model_name] = self.model
        self.normalize = normalize
        self.embeddings = self.model.encode(
            [c.text for c in self.chunks],
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
        )

    def retrieve(self, query: str, *, top_k: int = 5) -> List[RetrievedChunk]:
        q = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=self.normalize)
        sims = cosine_similarity(q, self.embeddings)[0]
        idx = np.argsort(sims)[::-1][:top_k]
        return [RetrievedChunk(chunk=self.chunks[i], score=float(sims[i])) for i in idx]


class HybridRetriever:
    """
    Combines BM25 and Dense retrieval via Reciprocal Rank Fusion (RRF).
    Retrieves candidate_size from each retriever, merges with RRF, returns top_k.
    """

    def __init__(
        self,
        bm25_retriever: BM25Retriever,
        dense_retriever: DenseRetriever,
        *,
        candidate_size: int = 100,
        rrf_k: int = 60,
    ):
        self.bm25 = bm25_retriever
        self.dense = dense_retriever
        self.candidate_size = candidate_size
        self.rrf_k = rrf_k

    def retrieve(self, query: str, *, top_k: int = 5) -> List[RetrievedChunk]:
        bm25_results = self.bm25.retrieve(query, top_k=self.candidate_size)
        dense_results = self.dense.retrieve(query, top_k=self.candidate_size)
        bm25_list = [(r.chunk, r.score) for r in bm25_results]
        dense_list = [(r.chunk, r.score) for r in dense_results]
        fused = _reciprocal_rank_fusion([bm25_list, dense_list], k=self.rrf_k)
        return fused[:top_k]


class RerankerProtocol(Protocol):
    """Protocol for a reranker that reorders retrieved chunks by relevance to the query."""

    def rerank(self, query: str, chunks: List[RetrievedChunk], *, top_k: int) -> List[RetrievedChunk]:
        ...


class MonoT5Reranker:
    """
    Rerank passages using a T5 model that predicts true/false for query-document relevance.
    Uses castorini/monot5-base-msmarco or similar. Optional: requires transformers.
    """

    def __init__(self, model_name: str = "castorini/monot5-base-msmarco", device: str | None = None):
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as e:
            raise ImportError("MonoT5Reranker requires transformers and torch") from e
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model.to(self.device)
        self.model.eval()

    def rerank(self, query: str, chunks: List[RetrievedChunk], *, top_k: int) -> List[RetrievedChunk]:
        if not chunks:
            return []
        import torch
        inputs = [
            "Query: " + query + " Document: " + (r.chunk.text[:512])
            for r in chunks
        ]
        tokenized = self.tokenizer(
            inputs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            out = self.model.generate(**tokenized, max_new_tokens=2)
        scores = []
        for i, o in enumerate(out):
            decoded = self.tokenizer.decode(o, skip_special_tokens=True).strip().lower()
            scores.append(1.0 if decoded == "true" else 0.0)
        indexed = sorted(range(len(chunks)), key=lambda i: -scores[i])
        return [RetrievedChunk(chunk=chunks[j].chunk, score=scores[j]) for j in indexed[:top_k]]


