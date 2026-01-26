from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

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
        self.model = SentenceTransformer(model_name)
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

