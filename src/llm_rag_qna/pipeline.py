from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from .generator import GenerationConfig, HFText2TextGenerator
from .prompts import (
    build_qa_prompt,
    build_qa_rag_prompt,
    build_sum_prompt,
    build_sum_rag_prompt,
)
from .retrievers import BM25Retriever, DenseRetriever, RetrievedChunk
from .utils import Chunk


@dataclass
class RAGConfig:
    top_k: int = 5


def _contexts_from_retrieval(results: List[RetrievedChunk]) -> List[str]:
    return [r.chunk.text for r in results]


def answer_qa_llm(
    question: str,
    *,
    generator: HFText2TextGenerator,
    gen_cfg: GenerationConfig,
) -> str:
    prompt = build_qa_prompt(question)
    return generator.generate(prompt, cfg=gen_cfg)


def answer_qa_rag(
    question: str,
    *,
    retriever: BM25Retriever | DenseRetriever,
    rag_cfg: RAGConfig,
    generator: HFText2TextGenerator,
    gen_cfg: GenerationConfig,
) -> Dict[str, object]:
    retrieved = retriever.retrieve(question, top_k=rag_cfg.top_k)
    prompt = build_qa_rag_prompt(question, _contexts_from_retrieval(retrieved))
    answer = generator.generate(prompt, cfg=gen_cfg)
    return {"answer": answer, "retrieved": retrieved, "prompt": prompt}


def summarize_llm(
    article: str,
    *,
    generator: HFText2TextGenerator,
    gen_cfg: GenerationConfig,
) -> str:
    prompt = build_sum_prompt(article)
    return generator.generate(prompt, cfg=gen_cfg)


def summarize_rag_over_article(
    article_chunks: List[Chunk],
    *,
    query: str = "main points of the article",
    retriever_type: str = "bm25",
    rag_cfg: RAGConfig,
    generator: HFText2TextGenerator,
    gen_cfg: GenerationConfig,
    dense_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> Dict[str, object]:
    """
    RAG for summarization in a controlled setting:
    retrieval is performed *within the same article* (chunked),
    then the generator summarizes only the retrieved chunks.
    """
    if retriever_type == "bm25":
        retriever = BM25Retriever(article_chunks)
    elif retriever_type == "dense":
        retriever = DenseRetriever(article_chunks, model_name=dense_model_name)
    else:
        raise ValueError("retriever_type must be 'bm25' or 'dense'")

    retrieved = retriever.retrieve(query, top_k=rag_cfg.top_k)
    prompt = build_sum_rag_prompt(_contexts_from_retrieval(retrieved))
    summary = generator.generate(prompt, cfg=gen_cfg)
    return {"summary": summary, "retrieved": retrieved, "prompt": prompt}

