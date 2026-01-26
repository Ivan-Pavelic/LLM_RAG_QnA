from __future__ import annotations

from typing import Any, Dict, Iterable, List

from .utils import Chunk


def simple_word_chunks(
    doc_id: str,
    text: str,
    *,
    chunk_words: int = 200,
    overlap_words: int = 50,
    meta: Dict[str, Any] | None = None,
) -> List[Chunk]:
    """
    A simple, reproducible chunker that operates on whitespace tokenization.

    Why this chunking:
    - easy to explain and replicate
    - works reasonably well for BM25 and dense retrieval
    - avoids sentence-splitting dependencies
    """
    meta = dict(meta or {})
    words = text.split()
    if not words:
        return []

    step = max(1, chunk_words - overlap_words)
    out: List[Chunk] = []
    i = 0
    k = 0
    while i < len(words):
        window = words[i : i + chunk_words]
        if not window:
            break
        chunk_text = " ".join(window)
        chunk_id = f"{doc_id}::chunk{k}"
        out.append(
            Chunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                text=chunk_text,
                meta={**meta, "start_word": i, "end_word": min(len(words), i + chunk_words)},
            )
        )
        k += 1
        i += step
    return out


def chunk_corpus(
    docs: Iterable[Dict[str, Any]],
    *,
    text_key: str,
    id_key: str = "doc_id",
    chunk_words: int = 200,
    overlap_words: int = 50,
) -> List[Chunk]:
    chunks: List[Chunk] = []
    for d in docs:
        doc_id = str(d[id_key])
        text = str(d[text_key])
        meta = {k: v for k, v in d.items() if k not in {id_key, text_key}}
        chunks.extend(
            simple_word_chunks(
                doc_id,
                text,
                chunk_words=chunk_words,
                overlap_words=overlap_words,
                meta=meta,
            )
        )
    return chunks

