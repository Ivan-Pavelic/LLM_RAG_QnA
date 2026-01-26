from __future__ import annotations

from typing import List


def build_qa_prompt(question: str) -> str:
    # Keep prompts identical across systems except for context injection.
    return (
        "You are a careful QA system.\n"
        "Answer the question as concisely as possible.\n"
        "If the answer is not in the provided information, say: \"I don't know.\".\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


def build_qa_rag_prompt(question: str, contexts: List[str]) -> str:
    ctx = "\n\n".join([f"[{i+1}] {c}" for i, c in enumerate(contexts)])
    return (
        "You are a careful QA system.\n"
        "Use ONLY the provided context snippets.\n"
        "If the answer is not explicitly stated, say: \"I don't know.\".\n\n"
        f"Context:\n{ctx}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


def build_sum_prompt(article: str) -> str:
    return (
        "Summarize the following article in 3-5 sentences.\n"
        "Focus on the key facts and avoid adding information not present in the text.\n\n"
        f"Article:\n{article}\n\n"
        "Summary:"
    )


def build_sum_rag_prompt(contexts: List[str]) -> str:
    ctx = "\n\n".join([f"[{i+1}] {c}" for i, c in enumerate(contexts)])
    return (
        "Write a 3-5 sentence summary using ONLY the provided context snippets.\n"
        "Do not add any facts that are not present.\n\n"
        f"Context:\n{ctx}\n\n"
        "Summary:"
    )

