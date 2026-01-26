from __future__ import annotations

"""
Downloads and samples datasets for the coursework-scale experiments.

Q&A: SQuAD v1.1
- Sample N question-answer pairs from the training split.
- Build a retrieval corpus from the unique contexts in the sampled set.

Summarization: CNN/DailyMail (version 3.0.0)
- Sample M articles from the test split (short/medium length by default).

Outputs (JSONL):
- data/processed/qa_questions.jsonl
- data/processed/qa_corpus.jsonl
- data/processed/sum_articles.jsonl
"""

import random
from pathlib import Path

from datasets import load_dataset

try:
    from llm_rag_qna.utils import ensure_dir, set_seed, write_jsonl
except ModuleNotFoundError:  # pragma: no cover
    # Beginner-proofing: allow running without `pip install -e .`
    import sys
    from pathlib import Path as _Path

    sys.path.insert(0, str((_Path(__file__).resolve().parents[1] / "src")))
    from llm_rag_qna.utils import ensure_dir, set_seed, write_jsonl


def main(
    *,
    seed: int = 13,
    qa_n: int = 80,
    sum_n: int = 25,
) -> None:
    set_seed(seed)
    random.seed(seed)

    out_dir = ensure_dir(Path("data") / "processed")

    # -------------------------
    # Q&A (SQuAD v1.1)
    # -------------------------
    squad = load_dataset("squad")
    qa_split = squad["train"]

    idx = list(range(len(qa_split)))
    random.shuffle(idx)
    idx = idx[:qa_n]

    qa_rows = []
    contexts = {}
    for i in idx:
        ex = qa_split[int(i)]
        qid = ex["id"]
        question = ex["question"]
        context = ex["context"]
        answer = ex["answers"]["text"][0] if ex["answers"]["text"] else ""
        title = ex.get("title", "")

        qa_rows.append(
            {
                "id": qid,
                "question": question,
                "answer": answer,
                "doc_id": qid,  # link to its "source" context (for analysis)
                "title": title,
            }
        )
        # Corpus: store unique contexts; doc_id uses the qid (simple, reproducible).
        contexts[qid] = {"doc_id": qid, "text": context, "title": title}

    write_jsonl(out_dir / "qa_questions.jsonl", qa_rows)
    write_jsonl(out_dir / "qa_corpus.jsonl", contexts.values())

    # -------------------------
    # Summarization (CNN/DailyMail)
    # -------------------------
    cnndm = load_dataset("cnn_dailymail", "3.0.0")
    sum_split = cnndm["test"]

    # Prefer short/medium articles to keep runtime reasonable.
    candidates = []
    for i in range(len(sum_split)):
        art = sum_split[i]["article"]
        if 300 <= len(art) <= 3000:
            candidates.append(i)

    random.shuffle(candidates)
    candidates = candidates[:sum_n]

    sum_rows = []
    for i in candidates:
        ex = sum_split[int(i)]
        sum_rows.append(
            {
                "id": str(i),
                "article": ex["article"],
                "highlights": ex["highlights"],
                "source": "cnn_dailymail/3.0.0:test",
            }
        )

    write_jsonl(out_dir / "sum_articles.jsonl", sum_rows)

    print(f"Wrote {len(qa_rows)} QA questions and {len(contexts)} QA docs.")
    print(f"Wrote {len(sum_rows)} summarization articles.")
    print(f"Output directory: {out_dir.resolve()}")


if __name__ == "__main__":
    main()

