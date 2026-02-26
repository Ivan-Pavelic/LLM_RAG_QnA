# LLM vs RAG for Q&A and Summarization

This repository supports an undergraduate NLP research project:
**Comparative Analysis of LLM and Retrieval-Augmented Generation (RAG) Systems for Question Answering and Document Summarization**.

## What you will run
- **LLM baseline**: answer questions or summarize articles without retrieval (non-RAG baseline).
- **RAG-BM25**: sparse retrieval with BM25 plus the same generator.
- **RAG-Dense**: dense retrieval with SentenceTransformers plus the same generator.
- **RAG-Hybrid**: BM25 + dense retrieval combined via Reciprocal Rank Fusion (RRF).
- Optional **reranker** (e.g. MonoT5) and **larger generators** (GPT-3.5/4 via API, LLaMA 2, Mistral via Hugging Face) for extended experiments.

Datasets (reproducible with fixed seed; use larger sizes for statistical robustness):
- **Q&A**: SQuAD v1.1 subset (default 400 questions; run `python -m scripts.download_data --qa-n 400 --sum-n 100`).
- **Summarization**: CNN/DailyMail subset (default 100 articles).
- For quick runs use smaller sizes: `--qa-n 80 --sum-n 25`.

## Setup (Windows / PowerShell)
Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

## Download and sample datasets
This downloads from Hugging Face Datasets and writes small JSONL files under `data/processed/`.

```bash
python -m scripts.download_data
```

Outputs:
- `data/processed/qa_questions.jsonl`
- `data/processed/qa_corpus.jsonl`
- `data/processed/sum_articles.jsonl`

## Run experiments
Open and run the notebook:
- `notebooks/llm_vs_rag_qa_summarization.ipynb`

## Reproducibility
- Sampling uses a fixed seed.
- All core hyperparameters (chunk size, overlap, top-k, decoding settings) are defined in one config cell in the notebook.
