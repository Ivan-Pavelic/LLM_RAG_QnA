# LLM vs RAG for Q&A and Summarization

This repository supports an undergraduate NLP research project:
**Comparative Analysis of LLM and Retrieval-Augmented Generation (RAG) Systems for Question Answering and Document Summarization**.

## What you will run
- **LLM baseline**: answer questions / summarize articles without retrieval.
- **RAG-BM25**: sparse retrieval with BM25 + the same generator.
- **RAG-Dense**: dense retrieval with SentenceTransformers + the same generator.

Datasets (sampled for coursework scale, fully reproducible):
- **Q&A**: SQuAD v1.1 subset (80 questions by default) with a corpus built from all sampled contexts.
- **Summarization**: CNN/DailyMail subset (25 articles by default).

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
