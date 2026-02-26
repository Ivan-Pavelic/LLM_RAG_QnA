from __future__ import annotations

"""
Reproducible end-to-end runner (no notebook required).

Assumes datasets were prepared via:
  python -m scripts.download_data

Outputs:
- outputs/predictions_qa.csv
- outputs/predictions_sum.csv
- outputs/predictions_all_scored.csv
- outputs/metrics_qa.csv
- outputs/metrics_sum.csv
- outputs/figures/*.png
- outputs/manual_annotations_template.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import ParserError
from tqdm.auto import tqdm

try:
    from llm_rag_qna.chunking import chunk_corpus, simple_word_chunks
    from llm_rag_qna.evaluation import (
        compute_bertscore,
        compute_bleu,
        compute_rouge,
        correlation_manual_vs_metric,
        manual_label_to_numeric,
        qa_accuracy,
    )
    from llm_rag_qna.generator import GenerationConfig, HFText2TextGenerator, OpenAIGenerator, HFCausalLMGenerator
    from llm_rag_qna.pipeline import (
        RAGConfig,
        answer_qa_llm,
        answer_qa_rag,
        summarize_llm,
        summarize_rag_over_article,
    )
    from llm_rag_qna.utils import ensure_dir, normalize_text, read_jsonl, set_seed
    from llm_rag_qna import viz
except ModuleNotFoundError:  # pragma: no cover
    # Beginner-proofing: allow running without `pip install -e .`
    import sys
    from pathlib import Path as _Path

    sys.path.insert(0, str((_Path(__file__).resolve().parents[1] / "src")))
    from llm_rag_qna.chunking import chunk_corpus, simple_word_chunks
    from llm_rag_qna.evaluation import (
        compute_bertscore,
        compute_bleu,
        compute_rouge,
        correlation_manual_vs_metric,
        manual_label_to_numeric,
        qa_accuracy,
    )
    from llm_rag_qna.generator import GenerationConfig, HFText2TextGenerator, OpenAIGenerator, HFCausalLMGenerator
    from llm_rag_qna.pipeline import (
        RAGConfig,
        answer_qa_llm,
        answer_qa_rag,
        summarize_llm,
        summarize_rag_over_article,
    )
    from llm_rag_qna.utils import ensure_dir, normalize_text, read_jsonl, set_seed
    from llm_rag_qna import viz

from rouge_score import rouge_scorer


def rougeL_f1(pred: str, ref: str) -> float:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return float(scorer.score(ref, pred)["rougeL"].fmeasure)


def read_manual_annotations_csv(path: Path) -> pd.DataFrame:
    """
    Robust reader for filled annotation files.

    Why this exists:
    - Excel/Sheets often saves CSV with `;` delimiter in some locales
    - fields contain commas and embedded newlines (predictions/references)
    - numeric columns may be reformatted (we only trust manual_label + keys)
    """
    # 1) Try the standard comma-separated format.
    try:
        df = pd.read_csv(path)
    except ParserError:
        df = None

    # 2) If parsing failed (or produced a single-column DF), try delimiter sniffing.
    if df is None or df.shape[1] == 1:
        df = pd.read_csv(path, sep=None, engine="python")  # sniff delimiter (often ';')

    # Normalize column names (strip whitespace + UTF-8 BOM if Excel added it)
    def _clean_col(c: object) -> str:
        s = str(c).strip()
        return s.lstrip("\ufeff")

    df.columns = [_clean_col(c) for c in df.columns]
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--only-manual-eval",
        action="store_true",
        help="Skip generation; only create manual-eval figures using existing outputs/predictions_all_scored.csv",
    )
    ap.add_argument(
        "--only-summarization",
        action="store_true",
        help="Skip QA; load outputs/predictions_qa.csv and metrics_qa.csv, run only summarization, then merge and write figures.",
    )
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--chunk-words", type=int, default=200)
    ap.add_argument("--overlap-words", type=int, default=50)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--dense-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--gen-model", type=str, default="google/flan-t5-small")
    ap.add_argument("--use-openai", action="store_true", help="Use OpenAI API (GPT) instead of local Hugging Face model. Set OPENAI_API_KEY in the environment.")
    ap.add_argument("--openai-model", type=str, default="gpt-4o-mini", help="OpenAI model when --use-openai (e.g. gpt-4o, gpt-4o-mini, gpt-3.5-turbo).")
    ap.add_argument("--use-mistral", action="store_true", help="Use Mistral 7B Instruct (Hugging Face, local; ~14GB download, slow on CPU).")
    ap.add_argument("--use-llama", action="store_true", help="Use LLaMA 2 7B Chat (Hugging Face, local; may require HF login & Meta approval).")
    ap.add_argument("--use-tinyllama", action="store_true", help="Use TinyLlama 1.1B Chat (fast, ~600MB; good for testing without GPU).")
    ap.add_argument("--device", type=str, default=None, help="Device for Hugging Face models: cuda, cuda:0, cpu, etc. If not set, uses CUDA when available.")
    ap.add_argument("--causal-lm-model", type=str, default=None, help="Custom Hugging Face causal LM (e.g. mistralai/Mistral-7B-Instruct-v0.2). Overrides --use-mistral/--use-llama/--use-tinyllama if set.")
    ap.add_argument("--hybrid", action="store_true", help="Also run RAG with hybrid retrieval (BM25 + dense, RRF).")
    ap.add_argument("--reranker", action="store_true", help="Also run RAG-Dense with MonoT5 reranker (retrieve 20, rerank to top-k).")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--bertscore-model", type=str, default="distilroberta-base")
    ap.add_argument("--skip-bertscore", action="store_true")
    ap.add_argument("--manual-qa-n", type=int, default=30)
    ap.add_argument("--manual-sum-n", type=int, default=10)
    ap.add_argument(
        "--sum-limit",
        type=int,
        default=None,
        help="Optional: limit number of summarization articles processed (useful for slow TinyLlama runs).",
    )
    args = ap.parse_args()

    set_seed(args.seed)

    data_dir = Path("data") / "processed"
    out_dir = ensure_dir("outputs")
    fig_dir = ensure_dir(out_dir / "figures")

    # Fast path: manual evaluation only (no re-generation)
    if args.only_manual_eval:
        scored_path = out_dir / "predictions_all_scored.csv"
        if not scored_path.exists():
            raise FileNotFoundError(f"Missing {scored_path}. Run the full pipeline once before --only-manual-eval.")
        all_df = pd.read_csv(scored_path)

        filled_path = out_dir / "manual_annotations_filled.csv"
        if not filled_path.exists():
            raise FileNotFoundError(f"Missing {filled_path}. Fill the template first.")

        annotations = read_manual_annotations_csv(filled_path)
        keep_cols = ["task", "id", "system", "manual_label"]
        missing = [c for c in keep_cols if c not in annotations.columns]
        if missing:
            raise ValueError(f"manual_annotations_filled.csv is missing columns: {missing}")

        annotations = annotations[keep_cols].copy()
        annotations["manual_label"] = annotations["manual_label"].astype(str).str.strip()

        filled = all_df.merge(annotations, on=["task", "id", "system"], how="inner")
        filled["manual_numeric"] = manual_label_to_numeric(filled["manual_label"].tolist())

        fig = viz.scatter_manual_vs_metric(
            filled.dropna(subset=["manual_numeric", "rougeL_f1"]),
            manual_numeric_col="manual_numeric",
            metric_col="rougeL_f1",
            title="Manual vs ROUGE-L F1 (all tasks)",
        )
        fig.savefig(fig_dir / "scatter_manual_vs_rougeL.png", dpi=200)

        rho = correlation_manual_vs_metric(filled, manual_col="manual_numeric", metric_col="rougeL_f1")
        print("Spearman correlation (manual vs ROUGE-L F1):", rho)
        (out_dir / "spearman_rho.txt").write_text(str(rho), encoding="utf-8")

        fig = viz.stacked_hallucination_breakdown(
            filled,
            title="Manual label breakdown (proxy for hallucination/correctness)",
        )
        fig.savefig(fig_dir / "stacked_manual_breakdown.png", dpi=200)

        print("Done.")
        print("Outputs:", out_dir.resolve())
        return

    if args.only_summarization:
        qa_path = out_dir / "predictions_qa.csv"
        qa_metrics_path = out_dir / "metrics_qa.csv"
        if not qa_path.exists() or not qa_metrics_path.exists():
            raise FileNotFoundError(
                "For --only-summarization, outputs/predictions_qa.csv and metrics_qa.csv must exist. Run full pipeline once (e.g. with TinyLlama) to generate them."
            )
        qa_df = pd.read_csv(qa_path)
        qa_metrics_df = pd.read_csv(qa_metrics_path)
        sum_articles = read_jsonl(data_dir / "sum_articles.jsonl")
        rag_cfg = RAGConfig(top_k=args.top_k)
        rag_cfg_rerank = None
    else:
        qa_questions = read_jsonl(data_dir / "qa_questions.jsonl")
        qa_corpus = read_jsonl(data_dir / "qa_corpus.jsonl")
        sum_articles = read_jsonl(data_dir / "sum_articles.jsonl")

    # Build QA retrieval index (skip when --only-summarization)
    if not args.only_summarization:
        qa_chunks = chunk_corpus(
        qa_corpus,
        text_key="text",
        id_key="doc_id",
        chunk_words=args.chunk_words,
        overlap_words=args.overlap_words,
    )
    if not args.only_summarization:
        from llm_rag_qna.retrievers import BM25Retriever, DenseRetriever, HybridRetriever

        bm25_retriever = BM25Retriever(qa_chunks)
        dense_retriever = DenseRetriever(qa_chunks, model_name=args.dense_model)
        if getattr(args, "hybrid", False):
            hybrid_retriever = HybridRetriever(bm25_retriever, dense_retriever, candidate_size=min(50, len(qa_chunks)), rrf_k=60)
        else:
            hybrid_retriever = None

        rag_cfg = RAGConfig(top_k=args.top_k)
        if getattr(args, "reranker", False):
            try:
                from llm_rag_qna.retrievers import MonoT5Reranker
                _reranker = MonoT5Reranker()
                rag_cfg_rerank = RAGConfig(top_k=args.top_k, retrieve_candidates=20, reranker=_reranker)
            except Exception as e:
                print("Warning: MonoT5 reranker not available:", e, "- skipping --reranker")
                rag_cfg_rerank = None
        else:
            rag_cfg_rerank = None

    gen_cfg = GenerationConfig(max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_p=args.top_p)
    device_kw = {"device": args.device} if getattr(args, "device", None) is not None else {}
    use_causal = any([
        getattr(args, "causal_lm_model", None),
        getattr(args, "use_llama", False),
        getattr(args, "use_mistral", False),
        getattr(args, "use_tinyllama", False),
    ])
    if use_causal:
        print("Preparing causal LM generator (downloading/loading model on first run; can take 1–5 min) ...", flush=True)
    if getattr(args, "use_openai", False):
        import os
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            # Try loading from .env in project root
            root = Path(__file__).resolve().parents[1]
            env_path = root / ".env"
            if env_path.exists():
                for line in env_path.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        if k.strip() == "OPENAI_API_KEY":
                            api_key = v.strip().strip('"').strip("'")
                            os.environ["OPENAI_API_KEY"] = api_key
                            break
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise SystemExit(
                "OPENAI_API_KEY is not set. In PowerShell run: $env:OPENAI_API_KEY=\"sk-your-key-here\"\n"
                "Or create a .env file in the project root with: OPENAI_API_KEY=sk-your-key-here"
            )
        generator = OpenAIGenerator(model_name=args.openai_model)
        print("Using OpenAI model:", args.openai_model)
    elif getattr(args, "causal_lm_model", None):
        generator = HFCausalLMGenerator(model_name=args.causal_lm_model, **device_kw)
        print("Using causal LM:", args.causal_lm_model)
    elif getattr(args, "use_llama", False):
        generator = HFCausalLMGenerator(model_name="meta-llama/Llama-2-7b-chat-hf", **device_kw)
        print("Using LLaMA 2 7B Chat (meta-llama/Llama-2-7b-chat-hf)")
    elif getattr(args, "use_mistral", False):
        generator = HFCausalLMGenerator(model_name="mistralai/Mistral-7B-Instruct-v0.2", **device_kw)
        print("Using Mistral 7B Instruct (mistralai/Mistral-7B-Instruct-v0.2)")
    elif getattr(args, "use_tinyllama", False):
        generator = HFCausalLMGenerator(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0", **device_kw)
        print("Using TinyLlama 1.1B Chat (fast, small model)")
    else:
        generator = HFText2TextGenerator(model_name=args.gen_model, **device_kw)

    # -----------------
    # QA inference (skip when --only-summarization)
    # -----------------
    if not args.only_summarization:
        qa_rows = []
        for ex in tqdm(qa_questions, desc="QA"):
            qid = ex["id"]
            question = ex["question"]
            ref = ex["answer"]

            pred_llm = answer_qa_llm(question, generator=generator, gen_cfg=gen_cfg)
            out_bm25 = answer_qa_rag(
                question,
                retriever=bm25_retriever,
                rag_cfg=rag_cfg,
                generator=generator,
                gen_cfg=gen_cfg,
            )
            out_dense = answer_qa_rag(
                question,
                retriever=dense_retriever,
                rag_cfg=rag_cfg,
                generator=generator,
                gen_cfg=gen_cfg,
            )
            qa_rows.append({"task": "qa", "id": qid, "system": "llm", "prediction": pred_llm, "reference": ref})
            qa_rows.append({"task": "qa", "id": qid, "system": "rag_bm25", "prediction": out_bm25["answer"], "reference": ref})
            qa_rows.append({"task": "qa", "id": qid, "system": "rag_dense", "prediction": out_dense["answer"], "reference": ref})
            if hybrid_retriever is not None:
                out_hybrid = answer_qa_rag(
                    question,
                    retriever=hybrid_retriever,
                    rag_cfg=rag_cfg,
                    generator=generator,
                    gen_cfg=gen_cfg,
                )
                qa_rows.append({"task": "qa", "id": qid, "system": "rag_hybrid", "prediction": out_hybrid["answer"], "reference": ref})
            if rag_cfg_rerank is not None:
                out_rerank = answer_qa_rag(
                    question,
                    retriever=dense_retriever,
                    rag_cfg=rag_cfg_rerank,
                    generator=generator,
                    gen_cfg=gen_cfg,
                )
                qa_rows.append({"task": "qa", "id": qid, "system": "rag_dense_rerank", "prediction": out_rerank["answer"], "reference": ref})

        qa_df = pd.DataFrame(qa_rows)
        qa_df.to_csv(out_dir / "predictions_qa.csv", index=False)

        qa_metrics = []
        for system, sdf in qa_df.groupby("system"):
            preds = sdf["prediction"].tolist()
            refs = sdf["reference"].tolist()
            m = {
                "system": system,
                "qa_accuracy": qa_accuracy(preds, refs),
                "bleu": compute_bleu(preds, refs),
                **compute_rouge(preds, refs),
            }
            if not args.skip_bertscore:
                m.update(compute_bertscore(preds, refs, model_type=args.bertscore_model))
            qa_metrics.append(m)
        qa_metrics_df = pd.DataFrame(qa_metrics)
        qa_metrics_df.to_csv(out_dir / "metrics_qa.csv", index=False)

    # -----------------
    # Summarization inference
    # -----------------
    sum_rows = []
    for idx, ex in enumerate(tqdm(sum_articles, desc="Summarization")):
        if args.sum_limit is not None and idx >= args.sum_limit:
            break
        sid = ex["id"]
        article = ex["article"]
        ref = ex["highlights"]

        pred_llm = summarize_llm(article, generator=generator, gen_cfg=gen_cfg)

        art_chunks = simple_word_chunks(
            doc_id=f"sum_{sid}",
            text=article,
            chunk_words=args.chunk_words,
            overlap_words=args.overlap_words,
            meta={"article_id": sid},
        )
        out_bm25 = summarize_rag_over_article(
            art_chunks,
            retriever_type="bm25",
            rag_cfg=rag_cfg,
            generator=generator,
            gen_cfg=gen_cfg,
            dense_model_name=args.dense_model,
        )
        out_dense = summarize_rag_over_article(
            art_chunks,
            retriever_type="dense",
            rag_cfg=rag_cfg,
            generator=generator,
            gen_cfg=gen_cfg,
            dense_model_name=args.dense_model,
        )

        sum_rows.append({"task": "sum", "id": sid, "system": "llm", "prediction": pred_llm, "reference": ref})
        sum_rows.append({"task": "sum", "id": sid, "system": "rag_bm25", "prediction": out_bm25["summary"], "reference": ref})
        sum_rows.append({"task": "sum", "id": sid, "system": "rag_dense", "prediction": out_dense["summary"], "reference": ref})
        if getattr(args, "hybrid", False):
            out_hybrid_sum = summarize_rag_over_article(
                art_chunks,
                retriever_type="hybrid",
                rag_cfg=rag_cfg,
                generator=generator,
                gen_cfg=gen_cfg,
                dense_model_name=args.dense_model,
            )
            sum_rows.append({"task": "sum", "id": sid, "system": "rag_hybrid", "prediction": out_hybrid_sum["summary"], "reference": ref})

    sum_df = pd.DataFrame(sum_rows)
    sum_df.to_csv(out_dir / "predictions_sum.csv", index=False)

    sum_metrics = []
    for system, sdf in sum_df.groupby("system"):
        preds = sdf["prediction"].tolist()
        refs = sdf["reference"].tolist()
        m = {"system": system, "bleu": compute_bleu(preds, refs), **compute_rouge(preds, refs)}
        if not args.skip_bertscore:
            m.update(compute_bertscore(preds, refs, model_type=args.bertscore_model))
        sum_metrics.append(m)
    sum_metrics_df = pd.DataFrame(sum_metrics)
    sum_metrics_df.to_csv(out_dir / "metrics_sum.csv", index=False)

    # -----------------
    # Per-example scoring
    # -----------------
    all_df = pd.concat([qa_df, sum_df], ignore_index=True)
    all_df["rougeL_f1"] = [rougeL_f1(p, r) for p, r in zip(all_df["prediction"], all_df["reference"])]
    all_df["exact_match_norm"] = [
        1.0 if normalize_text(p) == normalize_text(r) else 0.0 for p, r in zip(all_df["prediction"], all_df["reference"])
    ]
    all_df.to_csv(out_dir / "predictions_all_scored.csv", index=False)

    # -----------------
    # Mandatory figures
    # -----------------
    if "bertscore_F1" in qa_metrics_df.columns:
        qa_bar = qa_metrics_df[["system", "qa_accuracy", "rougeL", "bertscore_F1"]].copy()
        fig = viz.bar_avg_metrics(qa_bar, ["qa_accuracy", "rougeL", "bertscore_F1"], title="QA: average metrics by system")
    else:
        qa_bar = qa_metrics_df[["system", "qa_accuracy", "rougeL"]].copy()
        fig = viz.bar_avg_metrics(qa_bar, ["qa_accuracy", "rougeL"], title="QA: average metrics by system")
    fig.savefig(fig_dir / "qa_bar_avg_metrics.png", dpi=200)

    if "bertscore_F1" in sum_metrics_df.columns:
        sum_bar = sum_metrics_df[["system", "rougeL", "bertscore_F1", "bleu"]].copy()
        sum_bar["bleu"] = sum_bar["bleu"] / 100.0
        fig = viz.bar_avg_metrics(sum_bar, ["rougeL", "bertscore_F1", "bleu"], title="Summarization: average metrics by system")
        fig.savefig(fig_dir / "sum_bar_avg_metrics.png", dpi=200)
    else:
        sum_bar = sum_metrics_df[["system", "rougeL", "bleu"]].copy()
        sum_bar["bleu"] = sum_bar["bleu"] / 100.0
        fig = viz.bar_avg_metrics(sum_bar, ["rougeL", "bleu"], title="Summarization: average metrics by system")
        fig.savefig(fig_dir / "sum_bar_avg_metrics.png", dpi=200)

    long = all_df[["task", "system", "rougeL_f1"]].rename(columns={"rougeL_f1": "score"})
    long["metric"] = "rougeL_f1"
    for task in ["qa", "sum"]:
        fig = viz.boxplot_distributions(long[long["task"] == task][["system", "metric", "score"]], title=f"{task.upper()}: ROUGE-L F1 distribution")
        fig.savefig(fig_dir / f"{task}_box_rougeL.png", dpi=200)

    # Radar chart requires same columns; only when BERTScore exists
    if "bertscore_F1" in sum_metrics_df.columns:
        rad = sum_metrics_df[["system", "rouge1", "rouge2", "rougeL", "bertscore_F1", "bleu"]].copy()
        rad["bleu"] = rad["bleu"] / 100.0
        fig = viz.radar_multimetric(rad, ["rouge1", "rouge2", "rougeL", "bertscore_F1", "bleu"], title="Summarization: multi-metric comparison")
        fig.savefig(fig_dir / "sum_radar_multimetric.png", dpi=200)

    # -----------------
    # Manual annotation template
    # -----------------
    rng = np.random.default_rng(args.seed)
    qa_ids = qa_df["id"].drop_duplicates().tolist()
    sum_ids = sum_df["id"].drop_duplicates().tolist()
    qa_sample = rng.choice(qa_ids, size=min(args.manual_qa_n, len(qa_ids)), replace=False)
    sum_sample = rng.choice(sum_ids, size=min(args.manual_sum_n, len(sum_ids)), replace=False)

    manual_df = all_df[
        ((all_df["task"] == "qa") & (all_df["id"].isin(qa_sample)))
        | ((all_df["task"] == "sum") & (all_df["id"].isin(sum_sample)))
    ].copy()
    manual_df["manual_label"] = ""
    manual_template_path = out_dir / "manual_annotations_template.csv"
    manual_df.to_csv(manual_template_path, index=False)

    # If filled file exists, generate manual-eval figures
    filled_path = out_dir / "manual_annotations_filled.csv"
    if filled_path.exists():
        annotations = read_manual_annotations_csv(filled_path)
        # Keep only keys + manual label; recompute metrics from `all_df` to avoid Excel corruption.
        keep_cols = ["task", "id", "system", "manual_label"]
        missing = [c for c in keep_cols if c not in annotations.columns]
        if missing:
            raise ValueError(f"manual_annotations_filled.csv is missing columns: {missing}")

        annotations = annotations[keep_cols].copy()
        annotations["manual_label"] = annotations["manual_label"].astype(str).str.strip()

        filled = all_df.merge(annotations, on=["task", "id", "system"], how="inner")
        filled["manual_numeric"] = manual_label_to_numeric(filled["manual_label"].tolist())

        fig = viz.scatter_manual_vs_metric(
            filled.dropna(subset=["manual_numeric", "rougeL_f1"]),
            manual_numeric_col="manual_numeric",
            metric_col="rougeL_f1",
            title="Manual vs ROUGE-L F1 (all tasks)",
        )
        fig.savefig(fig_dir / "scatter_manual_vs_rougeL.png", dpi=200)

        rho = correlation_manual_vs_metric(filled, manual_col="manual_numeric", metric_col="rougeL_f1")
        print("Spearman correlation (manual vs ROUGE-L F1):", rho)
        (out_dir / "spearman_rho.txt").write_text(str(rho), encoding="utf-8")

        fig = viz.stacked_hallucination_breakdown(
            filled,
            title="Manual label breakdown (proxy for hallucination/correctness)",
        )
        fig.savefig(fig_dir / "stacked_manual_breakdown.png", dpi=200)

    print("Done.")
    print("Outputs:", out_dir.resolve())


if __name__ == "__main__":
    main()

