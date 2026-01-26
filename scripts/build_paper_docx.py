from __future__ import annotations

"""
Build a .docx paper following the provided template style cues (Times New Roman, IMRaD).

Inputs (from this repo):
- paper/IMRaD_draft.md (text source)
- paper/figure_captions.md (caption text source)
- outputs/metrics_qa.csv, outputs/metrics_sum.csv (tables)
- outputs/figures/*.png (figures)
- outputs/manual_annotations_filled.csv (optional; used indirectly via correlation already computed)

Output:
- paper/LLM_vs_RAG_QA_Summarization.docx
"""

from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt


ROOT = Path(__file__).resolve().parents[1]


def set_run_font(run, *, size_pt: int, bold: bool = False, italic: bool = False) -> None:
    run.font.name = "Times New Roman"
    run.font.size = Pt(size_pt)
    run.bold = bold
    run.italic = italic


def add_paragraph(
    doc: Document,
    text: str,
    *,
    size_pt: int = 12,
    bold: bool = False,
    italic: bool = False,
    align: str = "justify",  # 'left'|'center'|'right'|'justify'
) -> None:
    p = doc.add_paragraph()
    r = p.add_run(text)
    set_run_font(r, size_pt=size_pt, bold=bold, italic=italic)
    if align == "center":
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    elif align == "right":
        p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    elif align == "left":
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    else:
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY


def add_heading_1(doc: Document, title: str) -> None:
    # Template: "1. Introduction (TNR 14pt., bold)"
    add_paragraph(doc, title, size_pt=14, bold=True, align="left")


def add_heading_2(doc: Document, title: str) -> None:
    # Template: "1.1 Title of the 2nd level (TNR 12pt., bold)"
    add_paragraph(doc, title, size_pt=12, bold=True, align="left")


def add_table_from_df(doc: Document, df: pd.DataFrame, *, float_fmt: str = "{:.4f}") -> None:
    table = doc.add_table(rows=1, cols=len(df.columns))
    table.style = "Table Grid"
    hdr = table.rows[0].cells
    for j, col in enumerate(df.columns):
        hdr[j].text = str(col)
        for run in hdr[j].paragraphs[0].runs:
            set_run_font(run, size_pt=10, bold=True)

    for _, row in df.iterrows():
        cells = table.add_row().cells
        for j, col in enumerate(df.columns):
            v = row[col]
            if isinstance(v, float):
                txt = float_fmt.format(v)
            else:
                txt = str(v)
            cells[j].text = txt
            for run in cells[j].paragraphs[0].runs:
                set_run_font(run, size_pt=10)


def add_figure(doc: Document, img_path: Path, caption: str, *, width_in: float = 6.0) -> None:
    if img_path.exists():
        doc.add_picture(str(img_path), width=Inches(width_in))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    # Caption format (template): TNR 10pt, centered, italics
    add_paragraph(doc, caption, size_pt=10, italic=True, align="center")
    add_paragraph(doc, "Source: Authors' experiments.", size_pt=10, italic=True, align="center")


def main() -> None:
    out_path = ROOT / "paper" / "LLM_vs_RAG_QA_Summarization.docx"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load numeric results
    qa = pd.read_csv(ROOT / "outputs" / "metrics_qa.csv")
    su = pd.read_csv(ROOT / "outputs" / "metrics_sum.csv")

    # Map internal ids to paper naming
    name_map = {"llm": "LLM", "rag_bm25": "RAG-BM25", "rag_dense": "RAG-Dense"}
    qa["system"] = qa["system"].map(name_map)
    su["system"] = su["system"].map(name_map)

    # Keep compact columns for tables (paper-friendly)
    qa_tbl = qa[["system", "qa_accuracy", "rougeL", "bertscore_F1"]].copy()
    su_tbl = su[["system", "rouge1", "rouge2", "rougeL", "bertscore_F1", "bleu"]].copy()

    # Known correlation from your run
    spearman_rho = 0.08970522700074382

    doc = Document()

    # -------- Title block (template cues) --------
    add_paragraph(
        doc,
        "Comparative Analysis of LLM and Retrieval-Augmented Generation (RAG) Systems for Question Answering and Document Summarization",
        size_pt=20,
        bold=True,
        align="center",
    )
    add_paragraph(doc, "Full First Author1, Full Second Author2, * and Full Third Author3", size_pt=12, align="center")
    add_paragraph(doc, "1 Full affiliation of first author, including country", size_pt=10, align="center")
    add_paragraph(doc, "2 Full affiliation of second author, including country", size_pt=10, align="center")
    add_paragraph(doc, "3 Full affiliation of third author, including country", size_pt=10, align="center")
    add_paragraph(doc, "* Corresponding author", size_pt=10, align="center")

    # -------- Abstract (structured) --------
    add_paragraph(doc, "Abstract", size_pt=12, bold=True, align="left")
    add_paragraph(
        doc,
        "Purpose – This study compares a standalone Large Language Model (LLM) against Retrieval-Augmented Generation (RAG) pipelines for question answering and document summarization, focusing on correctness, hallucination reduction, and consistency.",
        size_pt=12,
    )
    add_paragraph(
        doc,
        "Design/Methodology/Approach – We evaluate three systems (LLM, RAG-BM25, RAG-Dense) under controlled conditions: the same generator model and decoding settings, identical prompts except for context injection, and a reproducible dataset subset (SQuAD for Q&A; CNN/DailyMail for summarization). Retrieval uses BM25 and SentenceTransformers dense embeddings with top-k=5 and word-based document chunking (200 words, 50 overlap).",
        size_pt=12,
    )
    add_paragraph(
        doc,
        "Findings – In our setting, dense retrieval yields higher QA exact-match accuracy than the LLM baseline, while summarization overlap metrics favor the baseline LLM. Manual evaluation shows that automatic overlap metrics are weakly aligned with human correctness judgments (Spearman’s ρ = 0.0897).",
        size_pt=12,
    )
    add_paragraph(
        doc,
        "Originality/Value – The project provides a fully reproducible coursework-scale benchmark and highlights the necessity of manual evaluation for factuality-sensitive generation tasks.",
        size_pt=12,
    )
    add_paragraph(doc, "Keywords – Large Language Models; Retrieval-Augmented Generation; BM25; Dense Retrieval; Question Answering; Summarization; Hallucination.", size_pt=12)
    add_paragraph(doc, "Paper Type – Research paper.", size_pt=12)

    # -------- IMRaD body --------
    add_heading_1(doc, "1. Introduction")
    add_paragraph(
        doc,
        "Large Language Models (LLMs) are widely used for generative NLP tasks such as question answering and summarization. However, LLMs can produce hallucinations—fluent but unsupported statements—which undermines reliability in high-stakes use cases. Retrieval-Augmented Generation (RAG) mitigates this risk by retrieving evidence from documents and conditioning generation on the retrieved context (Lewis et al., 2020). This study investigates whether RAG improves correctness and reduces hallucinations compared to a standalone LLM baseline, and whether sparse (BM25) and dense retrieval differ in performance.",
    )
    add_paragraph(
        doc,
        "Research questions: (1) Does RAG improve answer correctness compared to a standalone LLM? (2) Does RAG reduce hallucinations in Q&A and summarization tasks? (3) How do BM25 and dense retrieval differ in performance? (4) Are automatic metrics aligned with manual evaluation?",
    )

    add_heading_1(doc, "2. Related work")
    add_paragraph(
        doc,
        "RAG combines retrieval and generation to ground model outputs in external text, originally formalized by Lewis et al. (2020). Retrieval itself can be implemented via lexical methods such as BM25 (Robertson & Zaragoza, 2009) or dense bi-encoders such as DPR (Karpukhin et al., 2020). Dense retrieval is typically trained to match semantically related passages and is commonly implemented using Sentence-BERT-style encoders (Reimers & Gurevych, 2019).",
    )
    add_paragraph(
        doc,
        "Hallucination and factuality have been studied extensively in summarization and other NLG settings, where overlap-based metrics may not capture factual errors (Maynez et al., 2020). Recent surveys summarize hallucination causes and mitigation strategies for modern LLMs (Ji et al., 2023). On the evaluation side, ROUGE (Lin, 2004) and BLEU (Papineni et al., 2002) measure n-gram overlap, while BERTScore (Zhang et al., 2020) measures semantic similarity using contextual embeddings; none of these guarantees factual correctness, motivating manual evaluation in factuality-sensitive experiments.",
    )
    add_paragraph(
        doc,
        "Additional related approaches include retrieval-augmented pretraining (Guu et al., 2020) and reader architectures that aggregate multiple retrieved passages (Izacard & Grave, 2021). Surveys of retrieval-augmented LLM systems further discuss retrieval design choices and failure modes (Gao et al., 2023).",
    )

    add_heading_1(doc, "3. Methodology")
    add_heading_2(doc, "3.1 Dataset description")
    add_paragraph(
        doc,
        "Q&A dataset: We use a reproducible subset of SQuAD v1.1 with N=80 questions. Each question has a reference answer and an associated evidence passage. We build the retrieval corpus from these passages to ensure questions are answerable from retrieved documents.",
    )
    add_paragraph(
        doc,
        "Summarization dataset: We use a reproducible subset of CNN/DailyMail (3.0.0) with M=25 articles (test split). Reference summaries are the provided highlights. Article length is restricted to short/medium range to keep runtime feasible on CPU.",
    )
    add_paragraph(
        doc,
        "Limitations: The subsets are small compared to standard benchmark sizes; this is intentional for a controlled, reproducible comparison under limited compute. Conclusions should be interpreted as comparative rather than state-of-the-art claims.",
    )

    add_heading_2(doc, "3.2 Preprocessing and chunking")
    add_paragraph(
        doc,
        "We apply word-based sliding-window chunking with chunk size C=200 words and overlap O=50 words. Overlap reduces boundary effects where evidence may be split across chunks. The same chunking is used for both BM25 and dense retrieval to ensure fairness.",
    )

    add_heading_2(doc, "3.3 Systems and retrievers")
    add_paragraph(
        doc,
        "LLM baseline: the generator answers questions or summarizes full articles without external retrieval.",
    )
    add_paragraph(
        doc,
        "RAG-BM25: retrieve top-k=5 chunks using BM25 (lexical matching) and inject the retrieved text as context into the prompt.",
    )
    add_paragraph(
        doc,
        "RAG-Dense: retrieve top-k=5 chunks using sentence-transformers/all-MiniLM-L6-v2 embeddings and cosine similarity, then inject the retrieved text as context.",
    )

    add_heading_2(doc, "3.4 Prompting and inference settings")
    add_paragraph(
        doc,
        "To control confounds, prompts are identical across systems except for the presence of a context block in RAG. The generator model is google/flan-t5-small with deterministic decoding (temperature=0.0) and max_new_tokens=128.",
    )

    add_heading_2(doc, "3.5 Evaluation")
    add_paragraph(
        doc,
        "Automatic metrics: Q&A accuracy (normalized exact match), BLEU, ROUGE-1/2/L, and BERTScore are reported. Manual evaluation uses three labels: correct, partially_correct, incorrect_or_hallucinated. Manual labels are mapped to numeric scores (1, 0.5, 0) for correlation analysis.",
    )

    add_heading_1(doc, "4. Results")
    add_paragraph(doc, "Tab. 1 and Tab. 2 report the automatic metric averages for Q&A and summarization, respectively.")

    add_paragraph(doc, "Table 1: Q&A metrics by system (SQuAD subset, N=80).", size_pt=10, italic=True, align="center")
    add_table_from_df(doc, qa_tbl, float_fmt="{:.4f}")
    add_paragraph(doc, "Source: Authors' experiments.", size_pt=10, italic=True, align="center")

    add_paragraph(doc, "Table 2: Summarization metrics by system (CNN/DailyMail subset, M=25).", size_pt=10, italic=True, align="center")
    add_table_from_df(doc, su_tbl, float_fmt="{:.4f}")
    add_paragraph(doc, "Source: Authors' experiments.", size_pt=10, italic=True, align="center")

    add_paragraph(
        doc,
        f"Manual vs automatic alignment: Spearman’s ρ between manual correctness and ROUGE-L F1 is {spearman_rho:.4f}, indicating weak alignment in this setting.",
    )

    # Figures (embed PNGs + captions)
    fig_dir = ROOT / "outputs" / "figures"
    add_paragraph(doc, "Fig. 1–7 visualize averages, distributions, and manual evaluation outcomes.", size_pt=12)

    add_figure(doc, fig_dir / "qa_bar_avg_metrics.png", "Figure 1: Average Q&A performance across systems on the SQuAD subset (N=80).", width_in=6.2)
    add_figure(doc, fig_dir / "sum_bar_avg_metrics.png", "Figure 2: Average summarization performance across systems on the CNN/DailyMail subset (M=25).", width_in=6.2)
    add_figure(doc, fig_dir / "qa_box_rougeL.png", "Figure 3: Distribution of per-example ROUGE-L F1 for Q&A across systems.", width_in=6.2)
    add_figure(doc, fig_dir / "sum_box_rougeL.png", "Figure 4: Distribution of per-example ROUGE-L F1 for summarization across systems.", width_in=6.2)
    add_figure(doc, fig_dir / "sum_radar_multimetric.png", "Figure 5: Multi-metric comparison for summarization (ROUGE-1/2/L, BERTScore F1, BLEU).", width_in=6.2)
    add_figure(doc, fig_dir / "scatter_manual_vs_rougeL.png", "Figure 6: Manual correctness (numeric) vs ROUGE-L F1; Spearman’s ρ is reported in text.", width_in=6.0)
    add_figure(doc, fig_dir / "stacked_manual_breakdown.png", "Figure 7: Manual label breakdown across systems (correct / partial / incorrect-hallucinated).", width_in=6.0)

    add_heading_1(doc, "5. Discussion")
    add_heading_2(doc, "5.1 Conclusions")
    add_paragraph(
        doc,
        "For Q&A, the dense RAG variant achieved higher strict accuracy than the standalone LLM baseline, suggesting that evidence retrieval can improve correctness when the retriever successfully surfaces answer-bearing chunks. However, improvements are not uniform across metrics and systems, and retrieval failures can still lead to incorrect answers.",
    )
    add_paragraph(
        doc,
        "For summarization, overlap-based metrics favored the baseline LLM in this configuration. A plausible explanation is that retrieval (within-article chunk selection) can omit important information if top-k chunks do not cover the full article content, reducing overlap with reference highlights.",
    )

    add_heading_2(doc, "5.2 Theoretical implications")
    add_paragraph(
        doc,
        "The weak correlation between manual correctness and ROUGE-L (ρ≈0.09) supports prior observations that overlap metrics do not reliably capture factuality. This underscores the need to complement automatic metrics with manual evaluation when hallucination reduction is a key goal.",
    )

    add_heading_2(doc, "5.3 Practical implications")
    add_paragraph(
        doc,
        "In practice, BM25 can be strong when queries share exact terms with relevant passages, while dense retrieval may help with paraphrases but can retrieve topically related yet non-answering content. Prompt constraints (“use only the provided context”) reduce but do not eliminate hallucinations if the retrieved context is itself misleading or incomplete.",
    )

    add_heading_2(doc, "5.4 Limitations and future research")
    add_paragraph(
        doc,
        "Limitations include small dataset subsets, a single generator model, and limited hyperparameter exploration. Future work should test larger subsets, additional generators (including API-based models), alternative chunk sizes and top-k values, reranking, and citation-style prompting to improve grounding.",
    )

    add_heading_1(doc, "References")
    add_paragraph(doc, "References are formatted in APA 7th edition.", size_pt=12)

    # Minimal, real references (APA 7 style). Students can expand/adjust.
    refs = [
        "Gao, Y., Xiong, Y., Gao, X., Jia, K., Pan, J., Bi, Y., Dai, Y., & Wang, H. (2023). Retrieval-Augmented Generation for Large Language Models: A Survey. arXiv preprint arXiv:2312.10997.",
        "Guu, K., Lee, K., Tung, Z., Pasupat, P., & Chang, M.-W. (2020). REALM: Retrieval-Augmented Language Model Pre-Training. Proceedings of the 37th International Conference on Machine Learning (ICML).",
        "Izacard, G., & Grave, E. (2021). Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering. Proceedings of the 16th Conference of the European Chapter of the ACL (EACL).",
        "Ji, Z., Lee, N., Frieske, R., Yu, T., Su, D., Xu, Y., Ishii, E., Bang, Y. J., Madotto, A., & Fung, P. (2023). Survey of Hallucination in Natural Language Generation. ACM Computing Surveys, 55(12).",
        "Karpukhin, V., Oguz, B., Min, S., Lewis, P., Wu, L., Edunov, S., Chen, D., & Yih, W.-T. (2020). Dense Passage Retrieval for Open-Domain Question Answering. Proceedings of EMNLP.",
        "Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M., Yih, W.-T., Rocktäschel, T., Riedel, S., & Kiela, D. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. Advances in Neural Information Processing Systems (NeurIPS).",
        "Lin, C.-Y. (2004). ROUGE: A Package for Automatic Evaluation of Summaries. Workshop on Text Summarization Branches Out.",
        "Maynez, J., Narayan, S., Bohnet, B., & McDonald, R. (2020). On Faithfulness and Factuality in Abstractive Summarization. Proceedings of ACL.",
        "Papineni, K., Roukos, S., Ward, T., & Zhu, W.-J. (2002). BLEU: A Method for Automatic Evaluation of Machine Translation. Proceedings of ACL.",
        "Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. Proceedings of EMNLP-IJCNLP.",
        "Robertson, S., & Zaragoza, H. (2009). The Probabilistic Relevance Framework: BM25 and Beyond. Foundations and Trends in Information Retrieval, 3(4), 333–389.",
        "Zhang, T., Kishore, V., Wu, F., Weinberger, K. Q., & Artzi, Y. (2020). BERTScore: Evaluating Text Generation with BERT. International Conference on Learning Representations (ICLR).",
    ]
    for r in refs:
        add_paragraph(doc, r, size_pt=12, align="left")

    doc.save(str(out_path))
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()

