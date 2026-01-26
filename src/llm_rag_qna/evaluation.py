from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from bert_score import score as bert_score
from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU

from .utils import normalize_text


def qa_accuracy(preds: Sequence[str], refs: Sequence[str]) -> float:
    """
    Accuracy = exact match after normalization.
    This is intentionally strict; report manual evaluation alongside it.
    """
    correct = 0
    for p, r in zip(preds, refs):
        if normalize_text(p) == normalize_text(r):
            correct += 1
    return correct / max(1, len(refs))


def compute_rouge(preds: Sequence[str], refs: Sequence[str]) -> Dict[str, float]:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    agg = {"rouge1": [], "rouge2": [], "rougeL": []}
    for p, r in zip(preds, refs):
        scores = scorer.score(r, p)  # (target, prediction)
        for k in agg:
            agg[k].append(scores[k].fmeasure)
    return {k: float(np.mean(v)) for k, v in agg.items()}


def compute_bleu(preds: Sequence[str], refs: Sequence[str]) -> float:
    bleu = BLEU(effective_order=True)
    # sacrebleu expects list of hypotheses and list of list of refs
    return float(bleu.corpus_score(preds, [list(refs)]).score)


def compute_bertscore(
    preds: Sequence[str],
    refs: Sequence[str],
    *,
    lang: str = "en",
    model_type: str = "microsoft/deberta-xlarge-mnli",
) -> Dict[str, float]:
    """
    Returns mean precision/recall/F1.
    Warning: this model is large; for coursework you can switch to a smaller one.
    """
    P, R, F1 = bert_score(preds, refs, lang=lang, model_type=model_type, verbose=True)
    return {"bertscore_P": float(P.mean()), "bertscore_R": float(R.mean()), "bertscore_F1": float(F1.mean())}


@dataclass
class ManualLabelScheme:
    correct: str = "correct"
    partial: str = "partially_correct"
    incorrect: str = "incorrect_or_hallucinated"


def manual_label_to_numeric(
    labels: Sequence[str],
    scheme: ManualLabelScheme = ManualLabelScheme(),
) -> List[float]:
    m = {
        scheme.correct: 1.0,
        scheme.partial: 0.5,
        scheme.incorrect: 0.0,
    }
    return [m.get(str(x).strip().lower(), np.nan) for x in labels]


def correlation_manual_vs_metric(
    df: pd.DataFrame,
    *,
    manual_col: str,
    metric_col: str,
) -> float:
    a = df[manual_col].astype(float)
    b = df[metric_col].astype(float)
    ok = a.notna() & b.notna()
    if ok.sum() < 3:
        return float("nan")
    return float(a[ok].corr(b[ok], method="spearman"))

