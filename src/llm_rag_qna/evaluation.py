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


def bootstrap_confidence_interval(
    per_example_scores: Sequence[float],
    *,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    random_state: Optional[int] = None,
) -> Tuple[float, float, float]:
    """
    Compute mean and bootstrap (percentile) confidence interval for a metric.
    Returns (mean, lower_bound, upper_bound).
    """
    scores = np.asarray(per_example_scores, dtype=float)
    n = len(scores)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(random_state)
    means = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        means.append(float(np.mean(scores[idx])))
    means = np.array(means)
    alpha = (1 - ci) / 2
    low = float(np.percentile(means, 100 * alpha))
    high = float(np.percentile(means, 100 * (1 - alpha)))
    return float(np.mean(scores)), low, high


def paired_bootstrap_p_value(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    *,
    n_bootstrap: int = 1000,
    random_state: Optional[int] = None,
) -> Tuple[float, float]:
    """
    Paired bootstrap test for H0: mean(scores_a) <= mean(scores_b).
    Returns (difference = mean_a - mean_b, p_value one-sided).
    """
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    if len(a) != len(b) or len(a) == 0:
        return float("nan"), float("nan")
    diff_obs = float(np.mean(a) - np.mean(b))
    rng = np.random.default_rng(random_state)
    n = len(a)
    diffs = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        diffs.append(float(np.mean(a[idx]) - np.mean(b[idx])))
    diffs = np.array(diffs)
    # p_value: proportion of bootstrap diffs <= 0 when testing A > B
    p_value = float(np.mean(diffs <= 0))
    return diff_obs, p_value


def error_analysis_breakdown(
    df: pd.DataFrame,
    *,
    pred_col: str = "prediction",
    ref_col: str = "reference",
    id_col: str = "id",
    system_col: str = "system",
    correct_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Build an error analysis table: per row, whether prediction matches reference
    (normalized exact match), reference length (words), and optional system.
    If correct_col is provided, it is used as the correctness indicator; otherwise
    it is computed from pred_col and ref_col.
    """
    out = df[[id_col, system_col, pred_col, ref_col]].copy()
    if correct_col and correct_col in df.columns:
        out["correct"] = df[correct_col]
    else:
        out["correct"] = [
            1.0 if normalize_text(p) == normalize_text(r) else 0.0
            for p, r in zip(df[pred_col], df[ref_col])
        ]
    out["ref_length"] = [len(str(r).split()) for r in df[ref_col]]
    out["pred_length"] = [len(str(p).split()) for p in df[pred_col]]
    return out

