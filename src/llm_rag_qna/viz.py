from __future__ import annotations

from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def bar_avg_metrics(df_metrics: pd.DataFrame, metric_cols: Sequence[str], *, title: str) -> plt.Figure:
    """
    df_metrics columns: ['system', ...metrics...]
    """
    melted = df_metrics.melt(id_vars=["system"], value_vars=list(metric_cols), var_name="metric", value_name="score")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=melted, x="metric", y="score", hue="system", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    ax.legend(title="System", loc="best")
    fig.tight_layout()
    return fig


def boxplot_distributions(df_long: pd.DataFrame, *, title: str) -> plt.Figure:
    """
    df_long columns: ['system', 'metric', 'score']
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=df_long, x="metric", y="score", hue="system", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    fig.tight_layout()
    return fig


def scatter_manual_vs_metric(
    df: pd.DataFrame,
    *,
    manual_numeric_col: str,
    metric_col: str,
    title: str,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.scatterplot(data=df, x=manual_numeric_col, y=metric_col, hue="system", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Manual (numeric)")
    ax.set_ylabel(metric_col)
    fig.tight_layout()
    return fig


def radar_multimetric(
    df_metrics: pd.DataFrame,
    metric_cols: Sequence[str],
    *,
    title: str,
) -> plt.Figure:
    """
    Simple radar chart (normalized 0..1 expected).
    """
    import numpy as np

    metrics = list(metric_cols)
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(7, 7))
    ax = plt.subplot(111, polar=True)

    for _, row in df_metrics.iterrows():
        vals = [float(row[m]) for m in metrics]
        vals += vals[:1]
        ax.plot(angles, vals, linewidth=2, label=row["system"])
        ax.fill(angles, vals, alpha=0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_yticklabels([])
    ax.set_title(title)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.15))
    fig.tight_layout()
    return fig


def stacked_hallucination_breakdown(df_manual: pd.DataFrame, *, title: str) -> plt.Figure:
    """
    df_manual columns: ['system', 'manual_label'].
    """
    counts = df_manual.groupby(["system", "manual_label"]).size().reset_index(name="n")
    pivot = counts.pivot(index="system", columns="manual_label", values="n").fillna(0)
    fig, ax = plt.subplots(figsize=(8, 5))
    pivot.plot(kind="bar", stacked=True, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("Count")
    fig.tight_layout()
    return fig

