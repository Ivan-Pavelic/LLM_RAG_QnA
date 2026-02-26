"""
Run chunk-size and top-k ablations and write outputs/ablation_qa.csv and outputs/ablation_sum.csv.

Usage:
  python -m scripts.run_ablations [--chunk-words 100 200 300 500] [--top-k 3 5 7 10] [--hybrid] [--skip-bertscore]

Before running: ensure outputs/metrics_qa.csv and metrics_sum.csv exist (e.g. from a main run with
  python -m scripts.run_experiments --hybrid --reranker --skip-bertscore).
This script backs up those files at start and restores them at end so the main paper outputs
are unchanged. Ablation results are appended to outputs/ablation_qa.csv and outputs/ablation_sum.csv.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"


def main() -> None:
    ap = argparse.ArgumentParser(description="Run chunk and top-k ablations; append to ablation_qa.csv / ablation_sum.csv")
    ap.add_argument("--chunk-words", type=int, nargs="+", default=[100, 200, 300], help="Chunk sizes (words); default 100 200 300")
    ap.add_argument("--top-k", type=int, nargs="+", default=[3, 5, 7], help="Top-k values; default 3 5 7")
    ap.add_argument("--hybrid", action="store_true", default=True, help="Pass --hybrid to run_experiments (default True)")
    ap.add_argument("--skip-bertscore", action="store_true", default=True, help="Skip BERTScore (default True)")
    ap.add_argument("--no-skip-bertscore", action="store_false", dest="skip_bertscore", help="Do not skip BERTScore")
    ap.add_argument("--device", type=str, default=None, help="Pass --device cuda to run_experiments for GPU")
    args = ap.parse_args()

    # Back up main metrics so we can restore after ablations
    qa_path = OUT / "metrics_qa.csv"
    sum_path = OUT / "metrics_sum.csv"
    backup_qa = OUT / "_backup_metrics_qa.csv"
    backup_sum = OUT / "_backup_metrics_sum.csv"
    if not qa_path.exists() or not sum_path.exists():
        print("Missing outputs/metrics_qa.csv or metrics_sum.csv. Run a full experiment first (e.g. --hybrid --reranker).", file=sys.stderr)
        sys.exit(1)
    backup_qa.write_bytes(qa_path.read_bytes())
    backup_sum.write_bytes(sum_path.read_bytes())
    print("Backed up main metrics to _backup_metrics_*.csv")

    ablation_qa_path = OUT / "ablation_qa.csv"
    ablation_sum_path = OUT / "ablation_sum.csv"
    qa_rows: list[dict] = []
    sum_rows: list[dict] = []

    chunk_vals = args.chunk_words
    topk_vals = args.top_k
    base_cmd = [
        sys.executable, "-m", "scripts.run_experiments",
        "--hybrid" if args.hybrid else "",
        "--skip-bertscore" if args.skip_bertscore else "",
    ]
    base_cmd = [x for x in base_cmd if x]
    if getattr(args, "device", None):
        base_cmd.extend(["--device", args.device])

    configs: list[tuple[int, int]] = []
    for cw in chunk_vals:
        for k in topk_vals:
            configs.append((cw, k))

    for i, (chunk_words, top_k) in enumerate(configs):
        print(f"\n--- Ablation {i + 1}/{len(configs)}: chunk_words={chunk_words}, top_k={top_k} ---")
        cmd = base_cmd + ["--chunk-words", str(chunk_words), "--top-k", str(top_k)]
        r = subprocess.run(cmd, cwd=str(ROOT))
        if r.returncode != 0:
            print(f"run_experiments failed with exit code {r.returncode}", file=sys.stderr)
            qa_path.write_bytes(backup_qa.read_bytes())
            sum_path.write_bytes(backup_sum.read_bytes())
            sys.exit(r.returncode)

        qa_df = pd.read_csv(qa_path)
        sum_df = pd.read_csv(sum_path)
        for _, row in qa_df.iterrows():
            r = row.to_dict()
            r["chunk_words"] = chunk_words
            r["top_k"] = top_k
            qa_rows.append(r)
        for _, row in sum_df.iterrows():
            r = row.to_dict()
            r["chunk_words"] = chunk_words
            r["top_k"] = top_k
            sum_rows.append(r)

    if qa_rows:
        out_qa = pd.DataFrame(qa_rows)
        cols = ["chunk_words", "top_k"] + [c for c in out_qa.columns if c not in ("chunk_words", "top_k")]
        out_qa = out_qa[cols]
        out_qa.to_csv(ablation_qa_path, index=False)
        print(f"Wrote {ablation_qa_path} ({len(out_qa)} rows)")
    if sum_rows:
        out_sum = pd.DataFrame(sum_rows)
        cols = ["chunk_words", "top_k"] + [c for c in out_sum.columns if c not in ("chunk_words", "top_k")]
        out_sum = out_sum[cols]
        out_sum.to_csv(ablation_sum_path, index=False)
        print(f"Wrote {ablation_sum_path} ({len(out_sum)} rows)")

    # Restore main metrics from backup (we saved bytes at start)
    qa_path.write_bytes(backup_qa.read_bytes())
    sum_path.write_bytes(backup_sum.read_bytes())
    print("Restored outputs/metrics_qa.csv and metrics_sum.csv from backup.")
    print("Done.")


if __name__ == "__main__":
    main()
