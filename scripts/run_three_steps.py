"""
Run the three guideline steps in order: (1) Reranker, (2) Larger model, (3) Chunk/top-k ablations.
Then rebuild the paper.

Prerequisites:
  - data/processed/ populated (python -m scripts.download_data)
  - pip install protobuf  (so --reranker works with MonoT5)

Usage:
  python -m scripts.run_three_steps [--larger-model mistral|tinyllama] [--no-ablations] [--no-paper]

Step 1: FLAN-T5 + hybrid + reranker -> main outputs (metrics_qa.csv, metrics_sum.csv).
Step 2: Larger model run -> metrics_qa_mistral.csv, metrics_sum_mistral.csv; main outputs restored.
Step 3: Ablations -> ablation_qa.csv, ablation_sum.csv; main outputs restored.
Step 4: Build paper/LLM_vs_RAG_QA_Summarization.docx (unless --no-paper).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"


def run(cmd: list[str], desc: str) -> bool:
    print(f"\n>>> {desc}")
    print(" ".join(cmd))
    r = subprocess.run(cmd, cwd=str(ROOT))
    if r.returncode != 0:
        print(f"Failed: {desc}", file=sys.stderr)
        return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="Run reranker, larger model, ablations; rebuild paper")
    ap.add_argument("--larger-model", choices=["mistral", "tinyllama"], default="tinyllama", help="Larger model: mistral (slow) or tinyllama (fast)")
    ap.add_argument("--device", type=str, default=None, help="Pass --device cuda to run_experiments for GPU (e.g. TinyLlama, Mistral, FLAN-T5)")
    ap.add_argument("--no-ablations", action="store_true", help="Skip chunk/top-k ablations")
    ap.add_argument("--no-paper", action="store_true", help="Do not run build_paper_docx")
    args = ap.parse_args()

    # Step 1: Main run with reranker
    step1_cmd = [sys.executable, "-m", "scripts.run_experiments", "--hybrid", "--reranker", "--skip-bertscore"]
    if args.device:
        step1_cmd.extend(["--device", args.device])
    if not run(step1_cmd, "Step 1: FLAN-T5 + hybrid + reranker"):
        sys.exit(1)

    # Backup main metrics
    qa_main = OUT / "metrics_qa.csv"
    sum_main = OUT / "metrics_sum.csv"
    flan_qa = OUT / "metrics_qa_flan_reranker.csv"
    flan_sum = OUT / "metrics_sum_flan_reranker.csv"
    flan_qa.write_bytes(qa_main.read_bytes())
    flan_sum.write_bytes(sum_main.read_bytes())
    print("Backed up main metrics to metrics_*_flan_reranker.csv")

    # Step 2: Larger model
    larger_flag = "--use-mistral" if args.larger_model == "mistral" else "--use-tinyllama"
    step2_cmd = [sys.executable, "-m", "scripts.run_experiments", larger_flag, "--hybrid", "--skip-bertscore"]
    if args.device:
        step2_cmd.extend(["--device", args.device])
    if not run(step2_cmd, f"Step 2: Larger model ({args.larger_model})"):
        sys.exit(1)
    mistral_qa = OUT / "metrics_qa_mistral.csv"
    mistral_sum = OUT / "metrics_sum_mistral.csv"
    mistral_qa.write_bytes(qa_main.read_bytes())
    mistral_sum.write_bytes(sum_main.read_bytes())
    qa_main.write_bytes(flan_qa.read_bytes())
    sum_main.write_bytes(flan_sum.read_bytes())
    print("Saved larger-model metrics to metrics_*_mistral.csv; restored main metrics.")

    # Step 3: Ablations (uses default hybrid; backs up/restores main metrics internally)
    if not args.no_ablations:
        if not run(
            [sys.executable, "-m", "scripts.run_ablations", "--hybrid", "--skip-bertscore"],
            "Step 3: Chunk and top-k ablations",
        ):
            sys.exit(1)

    # Step 4: Paper
    if not args.no_paper:
        if not run([sys.executable, "-m", "scripts.build_paper_docx"], "Step 4: Build paper"):
            sys.exit(1)

    print("\nAll steps completed.")


if __name__ == "__main__":
    main()
