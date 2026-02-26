# Run TinyLlama on GPU, then save metrics to metrics_*_mistral.csv and restore main metrics.
# Run this AFTER run_ablations has finished (so outputs/ has your main FLAN-T5 metrics).
# Prerequisite: backup of main metrics exists (or we create one at start).

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

# 1) Backup current main metrics (in case they're the ones we want to keep)
if (Test-Path "outputs\metrics_qa.csv") {
    Copy-Item "outputs\metrics_qa.csv" "outputs\metrics_qa_main_backup.csv" -Force
    Copy-Item "outputs\metrics_sum.csv" "outputs\metrics_sum_main_backup.csv" -Force
    Write-Host "Backed up current metrics to metrics_*_main_backup.csv"
}

# 2) Run TinyLlama with GPU
Write-Host "Running TinyLlama on GPU (--device cuda) ..."
python -m scripts.run_experiments --use-tinyllama --hybrid --device cuda --skip-bertscore
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# 3) Save TinyLlama results as metrics_*_mistral.csv
Copy-Item "outputs\metrics_qa.csv" "outputs\metrics_qa_mistral.csv" -Force
Copy-Item "outputs\metrics_sum.csv" "outputs\metrics_sum_mistral.csv" -Force
Write-Host "Saved TinyLlama metrics to metrics_qa_mistral.csv and metrics_sum_mistral.csv"

# 4) Restore main metrics (FLAN-T5) so paper build uses them
if (Test-Path "outputs\metrics_qa_main_backup.csv") {
    Copy-Item "outputs\metrics_qa_main_backup.csv" "outputs\metrics_qa.csv" -Force
    Copy-Item "outputs\metrics_sum_main_backup.csv" "outputs\metrics_sum.csv" -Force
    Write-Host "Restored main metrics (outputs/metrics_qa.csv, metrics_sum.csv)"
}

Write-Host "Done. Rebuild paper with: python -m scripts.build_paper_docx"
