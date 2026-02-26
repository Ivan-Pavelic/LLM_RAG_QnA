param(
    [string]$BundleName = "colab_bundle",
    [switch]$IncludeQAOutputs
)

# Prepare a minimal bundle to upload to Google Colab:
# - Copies source code and scripts
# - Optionally includes QA outputs needed for --only-summarization
# - Creates a ZIP at the project root: $BundleName.zip

$scriptDir   = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Resolve-Path (Join-Path $scriptDir "..")

$bundleDir = Join-Path $projectRoot $BundleName
if (Test-Path $bundleDir) {
    Remove-Item $bundleDir -Recurse -Force
}
New-Item $bundleDir -ItemType Directory | Out-Null

# Helper: copy a directory if it exists
function Copy-IfExistsDir {
    param(
        [string]$Name
    )
    $src = Join-Path $projectRoot $Name
    if (Test-Path $src) {
        Copy-Item $src -Destination $bundleDir -Recurse -Force
    }
}

# Helper: copy a single file if it exists
function Copy-IfExistsFile {
    param(
        [string]$RelativePath
    )
    $src = Join-Path $projectRoot $RelativePath
    if (Test-Path $src) {
        Copy-Item $src -Destination $bundleDir -Force
    }
}

# Core project pieces
Copy-IfExistsDir "src"
Copy-IfExistsDir "scripts"
Copy-IfExistsDir "paper"
Copy-IfExistsDir "docs"

Copy-IfExistsFile "requirements.txt"
Copy-IfExistsFile "pyproject.toml"
Copy-IfExistsFile "setup.cfg"
Copy-IfExistsFile "setup.py"

# Optionally include QA outputs so Colab can run --only-summarization
if ($IncludeQAOutputs) {
    $outSrc = Join-Path $projectRoot "outputs"
    if (Test-Path $outSrc) {
        $outDst = Join-Path $bundleDir "outputs"
        New-Item $outDst -ItemType Directory -Force | Out-Null
        foreach ($f in @("predictions_qa.csv", "metrics_qa.csv")) {
            $p = Join-Path $outSrc $f
            if (Test-Path $p) {
                Copy-Item $p -Destination $outDst -Force
            }
        }
    }
}

# Create ZIP at project root
$zipPath = Join-Path $projectRoot ("{0}.zip" -f $BundleName)
if (Test-Path $zipPath) {
    Remove-Item $zipPath -Force
}

Compress-Archive -Path (Join-Path $bundleDir "*") -DestinationPath $zipPath

Write-Host ""
Write-Host "Created bundle:" -ForegroundColor Green
Write-Host "  $zipPath"
Write-Host ""
Write-Host "Next steps in Google Colab (Python cell):"
Write-Host "------------------------------------------------------------"
Write-Host "from google.colab import files"
Write-Host "uploaded = files.upload()  # select $BundleName.zip"
Write-Host "!unzip $BundleName.zip"
Write-Host "%cd $BundleName"
Write-Host "!pip install -r requirements.txt"
Write-Host "!pip install -e ."
Write-Host ""
Write-Host "# Full TinyLlama run (QA + summarization):"
Write-Host "!python -m scripts.run_experiments --use-tinyllama --hybrid --device cuda --skip-bertscore"
Write-Host ""
Write-Host "# Or only summarization if QA CSVs were included in the bundle:"
Write-Host "!python -m scripts.run_experiments --only-summarization --use-tinyllama --hybrid --device cuda --skip-bertscore"
Write-Host ""
Write-Host "# When done, package outputs and download back to your machine:"
Write-Host "!zip -r outputs_colab.zip outputs"
Write-Host "files.download('outputs_colab.zip')"
Write-Host "------------------------------------------------------------"

