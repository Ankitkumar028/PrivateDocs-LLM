param(
    [Parameter(Position=0)]
    [string]$Target = "help",
    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$RemainingArgs
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

function Show-Help {
    Write-Host "PrivateDocs-LLM Windows/PowerShell Target Runner" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Usage: .\make [target]" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Available targets:"
    Write-Host "  smoke-local     - Run local end-to-end smoke test"
    Write-Host "  test            - Run pytest suite"
    Write-Host "  lint            - Run ruff linter and mypy type checks"
    Write-Host "  ingest          - Run document indexer CLI"
    Write-Host "  setup-local     - Install dependencies with uv"
    Write-Host "  up-local        - Start local Docker stack"
    Write-Host "  down-local      - Stop local Docker stack"
    Write-Host "  eval-retrieval  - Run retrieval evaluation"
    Write-Host "  eval-answers    - Run answer quality evaluation"
    Write-Host "  loadtest        - Run async load benchmark"
    Write-Host "  report          - Generate evaluation summary report"
    Write-Host "  gpu-up          - Start GPU Docker stack"
}

switch ($Target.ToLower()) {
    "smoke-local" {
        Write-Host "Running local smoke test..." -ForegroundColor Green
        uv run python "$ScriptDir\scripts\local\smoke_test.sh" @RemainingArgs
    }
    "test" {
        Write-Host "Running unit tests..." -ForegroundColor Green
        uv run pytest -v "$ScriptDir\tests" @RemainingArgs
    }
    "lint" {
        Write-Host "Running ruff check..." -ForegroundColor Green
        uv run ruff check "$ScriptDir"
        Write-Host "Running mypy check..." -ForegroundColor Green
        uv run mypy "$ScriptDir\src\privatedocs"
    }
    "ingest" {
        Write-Host "Running document indexer..." -ForegroundColor Green
        uv run python -m privatedocs.ingestion.indexer @RemainingArgs
    }
    "setup-local" {
        Write-Host "Setting up environment with uv..." -ForegroundColor Green
        uv venv "$ScriptDir\.venv"
        uv pip install -e "$ScriptDir`[app,eval,dev`]"
    }
    "up-local" {
        docker compose -f "$ScriptDir\docker\compose.local.yml" up -d @RemainingArgs
    }
    "down-local" {
        docker compose -f "$ScriptDir\docker\compose.local.yml" down @RemainingArgs
    }
    "eval-retrieval" {
        uv run python "$ScriptDir\eval\run_retrieval_eval.py" @RemainingArgs
    }
    "eval-answers" {
        uv run python "$ScriptDir\eval\run_answer_eval.py" @RemainingArgs
    }
    "loadtest" {
        uv run python "$ScriptDir\benchmarks\load_test.py" @RemainingArgs
    }
    "report" {
        uv run python "$ScriptDir\eval\report.py" @RemainingArgs
    }
    "gpu-up" {
        docker compose -f "$ScriptDir\docker\compose.gpu.yml" up -d @RemainingArgs
    }
    default {
        Show-Help
    }
}
