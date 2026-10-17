#!/usr/bin/env bash
set -euo pipefail

EXP_ID="${1:-E0}"
echo "=========================================="
echo "Starting Experiment: $EXP_ID"
echo "=========================================="

RESULTS_DIR="results/$EXP_ID"
mkdir -p "$RESULTS_DIR"

python scripts/gpu_budget.py start "Experiment $EXP_ID"

# 1. Run answer evaluation against the running model
echo "--- Step 1: Running Answer Quality Evaluation ---"
python eval/run_answer_eval.py --out "$RESULTS_DIR/answer_eval.json"

# 2. Run load tests across concurrency levels (1, 8, 32, 64)
echo "--- Step 2: Running Concurrency Load Tests ---"
for c in 1 8 32 64; do
    echo "Running load test at concurrency=$c..."
    python benchmarks/load_test.py \
        --concurrency "$c" \
        --requests $((c * 10)) \
        --out "$RESULTS_DIR/load_c${c}.json"
done

# 3. Stop budget tracker
python scripts/gpu_budget.py stop

echo "Experiment $EXP_ID completed. Output files stored in $RESULTS_DIR"
