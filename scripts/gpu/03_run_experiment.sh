#!/usr/bin/env bash
set -euo pipefail

EXP_ID="${1:-E0}"
echo "=========================================="
echo "Starting Experiment: $EXP_ID"
echo "=========================================="

RESULTS_DIR="results/$EXP_ID"
mkdir -p "$RESULTS_DIR"

python scripts/gpu_budget.py start "Experiment $EXP_ID"

# 1. Run answer evaluation and save raw answers on every run (Pass 1 of two-pass flow)
echo "--- Step 1: Generating and Saving Raw Answers (Pass 1) ---"
if [ -n "${JUDGE_MODEL:-}" ]; then
    python eval/run_answer_eval.py \
        --raw-out "$RESULTS_DIR/raw_answers.jsonl" \
        --judged-out "$RESULTS_DIR/judged_answers.jsonl" \
        --sample-out "$RESULTS_DIR/manual_review_sample_20pct.jsonl" \
        --metrics-out "$RESULTS_DIR/answer_eval_metrics.json" \
        --judge-model "$JUDGE_MODEL"
else
    echo "Notice: JUDGE_MODEL not set; saving raw answers and using smoke-test rubric. Official judging runs in Pass 2."
    python eval/run_answer_eval.py \
        --raw-out "$RESULTS_DIR/raw_answers.jsonl" \
        --judged-out "$RESULTS_DIR/judged_answers.jsonl" \
        --sample-out "$RESULTS_DIR/manual_review_sample_20pct.jsonl" \
        --metrics-out "$RESULTS_DIR/answer_eval_metrics.json" \
        --smoke-test-only
fi

if [ ! -s "$RESULTS_DIR/raw_answers.jsonl" ]; then
    echo "ERROR: $RESULTS_DIR/raw_answers.jsonl was not generated or is empty!"
    exit 1
fi
echo "Raw answers successfully saved to $RESULTS_DIR/raw_answers.jsonl"

# 2. Run load tests across concurrency levels (1, 8, 32, 64)
echo "--- Step 2: Running Concurrency Load Tests ---"
for c in 1 8 32 64; do
    echo "Running load test at concurrency=$c..."
    # Ensure minimum 50 measured requests for statistical validity (p95/p99 tail latency)
    NUM_REQS=$((c * 10))
    if [ "$NUM_REQS" -lt 50 ]; then
        NUM_REQS=50
    fi
    python benchmarks/load_test.py \
        --concurrency "$c" \
        --requests "$NUM_REQS" \
        --out "$RESULTS_DIR/load_c${c}.json"
done

# 3. Stop budget tracker
python scripts/gpu_budget.py stop

echo "Experiment $EXP_ID completed. Output files stored in $RESULTS_DIR"
