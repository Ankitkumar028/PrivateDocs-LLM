# Evaluation & Benchmark Summary Report

## 1. Retrieval Metrics (Old Doc-Level vs New Chunk-Level Side-by-Side)

| Metric | Old Document-Level | New Chunk-Level (Unfiltered) | New Chunk-Level (Filtered by doc_id) |
|---|---|---|---|
| **Recall@1** | 97.06% | 38.24% | 38.24% |
| **Recall@3** | 100.00% | 55.88% | 58.82% |
| **Recall@5** | 100.00% | 58.82% | 61.76% |
| **Recall@10** | 100.00% | 76.47% | 76.47% |
| **MRR** | 0.9853 | 0.4920 | 0.5020 |

## 2. Answer Quality & Faithfulness (Decoupled Judging)

| Metric | Value |
|---|---|
| Total Evaluated | 40 |
| Judge Model / Method | deterministic |
| Mean Correctness (0-2) | 1.8 / 2.0 |
| Faithfulness Rate | 100.0% |
| Abstention Accuracy | 100.0% |
| Citation Precision | 27.5% |

Raw outputs saved to `results/raw_answers.jsonl`.
Scored answers saved to `results/judged_answers.jsonl`.
20% audit sample saved to `results/manual_review_sample_20pct.jsonl`.

## 3. Cost & GPU Capacity Modeling

- **Measured Throughput:** 137.2 tokens/sec
- **GPU Instance Rate:** $1.99/hr ($1432.80/month)
- **Effective Daily Capacity (70% util):** 8,297,856 tokens/day
- **Modeled Query Size:** 1500 prompt tokens + 150 completion tokens

| Platform / Model | Per-Query Input Cost | Per-Query Output Cost | Total Cost Per Query |
|---|---|---|---|
| openai/gpt-4o | $0.003750 | $0.001500 | $0.005250 |
| openai/gpt-4o-mini | $0.000225 | $0.000090 | $0.000315 |
| anthropic/claude-3-5-sonnet | $0.004500 | $0.002250 | $0.006750 |
| anthropic/claude-3-5-haiku | $0.001200 | $0.000600 | $0.001800 |