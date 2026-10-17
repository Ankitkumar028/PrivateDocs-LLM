"""Aggregates results into tables and summaries for docs and reporting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def generate_report(results_dir: Path, output_file: Path) -> None:
    """Read JSON metrics from results_dir and compile markdown summary."""
    retrieval_file = results_dir / "retrieval_metrics.json"
    answer_file = results_dir / "answer_eval_metrics.json"
    cost_file = results_dir / "cost_model_summary.json"

    retrieval_data = {}
    if retrieval_file.exists():
        retrieval_data = json.loads(retrieval_file.read_text(encoding="utf-8"))

    answer_data = {}
    if answer_file.exists():
        answer_data = json.loads(answer_file.read_text(encoding="utf-8"))

    cost_data = {}
    if cost_file.exists():
        cost_data = json.loads(cost_file.read_text(encoding="utf-8"))

    old_doc = retrieval_data.get("old_document_level_unfiltered", {})
    new_chunk = retrieval_data.get("new_chunk_level_unfiltered", {})
    new_filtered = retrieval_data.get("new_chunk_level_filtered", {})

    report_lines = [
        "# Evaluation & Benchmark Summary Report",
        "",
        "## 1. Retrieval Metrics (Old Doc-Level vs New Chunk-Level Side-by-Side)",
        "",
        "| Metric | Old Document-Level | New Chunk-Level (Unfiltered) | New Chunk-Level (Filtered by doc_id) |",
        "|---|---|---|---|",
        f"| **Recall@1** | {old_doc.get('recall_at_1', 0):.2%} | {new_chunk.get('recall_at_1', 0):.2%} | {new_filtered.get('recall_at_1', 0):.2%} |",
        f"| **Recall@3** | {old_doc.get('recall_at_3', 0):.2%} | {new_chunk.get('recall_at_3', 0):.2%} | {new_filtered.get('recall_at_3', 0):.2%} |",
        f"| **Recall@5** | {old_doc.get('recall_at_5', 0):.2%} | {new_chunk.get('recall_at_5', 0):.2%} | {new_filtered.get('recall_at_5', 0):.2%} |",
        f"| **Recall@10** | {old_doc.get('recall_at_10', 0):.2%} | {new_chunk.get('recall_at_10', 0):.2%} | {new_filtered.get('recall_at_10', 0):.2%} |",
        f"| **MRR** | {old_doc.get('mrr', 0):.4f} | {new_chunk.get('mrr', 0):.4f} | {new_filtered.get('mrr', 0):.4f} |",
        "",
        "## 2. Answer Quality & Faithfulness (Decoupled Judging)",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Total Evaluated | {answer_data.get('total_evaluated', 'N/A')} |",
        f"| Judge Model / Method | {answer_data.get('judge_method', 'deterministic')} |",
        f"| Mean Correctness (0-2) | {answer_data.get('mean_correctness_0_to_2', 'N/A')} / 2.0 |",
        f"| Faithfulness Rate | {answer_data.get('faithfulness_rate', 0):.1%} |",
        f"| Abstention Accuracy | {answer_data.get('abstention_accuracy', 0):.1%} |",
        f"| Citation Precision | {answer_data.get('citation_precision', 0):.1%} |",
        "",
        "Raw outputs saved to `results/raw_answers.jsonl`.",
        "Scored answers saved to `results/judged_answers.jsonl`.",
        "20% audit sample saved to `results/manual_review_sample_20pct.jsonl`.",
        "",
    ]

    if cost_data:
        g = cost_data.get("gpu_specs", {})
        p = cost_data.get("query_assumptions", {})
        report_lines.extend([
            "## 3. Cost & GPU Capacity Modeling",
            "",
            f"- **Measured Throughput:** {g.get('measured_throughput_tok_sec', 0):.1f} tokens/sec",
            f"- **GPU Instance Rate:** ${g.get('hourly_gpu_rate_usd', 0):.2f}/hr (${g.get('fixed_cost_per_gpu_month_usd', 0):.2f}/month)",
            f"- **Effective Daily Capacity (70% util):** {g.get('single_gpu_effective_daily_capacity', 0):,} tokens/day",
            f"- **Modeled Query Size:** {p.get('input_tokens_per_query', 0)} prompt tokens + {p.get('output_tokens_per_query', 0)} completion tokens",
            "",
            "| Platform / Model | Per-Query Input Cost | Per-Query Output Cost | Total Cost Per Query |",
            "|---|---|---|---|",
        ])
        for m_id, m_vals in cost_data.get("per_query_breakdown", {}).get("commercial_apis", {}).items():
            report_lines.append(
                f"| {m_id} | ${m_vals['input_cost_usd']:.6f} | ${m_vals['output_cost_usd']:.6f} | ${m_vals['total_query_cost_usd']:.6f} |"
            )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Generated comprehensive report at {output_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate evaluation summary report.")
    parser.add_argument("--results-dir", default="results", help="Results directory")
    parser.add_argument("--out", default="docs/evaluation-report.md", help="Output markdown path")
    args = parser.parse_args()

    generate_report(Path(args.results_dir), Path(args.out))


if __name__ == "__main__":
    main()
