"""Aggregates results into tables and summaries for docs and reporting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def generate_report(results_dir: Path, output_file: Path) -> None:
    """Read JSON metrics from results_dir and compile markdown summary."""
    retrieval_file = results_dir / "retrieval_metrics.json"
    answer_file = results_dir / "answer_eval_metrics.json"

    retrieval_data = {}
    if retrieval_file.exists():
        retrieval_data = json.loads(retrieval_file.read_text(encoding="utf-8"))

    answer_data = {}
    if answer_file.exists():
        answer_data = json.loads(answer_file.read_text(encoding="utf-8"))

    report_lines = [
        "# Evaluation & Benchmark Summary Report",
        "",
        "## 1. Retrieval Metrics",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Recall@3 | {retrieval_data.get('recall_at_3', 'N/A')} |",
        f"| Recall@5 | {retrieval_data.get('recall_at_5', 'N/A')} |",
        f"| Recall@10 | {retrieval_data.get('recall_at_10', 'N/A')} |",
        f"| MRR | {retrieval_data.get('mrr', 'N/A')} |",
        "",
        "## 2. Answer Quality & Faithfulness",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Total Evaluated | {answer_data.get('total_questions', 'N/A')} |",
        f"| Mean Correctness (0-2) | {answer_data.get('mean_correctness_0_to_2', 'N/A')} |",
        f"| Faithfulness Rate | {answer_data.get('faithfulness_rate', 'N/A')} |",
        f"| Abstention Accuracy | {answer_data.get('abstention_accuracy', 'N/A')} |",
        "",
    ]

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Generated report at {output_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate evaluation summary report.")
    parser.add_argument("--results-dir", default="results", help="Results directory")
    parser.add_argument("--out", default="docs/evaluation-report.md", help="Output markdown path")
    args = parser.parse_args()

    generate_report(Path(args.results_dir), Path(args.out))


if __name__ == "__main__":
    main()
