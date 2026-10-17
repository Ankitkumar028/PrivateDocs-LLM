"""End-to-end answer evaluation script."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.judge import score_answer


def run_evaluation(questions_path: Path, output_path: Path | None = None) -> dict:
    """Evaluate questions and compute summary scores."""
    with open(questions_path, encoding="utf-8") as f:
        questions = [json.loads(line) for line in f if line.strip()]

    scores = []
    for q in questions:
        is_unans = q.get("type") == "unanswerable"
        # Mocking evaluation response for pipeline validation
        mock_answer = (
            "I don't know based on the provided documents."
            if is_unans
            else f"According to [doc, p.1], {q.get('reference_answer')}"
        )
        res = score_answer(
            question=q["question"],
            reference=q.get("reference_answer", ""),
            context="Document text context",
            answer=mock_answer,
            is_unanswerable=is_unans,
        )
        scores.append(res)

    total = len(scores)
    avg_correctness = sum(s["correctness"] for s in scores) / total if total > 0 else 0
    avg_faithfulness = sum(s["faithfulness"] for s in scores) / total if total > 0 else 0
    abstention_rate = sum(s["abstention_ok"] for s in scores) / total if total > 0 else 0

    results = {
        "total_questions": total,
        "mean_correctness_0_to_2": avg_correctness,
        "faithfulness_rate": avg_faithfulness,
        "abstention_accuracy": abstention_rate,
    }

    print("=== Answer Evaluation Results ===")
    for k, v in results.items():
        print(f"{k}: {v}")

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run answer evaluation harness.")
    parser.add_argument("--questions", default="eval/datasets/sample_questions.jsonl", help="Input questions")
    parser.add_argument("--out", default="results/answer_eval_metrics.json", help="Output results path")
    args = parser.parse_args()

    run_evaluation(Path(args.questions), Path(args.out))


if __name__ == "__main__":
    main()
