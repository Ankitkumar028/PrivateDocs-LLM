"""End-to-end answer evaluation script testing real responses from /v1/ask."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

import httpx
import sys
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from eval.judge import score_answer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eval_answers")


def run_evaluation(
    questions_path: Path,
    api_url: str = "http://localhost:8000",
    api_key: str = "user-secret-dev-key-change-in-prod",
    output_path: Path | None = None,
    top_k: int = 3,
) -> dict[str, Any]:
    """Send test questions to PrivateDocs API and score responses."""
    with open(questions_path, encoding="utf-8") as f:
        questions = [json.loads(line) for line in f if line.strip()]

    if not questions:
        logger.warning("No questions found in %s", questions_path)
        return {}

    ask_url = f"{api_url.rstrip('/')}/v1/ask"
    headers = {"X-API-Key": api_key, "Content-Type": "application/json"}

    eval_records = []
    logger.info("Evaluating %d questions against %s...", len(questions), ask_url)

    with httpx.Client(timeout=90.0) as client:
        for idx, q in enumerate(questions, start=1):
            q_text = q["question"]
            ref_ans = q.get("reference_answer", "")
            is_unans = q.get("type") == "unanswerable"
            src_doc = q.get("source_doc") or q.get("doc_id")

            logger.info("[%d/%d] Q: %s", idx, len(questions), q_text[:60])
            try:
                resp = client.post(
                    ask_url,
                    headers=headers,
                    json={"question": q_text, "top_k": top_k},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    model_answer = data.get("answer", "")
                    prompt_toks = data.get("prompt_tokens", 0)
                    comp_toks = data.get("completion_tokens", 0)
                    citations = data.get("citations", [])
                    context_chunks = data.get("retrieved_chunks", [])
                    raw_context = "\n".join(c.get("content", "") for c in context_chunks)
                else:
                    logger.warning("API error %d: %s", resp.status_code, resp.text[:100])
                    model_answer = "API Error"
                    prompt_toks = 0
                    comp_toks = 0
                    citations = []
                    raw_context = ""
            except Exception as exc:
                logger.error("Failed API call for question %d: %s", idx, exc)
                model_answer = f"Connection Failed: {exc}"
                prompt_toks = 0
                comp_toks = 0
                citations = []
                raw_context = ""

            score = score_answer(
                question=q_text,
                reference=ref_ans,
                context=raw_context,
                answer=model_answer,
                is_unanswerable=is_unans,
                source_doc=src_doc,
            )

            record = {
                "id": q.get("id", f"q_{idx}"),
                "question": q_text,
                "type": q.get("type", "factual"),
                "reference_answer": ref_ans,
                "model_answer": model_answer,
                "prompt_tokens": prompt_toks,
                "completion_tokens": comp_toks,
                "citations_returned": citations,
                **score,
            }
            eval_records.append(record)

    total = len(eval_records)
    avg_correctness = round(sum(r["correctness"] for r in eval_records) / total, 3) if total > 0 else 0
    avg_faithfulness = round(sum(r["faithfulness"] for r in eval_records) / total, 3) if total > 0 else 0
    abstention_rate = round(sum(r["abstention_ok"] for r in eval_records) / total, 3) if total > 0 else 0
    citation_precision = round(sum(r["citation_score"] for r in eval_records) / total, 3) if total > 0 else 0

    results = {
        "total_questions_evaluated": total,
        "mean_correctness_0_to_2": avg_correctness,
        "faithfulness_rate": avg_faithfulness,
        "abstention_accuracy": abstention_rate,
        "citation_precision": citation_precision,
        "evaluation_records": eval_records,
    }

    print("\n" + "=" * 55)
    print("END-TO-END ANSWER EVALUATION RESULTS")
    print("=" * 55)
    print(f"Total Questions Evaluated: {total}")
    print(f"Mean Correctness (0-2):    {avg_correctness} / 2.0")
    print(f"Faithfulness Rate:         {avg_faithfulness:.1%}")
    print(f"Abstention Accuracy:       {abstention_rate:.1%}")
    print(f"Citation Precision:        {citation_precision:.1%}")
    print("=" * 55)

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Saved answer evaluation results to {out_p}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run end-to-end answer evaluation.")
    parser.add_argument("--questions", default="eval/datasets/sample_questions.jsonl", help="Input questions JSONL")
    parser.add_argument("--url", default=os.getenv("API_URL", "http://localhost:8000"), help="API Gateway URL")
    parser.add_argument("--key", default=os.getenv("API_KEY", "user-secret-dev-key-change-in-prod"), help="API Key")
    parser.add_argument("--out", default="results/answer_eval_metrics.json", help="Output path")
    parser.add_argument("--top-k", type=int, default=3, help="Top-K context chunks")
    args = parser.parse_args()

    run_evaluation(
        questions_path=Path(args.questions),
        api_url=args.url,
        api_key=args.key,
        output_path=Path(args.out),
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
