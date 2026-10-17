"""End-to-end answer evaluation script testing real responses from /v1/ask.

Decouples answer generation from judging:
1. Queries /v1/ask for all questions.
2. Saves every raw response immediately to results/raw_answers.jsonl.
3. Invokes the configurable judge to score answers and save to results/judged_answers.jsonl.
4. Exports a stratified 20% sample to results/manual_review_sample_20pct.jsonl for human audit.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any

import httpx

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from eval.judge import judge_saved_answers
from privatedocs.config import get_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eval_answers")


def generate_and_save_answers(
    questions_path: Path,
    raw_answers_path: Path,
    api_url: str = "http://localhost:8000",
    api_key: str | None = None,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """Query /v1/ask for all questions and persist raw outputs to JSONL."""
    if api_key is None:
        api_key = get_settings().USER_API_KEY
    with open(questions_path, encoding="utf-8") as f:
        questions = [json.loads(line) for line in f if line.strip()]

    if not questions:
        logger.warning("No questions found in %s", questions_path)
        return []

    ask_url = f"{api_url.rstrip('/')}/v1/ask"
    headers = {"X-API-Key": api_key, "Content-Type": "application/json"}

    raw_records = []
    logger.info("Generating answers for %d questions against %s...", len(questions), ask_url)

    with httpx.Client(timeout=90.0) as client:
        for idx, q in enumerate(questions, start=1):
            q_text = q["question"]
            ref_ans = q.get("reference_answer", "")
            q_type = q.get("type", "factual")
            src_doc = q.get("source_doc")
            ev_phrase = q.get("evidence_phrase", "")

            logger.info("[%d/%d] Q (%s): %s", idx, len(questions), q_type, q_text[:65])
            t0 = time.perf_counter()
            try:
                resp = client.post(
                    ask_url,
                    headers=headers,
                    json={"question": q_text, "top_k": top_k},
                )
                duration_ms = (time.perf_counter() - t0) * 1000.0

                if resp.status_code == 200:
                    data = resp.json()
                    model_answer = data.get("answer", "")
                    prompt_toks = data.get("prompt_tokens", 0)
                    comp_toks = data.get("completion_tokens", 0)
                    total_toks = data.get("total_tokens", data.get("tokens", 0))
                    citations = data.get("sources", [])
                    retrieved_chunks = [
                        {"content": c.get("snippet", ""), "doc_id": c.get("doc_id", ""), "page": c.get("page", 0)}
                        for c in citations
                    ]
                else:
                    logger.warning("API error %d: %s", resp.status_code, resp.text[:100])
                    model_answer = f"API Error: HTTP {resp.status_code}"
                    prompt_toks = 0
                    comp_toks = 0
                    total_toks = 0
                    citations = []
                    retrieved_chunks = []
            except Exception as exc:
                logger.error("Failed API call for question %d: %s", idx, exc)
                duration_ms = (time.perf_counter() - t0) * 1000.0
                model_answer = f"Connection Failed: {exc}"
                prompt_toks = 0
                comp_toks = 0
                total_toks = 0
                citations = []
                retrieved_chunks = []

            record = {
                "id": q.get("id", f"q_{idx}"),
                "question": q_text,
                "type": q_type,
                "source_doc": src_doc,
                "evidence_phrase": ev_phrase,
                "reference_answer": ref_ans,
                "model_answer": model_answer,
                "latency_ms": round(duration_ms, 1),
                "prompt_tokens": prompt_toks,
                "completion_tokens": comp_toks,
                "total_tokens": total_toks,
                "retrieved_chunks": retrieved_chunks,
                "citations_returned": citations,
                "human_reviewed": q.get("human_reviewed", True),
            }
            raw_records.append(record)

    raw_answers_path.parent.mkdir(parents=True, exist_ok=True)
    with open(raw_answers_path, "w", encoding="utf-8") as f:
        for r in raw_records:
            f.write(json.dumps(r) + "\n")
    logger.info("Saved %d raw answers to %s", len(raw_records), raw_answers_path)
    return raw_records


def run_full_evaluation(
    questions_path: Path,
    api_url: str = "http://localhost:8000",
    api_key: str = "user-secret-dev-key-change-in-prod",
    raw_answers_path: Path = Path("results/raw_answers.jsonl"),
    judged_answers_path: Path = Path("results/judged_answers.jsonl"),
    manual_sample_path: Path = Path("results/manual_review_sample_20pct.jsonl"),
    output_metrics_path: Path = Path("results/answer_eval_metrics.json"),
    top_k: int = 3,
    skip_generation: bool = False,
    judge_model: str = "deterministic",
    judge_base_url: str | None = None,
    judge_api_key: str | None = None,
) -> dict[str, Any]:
    """Run full answer evaluation with persistent JSONL artifacts."""
    if not skip_generation or not raw_answers_path.exists():
        generate_and_save_answers(
            questions_path=questions_path,
            raw_answers_path=raw_answers_path,
            api_url=api_url,
            api_key=api_key,
            top_k=top_k,
        )
    else:
        logger.info("Skipping generation: using existing answers from %s", raw_answers_path)

    metrics = judge_saved_answers(
        raw_answers_path=raw_answers_path,
        judged_output_path=judged_answers_path,
        sample_review_path=manual_sample_path,
        judge_model=judge_model,
        judge_base_url=judge_base_url,
        judge_api_key=judge_api_key,
        sample_pct=0.20,
    )

    output_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "=" * 55)
    print("END-TO-END ANSWER EVALUATION RESULTS")
    print("=" * 55)
    print(f"Total Questions Evaluated: {metrics['total_evaluated']}")
    print(f"Judge Method / Model:      {metrics['judge_method']}")
    print(f"Mean Correctness (0-2):    {metrics['mean_correctness_0_to_2']} / 2.0")
    print(f"Faithfulness Rate:         {metrics['faithfulness_rate']:.1%}")
    print(f"Abstention Accuracy:       {metrics['abstention_accuracy']:.1%}")
    print(f"Citation Precision:        {metrics['citation_precision']:.1%}")
    print("=" * 55)
    print(f"Saved raw answers:         {raw_answers_path}")
    print(f"Saved judged scores:       {judged_answers_path}")
    print(f"Saved 20% manual sample:   {manual_sample_path}")
    print(f"Saved metrics summary:     {output_metrics_path}")

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run end-to-end answer evaluation and judging.")
    parser.add_argument("--questions", default="eval/datasets/eval_questions_40.jsonl", help="Questions JSONL")
    parser.add_argument("--url", default=os.getenv("API_URL", "http://localhost:8000"), help="API Gateway URL")
    parser.add_argument("--key", default=os.getenv("API_KEY", get_settings().USER_API_KEY), help="API Key")
    parser.add_argument("--raw-out", default="results/raw_answers.jsonl", help="Path to raw answers JSONL")
    parser.add_argument("--judged-out", default="results/judged_answers.jsonl", help="Path to judged JSONL")
    parser.add_argument("--sample-out", default="results/manual_review_sample_20pct.jsonl", help="20 percent manual review sample")
    parser.add_argument("--metrics-out", default="results/answer_eval_metrics.json", help="Metrics summary JSON")
    parser.add_argument("--skip-generation", action="store_true", help="Re-judge existing raw answers")
    parser.add_argument("--top-k", type=int, default=3, help="Top-K context chunks")
    parser.add_argument("--judge-model", default=os.getenv("JUDGE_MODEL", "deterministic"), help="Judge model or method")
    parser.add_argument("--judge-url", default=os.getenv("JUDGE_BASE_URL", "https://api.openai.com/v1"), help="Judge API base URL")
    parser.add_argument("--judge-key", default=os.getenv("JUDGE_API_KEY"), help="Judge API Key")
    args = parser.parse_args()

    run_full_evaluation(
        questions_path=Path(args.questions),
        api_url=args.url,
        api_key=args.key,
        raw_answers_path=Path(args.raw_out),
        judged_answers_path=Path(args.judged_out),
        manual_sample_path=Path(args.sample_out),
        output_metrics_path=Path(args.metrics_out),
        top_k=args.top_k,
        skip_generation=args.skip_generation,
        judge_model=args.judge_model,
        judge_base_url=args.judge_url,
        judge_api_key=args.judge_key,
    )


if __name__ == "__main__":
    main()
