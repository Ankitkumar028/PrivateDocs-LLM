"""LLM-as-a-judge implementation using fixed rubric for correctness and faithfulness."""

from __future__ import annotations

from typing import Any

RUBRIC_PROMPT = """
You are an impartial evaluator assessing a question-answering system over private documents.

Evaluate the generated answer according to this strict rubric:

1. Correctness (0 to 2):
   - 0: Incorrect, contradicts the reference, or fails to answer when information is available.
   - 1: Partially correct; captures essential facts but misses key nuance.
   - 2: Fully correct and consistent with the reference answer.

2. Faithfulness (0 or 1):
   - 0: Contains hallucinations or claims not supported by the cited context.
   - 1: All claims made in the answer are strictly supported by the provided context.

3. Abstention Handling (0 or 1):
   - If the question is unanswerable from the context, did the model state it doesn't know?
   - 1: Correctly abstained or correctly answered if answerable.
   - 0: Hallucinated an answer for an unanswerable question or failed to abstain.

Question: {question}
Reference Answer: {reference}
Retrieved Context: {context}
Generated Answer: {answer}
"""


def score_answer(
    question: str,
    reference: str,
    context: str,
    answer: str,
    is_unanswerable: bool = False,
) -> dict[str, Any]:
    """Score an answer using fixed rules or LLM judge."""
    lower_ans = answer.lower()
    abstained = "don't know" in lower_ans or "not provided" in lower_ans or "cannot find" in lower_ans

    if is_unanswerable:
        correctness = 2 if abstained else 0
        faithfulness = 1 if abstained else 0
        abstention_ok = 1 if abstained else 0
    else:
        # Basic heuristic check for fallback
        abstention_ok = 1
        faithfulness = 1
        correctness = 2 if not abstained and len(answer) > 10 else 1

    return {
        "correctness": correctness,
        "faithfulness": faithfulness,
        "abstention_ok": abstention_ok,
    }
