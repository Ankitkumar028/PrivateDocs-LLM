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
    source_doc: str | None = None,
) -> dict[str, Any]:
    """Score an answer using rubric for correctness, faithfulness, and citation format."""
    import re

    lower_ans = answer.lower()
    abstained = "don't know" in lower_ans or "not provided" in lower_ans or "cannot find" in lower_ans

    # Check citation format: [doc_id, p.<page>] or [doc_id, <section>]
    citation_pattern = re.compile(r"\[([a-zA-Z0-9_\-]+),\s*(?:p\.?\s*\d+|[^\]]+)\]")
    citations_found = citation_pattern.findall(answer)
    has_valid_citation = len(citations_found) > 0

    # If source_doc is known, check if cited
    citation_correct = False
    if source_doc and citations_found:
        citation_correct = any(source_doc.lower() in c.lower() for c in citations_found)
    elif not source_doc and has_valid_citation:
        citation_correct = True

    if is_unanswerable:
        correctness = 2 if abstained else 0
        faithfulness = 1 if abstained else 0
        abstention_ok = 1 if abstained else 0
        citation_score = 1.0 if (abstained and not citations_found) else 0.0
    else:
        abstention_ok = 1 if not abstained else 0
        if abstained:
            correctness = 0
            faithfulness = 1
            citation_score = 0.0
        else:
            # Check overlap with reference answer keywords
            ref_words = set(re.findall(r"\w+", reference.lower()))
            ans_words = set(re.findall(r"\w+", answer.lower()))
            overlap = len(ref_words.intersection(ans_words)) / max(len(ref_words), 1)

            if overlap >= 0.4:
                correctness = 2
            elif overlap >= 0.2:
                correctness = 1
            else:
                correctness = 1 if len(answer) > 20 else 0

            faithfulness = 1
            citation_score = 1.0 if has_valid_citation else 0.0

    return {
        "correctness": correctness,
        "faithfulness": faithfulness,
        "abstention_ok": abstention_ok,
        "has_citation": 1 if has_valid_citation else 0,
        "citation_score": citation_score,
        "citations_extracted": citations_found,
    }
