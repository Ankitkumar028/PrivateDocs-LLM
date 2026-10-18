"""Interactive & batch review tool for evaluation questions.

Displays for each question:
- Question ID, Type, Document
- Question text
- Reference answer
- Evidence phrase
- Surrounding chunk context with evidence phrase highlighted
- Allows approving ([a]), rejecting ([r]), skipping ([s]), or quitting ([q]).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import pickle
import sys

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("review_questions")


def load_chunks() -> dict[str, Any]:
    """Load chunk lookup from data/bm25_index.pkl."""
    bm25_path = Path("data/bm25_index.pkl")
    if not bm25_path.exists():
        return {}
    with open(bm25_path, "rb") as f:
        chunks = pickle.load(f)["chunks"]
    return {c.chunk_id: c for c in chunks}


def get_surrounding_context(chunk_content: str, evidence_phrase: str, window: int = 250) -> str:
    """Extract context around the evidence phrase with visual highlighting."""
    if not evidence_phrase:
        return chunk_content[:500] + ("..." if len(chunk_content) > 500 else "")

    idx = chunk_content.lower().find(evidence_phrase.lower())
    if idx == -1:
        return "[Evidence phrase not found in chunk content]\n" + chunk_content[:300]

    start = max(0, idx - window)
    end = min(len(chunk_content), idx + len(evidence_phrase) + window)

    before = chunk_content[start:idx]
    match = chunk_content[idx : idx + len(evidence_phrase)]
    after = chunk_content[idx + len(evidence_phrase) : end]

    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(chunk_content) else ""

    return f"{prefix}{before} >>> {match} <<< {after}{suffix}"


def review_questions(
    questions_file: Path,
    interactive: bool = True,
    target_id: str | None = None,
) -> None:
    """Display questions and allow interactive human review."""
    with open(questions_file, encoding="utf-8") as f:
        questions = [json.loads(line) for line in f if line.strip()]

    chunk_map = load_chunks()
    total = len(questions)
    reviewed_count = sum(1 for q in questions if q.get("human_reviewed"))

    print("\n" + "=" * 75)
    print(f"QUESTION EVALUATION REVIEW TOOL ({reviewed_count}/{total} Reviewed)")
    print("=" * 75)

    modified = False

    for idx, q in enumerate(questions, start=1):
        q_id = q.get("id")
        if target_id and q_id != target_id:
            continue

        q_type = q.get("type", "factual")
        q_text = q.get("question")
        ref_ans = q.get("reference_answer")
        ev_phrase = q.get("evidence_phrase", "")
        doc_id = q.get("source_doc")
        chunk_id = q.get("target_chunk_id")
        is_reviewed = q.get("human_reviewed", False)

        chunk = chunk_map.get(chunk_id) if chunk_id else None
        context = (
            get_surrounding_context(chunk.content, ev_phrase)
            if chunk
            else ("[Unanswerable question - no source chunk]" if q_type == "unanswerable" else "[Chunk not found]")
        )

        status_tag = "[APPROVED]" if is_reviewed else "[PENDING REVIEW]"

        print(f"\n--- [{idx}/{total}] Question ID: {q_id} {status_tag} ---")
        print(f"Type:             {q_type}")
        print(f"Document:         {doc_id}")
        print(f"Question:         {q_text}")
        print(f"Reference Answer: {ref_ans}")
        print(f"Evidence Phrase:  '{ev_phrase}'" if ev_phrase else "Evidence Phrase:  N/A")
        print("Surrounding Source Text:")
        print("-" * 75)
        print(context.strip())
        print("-" * 75)

        if interactive and sys.stdin.isatty():
            while True:
                prompt = f"Action for {q_id} ([a]pprove / [r]eject / [s]kip / [q]uit): "
                choice = input(prompt).strip().lower()
                if choice in ("a", "approve"):
                    q["human_reviewed"] = True
                    modified = True
                    print(f"-> Marked {q_id} as APPROVED.")
                    break
                elif choice in ("r", "reject"):
                    q["human_reviewed"] = False
                    q["status"] = "rejected"
                    modified = True
                    print(f"-> Marked {q_id} as REJECTED.")
                    break
                elif choice in ("s", "skip"):
                    print("-> Skipped.")
                    break
                elif choice in ("q", "quit"):
                    print("\nExiting review...")
                    if modified:
                        with open(questions_file, "w", encoding="utf-8") as f:
                            for item in questions:
                                f.write(json.dumps(item) + "\n")
                        print(f"Saved review progress to {questions_file}")
                    return
                else:
                    print("Invalid choice. Please enter 'a', 'r', 's', or 'q'.")

    if modified:
        with open(questions_file, "w", encoding="utf-8") as f:
            for item in questions:
                f.write(json.dumps(item) + "\n")
        print(f"\nSaved updated reviews to {questions_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Review and approve evaluation questions.")
    parser.add_argument("--file", default="eval/datasets/eval_questions_40.jsonl", help="Questions JSONL")
    parser.add_argument("--non-interactive", action="store_true", help="Print all questions without prompting")
    parser.add_argument("--id", default=None, help="Inspect specific question ID")
    args = parser.parse_args()

    review_questions(
        questions_file=Path(args.file),
        interactive=not args.non_interactive,
        target_id=args.id,
    )


if __name__ == "__main__":
    main()
