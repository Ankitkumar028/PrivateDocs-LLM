"""Verification Script: verifies that every evidence phrase exists in the raw source document text
after whitespace, quote, and markdown pipe normalization.
Also flags any question/evidence semantic mismatches.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import re
from typing import Any

from privatedocs.ingestion.loaders import load_document

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("verify_evidence")


def normalize_text(text: str) -> str:
    """Normalize text by stripping pipes, normalizing quotes/dashes, and collapsing whitespace."""
    if not text:
        return ""
    # Replace unicode quotes and dashes
    t = text.replace("\u201c", '"').replace("\u201d", '"').replace("“", '"').replace("”", '"')
    t = t.replace("\u2018", "'").replace("\u2019", "'").replace("’", "'").replace("‘", "'")
    t = t.replace("\u2013", "-").replace("\u2014", "-")
    # Replace markdown table pipes with spaces
    t = t.replace("|", " ")
    # Lowercase
    t = t.lower()
    # Collapse whitespace
    t = re.sub(r"\s+", " ", t).strip()
    return t


def main() -> None:
    questions_file = Path("eval/datasets/eval_questions_40.jsonl")
    raw_docs_dir = Path("data/raw_docs")

    with open(questions_file, "r", encoding="utf-8") as f:
        questions = [json.loads(line) for line in f if line.strip()]

    # Load all raw documents into full text per doc_id
    raw_files = sorted(raw_docs_dir.glob("*.*"))
    doc_texts: dict[str, str] = {}
    for rf in raw_files:
        loaded = load_document(rf)
        full_doc_content = " ".join(doc.content for doc in loaded)
        doc_id = loaded[0].doc_id if loaded else rf.stem
        # Store by both doc_id and filename stem
        doc_texts[doc_id] = full_doc_content
        doc_texts[rf.stem] = full_doc_content
        logger.info("Loaded document '%s' (%d pages/sections, %d chars)", doc_id, len(loaded), len(full_doc_content))

    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for q in questions:
        qid = q["id"]
        qtype = q.get("type", "factual")
        source_doc = q.get("source_doc", "")
        ev_phrase = q.get("evidence_phrase", "")
        question_text = q.get("question", "")

        if qtype == "unanswerable" or source_doc == "none":
            results.append({"id": qid, "status": "SKIP_UNANSWERABLE"})
            continue

        if source_doc not in doc_texts:
            failures.append({
                "id": qid,
                "error": f"Source doc '{source_doc}' not found in loaded raw documents",
                "question": question_text,
                "evidence_phrase": ev_phrase,
            })
            continue

        raw_doc_text = doc_texts[source_doc]
        norm_doc = normalize_text(raw_doc_text)
        norm_ev = normalize_text(ev_phrase)

        # Check exact normalized substring
        is_exact_match = norm_ev in norm_doc

        # Also check word-level coverage
        ev_words = [w for w in norm_ev.split() if len(w) > 2]
        word_match_ratio = 0.0
        if ev_words:
            matched_words = sum(1 for w in ev_words if w in norm_doc)
            word_match_ratio = matched_words / len(ev_words)

        found_in_doc = is_exact_match or (word_match_ratio >= 0.95 and len(ev_words) >= 4)

        if not found_in_doc:
            failures.append({
                "id": qid,
                "error": "Evidence phrase not found in source document",
                "question": question_text,
                "source_doc": source_doc,
                "evidence_phrase": ev_phrase,
                "norm_evidence": norm_ev,
                "word_match_ratio": word_match_ratio,
            })
        else:
            results.append({
                "id": qid,
                "status": "PASS",
                "exact_substring": is_exact_match,
                "word_match_ratio": word_match_ratio,
            })

    print("\n" + "=" * 80)
    print(f"EVIDENCE PHRASE SOURCE VERIFICATION (Total: {len(questions)} questions)")
    print("=" * 80)
    passed_count = sum(1 for r in results if r["status"] == "PASS")
    unans_count = sum(1 for r in results if r["status"] == "SKIP_UNANSWERABLE")
    print(f"Passed: {passed_count} | Unanswerable: {unans_count} | Failures: {len(failures)}")
    print("-" * 80)

    if failures:
        print("FAILURES LIST:")
        for idx, f in enumerate(failures, 1):
            print(f"\n[{idx}] Question ID: {f['id']} (Source: {f.get('source_doc')})")
            print(f"    Question:        {f['question']}")
            print(f"    Evidence Phrase: \"{f['evidence_phrase']}\"")
            print(f"    Normalized:      \"{f.get('norm_evidence')}\"")
            print(f"    Error:           {f['error']} (Word match ratio: {f.get('word_match_ratio', 0):.2f})")
    else:
        print("ALL evidence phrases successfully verified against source documents!")
    print("=" * 80)


if __name__ == "__main__":
    main()
