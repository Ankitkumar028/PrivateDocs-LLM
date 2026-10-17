"""LLM-assisted question generator for human review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def generate_candidate_questions(doc_path: Path, output_file: Path) -> None:
    """Generate synthetic question-answer pairs for human review."""
    print(f"Scanning document: {doc_path}")
    # Template candidate pairs for bootstrapping
    sample_candidates = [
        {
            "id": f"gen_{doc_path.stem}_01",
            "question": f"What is the main finding or requirement outlined in {doc_path.name}?",
            "reference_answer": "Summary extracted during document review.",
            "source_doc": doc_path.stem,
            "source_page": 1,
            "type": "factual",
            "human_verified": False,
        }
    ]

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "a", encoding="utf-8") as f:
        for item in sample_candidates:
            f.write(json.dumps(item) + "\n")
    print(f"Appended candidate questions to {output_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate candidate questions for curation.")
    parser.add_argument("--doc", required=True, help="Path to input document")
    parser.add_argument("--out", default="eval/datasets/candidates.jsonl", help="Output file")
    args = parser.parse_args()

    generate_candidate_questions(Path(args.doc), Path(args.out))


if __name__ == "__main__":
    main()
