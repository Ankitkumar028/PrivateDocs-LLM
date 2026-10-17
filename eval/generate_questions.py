"""Generate high-quality evaluation questions using any OpenAI-compatible LLM endpoint."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import httpx

from privatedocs.ingestion.loaders import load_document

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("generate_questions")

QUESTION_GENERATION_PROMPT = """You are an expert financial and regulatory evaluation benchmark designer.
Given the following excerpt from a corporate SEC 10-K filing or regulatory document, generate {num_questions} high-quality evaluation questions with ground-truth answers and exact citations.

Include a mix of:
1. "factual": Direct factual lookups with specific numbers, dates, or named risks.
2. "synthesis": Requires understanding 2 or more facts in the excerpt.
3. "unanswerable": Plausible questions that seem related to the company/topic, but whose answer is CANNOT be determined from this excerpt. For unanswerable questions, the reference_answer MUST be: "I don't know based on the provided documents."

Output your response STRICTLY as a valid JSON array of objects with the following schema:
[
  {{
    "question": "Clear, specific question text",
    "reference_answer": "Accurate, concise ground truth answer (or 'I don't know based on the provided documents.')",
    "citation": "Exact citation tag, e.g. [{doc_id}, {section}] or [{doc_id}, p.{page}]",
    "type": "factual | synthesis | unanswerable"
  }}
]

Document ID: {doc_id}
Document Source: {source}
Section / Location: {section}
Document Excerpt:
{excerpt}
"""


def call_openai_compatible_endpoint(
    endpoint: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float = 0.2,
    timeout: float = 60.0,
) -> str:
    """Call an OpenAI-compatible /chat/completions endpoint."""
    url = f"{endpoint.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 1500,
    }

    with httpx.Client(timeout=timeout) as client:
        response = client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]


def generate_questions_for_doc(
    file_path: Path,
    endpoint: str,
    api_key: str,
    model: str,
    num_questions: int = 5,
) -> list[dict[str, Any]]:
    """Load document, select representative sections, and prompt LLM to draft questions."""
    docs = load_document(file_path)
    if not docs:
        logger.warning("No documents loaded from %s", file_path)
        return []

    # Pick 2-3 substantive sections (e.g. Item 1A, Item 7, Item 1, or middle pages)
    selected = [d for d in docs if len(d.content.strip()) > 300]
    if not selected:
        selected = docs

    # Pick up to 3 sections
    sample_docs = selected[:3]
    generated_items: list[dict[str, Any]] = []

    for d in sample_docs:
        excerpt = d.content[:3000].strip()
        loc_str = d.section if d.page == 0 else f"p.{d.page}"
        prompt = QUESTION_GENERATION_PROMPT.format(
            num_questions=max(2, num_questions // len(sample_docs)),
            doc_id=d.doc_id,
            source=d.source,
            section=loc_str,
            page=d.page,
            excerpt=excerpt,
        )

        messages = [
            {"role": "system", "content": "You are a professional benchmark test creator. Output only valid JSON."},
            {"role": "user", "content": prompt},
        ]

        try:
            content = call_openai_compatible_endpoint(
                endpoint=endpoint,
                api_key=api_key,
                model=model,
                messages=messages,
            )
            # Extract JSON array from response
            match = re.search(r"\[\s*\{.*\}\s*\]", content, re.DOTALL)
            if match:
                raw_json = match.group(0)
                items = json.loads(raw_json)
                for item in items:
                    item["doc_id"] = d.doc_id
                    item["source_file"] = d.source
                    item["source_section"] = d.section
                    item["source_page"] = d.page
                    item["human_verified"] = False
                    generated_items.append(item)
            else:
                logger.warning("Could not extract JSON array from LLM response for %s", d.doc_id)
        except Exception as exc:
            logger.error("Failed question generation for %s: %s", d.doc_id, exc)

    return generated_items


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate evaluation questions using OpenAI-compatible endpoint.")
    parser.add_argument("--doc", required=True, help="Path to input document or raw_docs directory")
    parser.add_argument(
        "--endpoint",
        default=os.getenv("OPENAI_BASE_URL", os.getenv("GEN_BASE_URL", "https://api.openai.com/v1")),
        help="OpenAI-compatible API base URL (e.g. https://api.openai.com/v1 or https://openrouter.ai/api/v1)",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("OPENAI_API_KEY", os.getenv("GEN_API_KEY", "")),
        help="API Key for the endpoint (or set OPENAI_API_KEY / GEN_API_KEY)",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("GEN_MODEL", "gpt-4o-mini"),
        help="Model identifier (e.g. gpt-4o, gpt-4o-mini, claude-3-5-haiku, etc.)",
    )
    parser.add_argument("--num-questions", type=int, default=5, help="Number of questions per document")
    parser.add_argument("--out", default="eval/datasets/generated_candidates.jsonl", help="Output JSONL file path")
    args = parser.parse_args()

    doc_path = Path(args.doc)
    if not doc_path.exists():
        raise FileNotFoundError(f"Document path does not exist: {doc_path}")

    files = [doc_path] if doc_path.is_file() else sorted(doc_path.glob("*.html")) + sorted(doc_path.glob("*.pdf"))

    if not args.api_key:
        print("\n[WARNING] No API key provided via --api-key, OPENAI_API_KEY, or GEN_API_KEY.")
        print("To generate questions with an external frontier model, supply an endpoint and key:")
        print("  python eval/generate_questions.py --doc data/raw_docs --endpoint https://api.openai.com/v1 --api-key <KEY> --model gpt-4o-mini\n")
        return

    out_file = Path(args.out)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    all_generated = []
    for f in files:
        logger.info("Generating questions for %s using %s at %s...", f.name, args.model, args.endpoint)
        items = generate_questions_for_doc(
            file_path=f,
            endpoint=args.endpoint,
            api_key=args.api_key,
            model=args.model,
            num_questions=args.num_questions,
        )
        all_generated.extend(items)

    with open(out_file, "a", encoding="utf-8") as f:
        for idx, item in enumerate(all_generated, start=1):
            if "id" not in item:
                item["id"] = f"gen_q_{idx:03d}"
            f.write(json.dumps(item) + "\n")

    print(f"\nSuccessfully generated {len(all_generated)} candidate questions saved to {out_file}!")


if __name__ == "__main__":
    main()
