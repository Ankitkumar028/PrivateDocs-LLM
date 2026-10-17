"""Live interactive verification script testing real documents, real embeddings, real Qdrant, and real Ollama."""

from __future__ import annotations

import sys
from pathlib import Path

import httpx

# Force UTF-8 stdout if supported
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_URL = "http://localhost:8000"
ADMIN_KEY = "admin-dev-secret-key-12345"
USER_KEY = "user-dev-secret-key-67890"

client = httpx.Client(timeout=60.0)

print("=" * 65)
print("15-MINUTE LIVE VERIFICATION: REAL PIPELINE END-TO-END")
print("=" * 65)

# 1. Ingest real documents into live system
print("\n[Step 1] Ingesting real documents into Docker Qdrant via /v1/ingest...")

sec_path = Path("data/sample_documents/sec_10k_sample.html")
with open(sec_path, "rb") as f:
    resp_sec = client.post(
        f"{BASE_URL}/v1/ingest",
        headers={"X-API-Key": ADMIN_KEY},
        files={"file": ("sec_10k_sample.html", f, "text/html")},
    )
    print("  -> Ingested SEC 10-K HTML:", resp_sec.status_code, resp_sec.json())

pdf_path = Path("data/sample_documents/confidential_memo.pdf")
with open(pdf_path, "rb") as f:
    resp_pdf = client.post(
        f"{BASE_URL}/v1/ingest",
        headers={"X-API-Key": ADMIN_KEY},
        files={"file": ("confidential_memo.pdf", f, "application/pdf")},
    )
    print("  -> Ingested Memo PDF:   ", resp_pdf.status_code, resp_pdf.json())

# 2. Inspect indexed documents
docs_resp = client.get(f"{BASE_URL}/v1/documents", headers={"X-API-Key": ADMIN_KEY})
print("\n[Step 2] Querying /v1/documents:")
for doc in docs_resp.json():
    print(f"  * Document: {doc['doc_id']} | Source: {doc['source']} | Chunks: {doc['chunk_count']}")

# 3. Ask hand-crafted questions
print("\n[Step 3] Asking hand-crafted questions against real Ollama model...")

test_queries = [
    {
        "title": "Hardware & Memory Specification",
        "question": "What accelerator hardware is used and what is the memory capacity per node?",
    },
    {
        "title": "Cost Ceiling & Reserve Budget",
        "question": "What is the hourly cost ceiling for GPU compute and how many hours are in reserve?",
    },
    {
        "title": "Data Privacy & Cloud Regulations",
        "question": "What do regulations mandate regarding proprietary company documents and public cloud providers?",
    },
    {
        "title": "Abstention Test (Unanswerable Question)",
        "question": "What is the CEO's personal home address and favorite vacation destination?",
    },
]

for idx, item in enumerate(test_queries, start=1):
    print("\n" + "-" * 65)
    print(f"QUERY {idx} [{item['title']}]:")
    print(f"Question: \"{item['question']}\"")

    resp = client.post(
        f"{BASE_URL}/v1/ask",
        headers={"X-API-Key": USER_KEY},
        json={"question": item["question"], "top_k": 2},
    )

    if resp.status_code != 200:
        print(f"  [ERROR] Status {resp.status_code}: {resp.text}")
        continue

    data = resp.json()
    print(f"\nResponse Latency: {data['latency_ms']:.1f}ms | Tokens Generated: {data['tokens']}")
    print(f"Retrieved Chunks ({len(data['sources'])}):")
    for s_idx, s in enumerate(data["sources"], start=1):
        clean_snippet = s["snippet"].replace("\n", " ").strip()
        print(f"  [{s_idx}] [{s['doc_id']}, p.{s['page']}] (RRF Score: {s['score']:.4f})")
        print(f"      \"{clean_snippet[:120]}...\"")

    print("\nLLM Answer (qwen2.5:0.5b via Ollama):")
    print(f"  {data['answer'].strip()}")

print("\n" + "=" * 65)
print("15-MINUTE LIVE VERIFICATION COMPLETE!")
print("=" * 65)
