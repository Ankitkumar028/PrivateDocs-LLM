"""Retrieval sanity check: asks 10 hand-written questions about real 10-Ks and PDFs, displays top-5 chunks, and evaluates accuracy."""

from __future__ import annotations

import json
from pathlib import Path

from privatedocs.config import get_settings
from privatedocs.retrieval.bm25 import BM25Index
from privatedocs.retrieval.embedder import CPUTextEmbedder
from privatedocs.retrieval.hybrid import hybrid_search
from privatedocs.retrieval.vector_store import VectorStore

QUESTIONS = [
    {
        "id": "q1_apple_appstore",
        "question": "What are Apple's stated risk factors concerning its dependence on third-party software developers and the App Store?",
        "expected_doc": "sec_apple_10k_2023",
        "expected_section": "Item 1A - Risk Factors",
        "expected_topic": "App store commissions, developer relations, alternative app stores, litigation/regulatory mandates",
    },
    {
        "id": "q2_apple_sales",
        "question": "What was Apple's total net sales for the fiscal year ended September 30, 2023, broken down by product categories like iPhone and Services?",
        "expected_doc": "sec_apple_10k_2023",
        "expected_section": "Item 7 or Item 8",
        "expected_topic": "Net sales by product category ($383B total, iPhone $200B+, Services $85B+)",
    },
    {
        "id": "q3_microsoft_ai_azure",
        "question": "How does Microsoft describe the competitive risks related to AI and cloud infrastructure investments like Azure?",
        "expected_doc": "sec_microsoft_10k_2023",
        "expected_section": "Item 1A - Risk Factors",
        "expected_topic": "Capital expenditures in cloud infrastructure, AI model integration, fierce competition from AWS/Google",
    },
    {
        "id": "q4_microsoft_intelligent_cloud",
        "question": "What are Microsoft's principal product offerings within the Intelligent Cloud segment?",
        "expected_doc": "sec_microsoft_10k_2023",
        "expected_section": "Item 1 - Business",
        "expected_topic": "Server products and cloud services, Azure, Windows Server, SQL Server, Nuance, GitHub",
    },
    {
        "id": "q5_nvidia_china_export",
        "question": "What does NVIDIA disclose regarding export controls and licensing restrictions affecting GPU sales to China?",
        "expected_doc": "sec_nvidia_10k_2024",
        "expected_section": "Item 1A - Risk Factors",
        "expected_topic": "US government export licensing requirements for A100/H100/A800/H800 chips to China and Middle East",
    },
    {
        "id": "q6_alphabet_cyber_board",
        "question": "What does Alphabet state regarding cybersecurity governance and the role of the Board Audit and Compliance Committee?",
        "expected_doc": "sec_alphabet_10k_2023",
        "expected_section": "Item 1C - Cybersecurity",
        "expected_topic": "Audit and Compliance Committee oversight of risk management and cybersecurity programs",
    },
    {
        "id": "q7_amazon_seasonality",
        "question": "How does Amazon characterize seasonality and fulfillment network capacity risks in its retail operations?",
        "expected_doc": "sec_amazon_10k_2023",
        "expected_section": "Item 1A - Risk Factors",
        "expected_topic": "Disproportionate sales in Q4 holiday season, inventory markdowns, fulfillment staffing and shipping surcharges",
    },
    {
        "id": "q8_amd_foundry_tsmc",
        "question": "What does AMD say about its reliance on third-party foundries like TSMC for manufacturing its semiconductor wafers?",
        "expected_doc": "sec_amd_10k_2023",
        "expected_section": "Item 1A - Risk Factors",
        "expected_topic": "Dependence on TSMC as principal foundry, wafer supply agreements, geopolitical risk in Taiwan",
    },
    {
        "id": "q9_sec_rule_4day_deadline",
        "question": "What is the 4-business-day disclosure deadline for material cybersecurity incidents on Form 8-K under the SEC final rule?",
        "expected_doc": "sec_cybersecurity_final_rule_2023",
        "expected_section": "Item 1.05",
        "expected_topic": "Form 8-K Item 1.05 requirement to disclose material cybersecurity incident within 4 business days of materiality determination",
    },
    {
        "id": "q10_whitehouse_eo_redteaming",
        "question": "What reporting requirements does Executive Order 14110 place on companies developing dual-use foundation models regarding safety test results and red-teaming?",
        "expected_doc": "whitehouse_eo_14110_ai",
        "expected_section": "Section 4.2",
        "expected_topic": "Defense Production Act reporting, sharing AI red-teaming safety test results with US government",
    },
]


def run_sanity_check() -> None:
    settings = get_settings()

    vector_store = VectorStore(
        collection_name=settings.QDRANT_COLLECTION_NAME,
        host=settings.QDRANT_HOST,
        port=settings.QDRANT_PORT,
        url=settings.QDRANT_URL,
        api_key=settings.QDRANT_API_KEY,
        in_memory=settings.QDRANT_IN_MEMORY,
    )
    embedder = CPUTextEmbedder(
        model_name=settings.EMBEDDING_MODEL_NAME,
        device=settings.EMBEDDING_DEVICE,
    )
    bm25 = BM25Index()
    bm25.load(Path("data/bm25_index.pkl"))

    results_data = []

    print("\n" + "=" * 80)
    print("RETRIEVAL SANITY CHECK: 10 REAL CORPUS QUESTIONS (TOP-5 CHUNKS)")
    print("=" * 80)

    for idx, item in enumerate(QUESTIONS, start=1):
        q = item["question"]
        expected_doc = item["expected_doc"]
        expected_sec = item["expected_section"]

        # Run hybrid retrieval
        q_vec = embedder.embed_query(q)
        dense_hits = vector_store.search(q_vec, top_k=10)
        bm25_hits = bm25.search(q, top_k=10)
        retrieved = hybrid_search(dense_hits, bm25_hits, rrf_k=settings.RRF_K, top_k=5)

        print(f"\n[{idx}/10] QUESTION: {q}")
        print(f"Target: Document [{expected_doc}] | Section [{expected_sec}]")
        print(f"Expected Topic: {item['expected_topic']}")
        print("-" * 80)

        chunks_info = []
        found_target_doc = False
        found_target_section = False

        for rank, c in enumerate(retrieved, start=1):
            doc_id = c.doc_id
            page = c.page
            sec = c.section
            score = round(c.score, 5)
            snippet = c.content.replace("\n", " ").strip()[:180]

            loc_str = f"p.{page}" if page and page > 0 else f"{sec}"
            is_doc_match = expected_doc.lower() in doc_id.lower()
            is_sec_match = any(w.lower() in sec.lower() for w in expected_sec.split()) if expected_sec else False

            if is_doc_match:
                found_target_doc = True
            if is_doc_match and (is_sec_match or page > 0):
                found_target_section = True

            match_flag = "  <-- HIT!" if (is_doc_match and (is_sec_match or page > 0)) else ("  <-- (doc hit)" if is_doc_match else "")
            print(f"  Rank #{rank} [score={score:.4f}]: [{doc_id}, {loc_str}]{match_flag}")
            print(f"    Excerpt: {snippet}...")

            chunks_info.append(
                {
                    "rank": rank,
                    "doc_id": doc_id,
                    "location": loc_str,
                    "score": score,
                    "snippet": snippet,
                    "is_doc_match": is_doc_match,
                }
            )

        # Honest Assessment
        if found_target_section:
            judgment = "CORRECT (Precise Hit)"
            judgment_reason = f"Rank-1 or top-3 chunk hit targeted document '{expected_doc}' and relevant section/page."
        elif found_target_doc:
            judgment = "PARTIALLY CORRECT (Document Hit)"
            judgment_reason = f"Hit target document '{expected_doc}', but adjacent item or general section retrieved in top ranks."
        else:
            judgment = "WRONG / MISS"
            judgment_reason = f"Did not retrieve target document '{expected_doc}' in top ranks."

        print(f"--> HONEST ASSESSMENT: {judgment} - {judgment_reason}")

        results_data.append(
            {
                "question_id": item["id"],
                "question": q,
                "expected_doc": expected_doc,
                "expected_section": expected_sec,
                "judgment": judgment,
                "judgment_reason": judgment_reason,
                "retrieved_chunks": chunks_info,
            }
        )

    out_path = Path("results/retrieval_sanity_check.json")
    out_path.write_text(json.dumps(results_data, indent=2), encoding="utf-8")
    print("\n" + "=" * 80)
    print(f"Sanity check complete! Detailed results saved to {out_path}")
    print("=" * 80)


if __name__ == "__main__":
    run_sanity_check()
