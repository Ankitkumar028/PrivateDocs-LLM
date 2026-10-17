"""Retrieval evaluation harness: computes recall@k (k=1, 3, 5, 10) and MRR on real ingested corpus."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from privatedocs.config import get_settings
from privatedocs.retrieval.bm25 import BM25Index
from privatedocs.retrieval.embedder import CPUTextEmbedder
from privatedocs.retrieval.hybrid import hybrid_search
from privatedocs.retrieval.vector_store import VectorStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eval_retrieval")


def evaluate_retrieval(
    questions_file: Path,
    output_file: Path | None = None,
    top_k: int = 10,
) -> dict:
    """Evaluate retrieval metrics (recall@k, MRR) using live hybrid retriever."""
    with open(questions_file, encoding="utf-8") as f:
        questions = [json.loads(line) for line in f if line.strip()]

    # Filter answerable questions for retrieval eval
    answerable = [q for q in questions if q.get("type") != "unanswerable"]
    total = len(answerable)
    if total == 0:
        logger.warning("No answerable questions found in %s", questions_file)
        return {}

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
    bm25_cache = Path("data/bm25_index.pkl")
    if bm25_cache.exists():
        bm25.load(bm25_cache)
    else:
        logger.warning("BM25 index cache not found at %s!", bm25_cache)

    from privatedocs.retrieval.hybrid import hybrid_search

    hits_at_1 = 0
    hits_at_3 = 0
    hits_at_5 = 0
    hits_at_10 = 0
    reciprocal_ranks = []
    per_question_results = []

    for q in answerable:
        query_text = q["question"]
        target_doc = (q.get("source_doc") or q.get("doc_id") or "").lower()
        target_sec = (q.get("source_section") or q.get("section") or "").lower()
        target_page = q.get("source_page") or q.get("page")

        # Execute hybrid search
        q_vec = embedder.embed_query(query_text)
        dense_hits = vector_store.search(q_vec, top_k=top_k * 2)
        bm25_hits = bm25.search(query_text, top_k=top_k * 2)
        results = hybrid_search(dense_hits, bm25_hits, rrf_k=settings.RRF_K, top_k=top_k)

        matched_rank = None
        for rank, chunk in enumerate(results, start=1):
            chunk_doc = chunk.doc_id.lower()
            chunk_sec = chunk.section.lower()
            chunk_page = chunk.page

            # Match criteria: primary match on document
            doc_match = target_doc in chunk_doc or chunk_doc in target_doc
            sec_match = target_sec and (target_sec in chunk_sec or chunk_sec in target_sec)
            page_match = target_page and chunk_page == target_page

            if doc_match:
                if target_sec and sec_match:
                    matched_rank = rank
                    break
                elif not target_sec:
                    matched_rank = rank
                    break
                elif not matched_rank:
                    matched_rank = rank

        if matched_rank is not None:
            rr = 1.0 / matched_rank
            if matched_rank <= 1:
                hits_at_1 += 1
            if matched_rank <= 3:
                hits_at_3 += 1
            if matched_rank <= 5:
                hits_at_5 += 1
            if matched_rank <= 10:
                hits_at_10 += 1
        else:
            rr = 0.0

        reciprocal_ranks.append(rr)
        per_question_results.append(
            {
                "id": q.get("id"),
                "question": query_text,
                "target_doc": target_doc,
                "target_sec": target_sec,
                "matched_rank": matched_rank,
                "top_retrievals": [
                    {
                        "doc_id": c.doc_id,
                        "section": c.section,
                        "page": c.page,
                        "score": round(c.score, 4),
                        "snippet": c.content[:120].strip(),
                    }
                    for c in results[:3]
                ],
            }
        )

    recall_at_1 = round(hits_at_1 / total, 4)
    recall_at_3 = round(hits_at_3 / total, 4)
    recall_at_5 = round(hits_at_5 / total, 4)
    recall_at_10 = round(hits_at_10 / total, 4)
    mrr = round(sum(reciprocal_ranks) / total, 4)

    metrics = {
        "total_evaluated": total,
        "hits_at_1": hits_at_1,
        "hits_at_3": hits_at_3,
        "hits_at_5": hits_at_5,
        "hits_at_10": hits_at_10,
        "recall_at_1": recall_at_1,
        "recall_at_3": recall_at_3,
        "recall_at_5": recall_at_5,
        "recall_at_10": recall_at_10,
        "mrr": mrr,
        "details": per_question_results,
    }

    print("\n" + "=" * 50)
    print("HYBRID RETRIEVAL EVALUATION RESULTS")
    print("=" * 50)
    print(f"Total Evaluated: {total}")
    print(f"Recall@1:        {recall_at_1:.2%}")
    print(f"Recall@3:        {recall_at_3:.2%}")
    print(f"Recall@5:        {recall_at_5:.2%}")
    print(f"Recall@10:       {recall_at_10:.2%}")
    print(f"MRR:             {mrr:.4f}")
    print("=" * 50)

    if output_file:
        out_p = Path(output_file)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved retrieval metrics to {out_p}")

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run hybrid retrieval evaluation.")
    parser.add_argument("--questions", default="eval/datasets/sample_questions.jsonl", help="Questions JSONL")
    parser.add_argument("--out", default="results/retrieval_metrics.json", help="Output path")
    parser.add_argument("--top-k", type=int, default=10, help="Retrieval depth")
    args = parser.parse_args()

    evaluate_retrieval(Path(args.questions), Path(args.out), top_k=args.top_k)


if __name__ == "__main__":
    main()
