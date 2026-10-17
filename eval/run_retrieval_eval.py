"""Retrieval evaluation harness: computes recall@k (k=1, 3, 5, 10) and MRR on real ingested corpus.

Supports:
- Chunk-level exact evidence phrase matching: hit if evidence_phrase.lower() in chunk.content.lower()
- Document-level matching (legacy baseline): hit if target_doc in chunk.doc_id.lower()
- Filtered retrieval: restricting retrieval to target_doc via Qdrant and BM25 filters
- Side-by-side comparison tables
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from privatedocs.config import get_settings
from privatedocs.retrieval.bm25 import BM25Index
from privatedocs.retrieval.embedder import CPUTextEmbedder
from privatedocs.retrieval.hybrid import hybrid_search
from privatedocs.retrieval.vector_store import VectorStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eval_retrieval")


def evaluate_retrieval_mode(
    questions: list[dict[str, Any]],
    vector_store: VectorStore,
    embedder: CPUTextEmbedder,
    bm25: BM25Index,
    match_mode: str = "chunk",  # "chunk" or "document"
    use_doc_filter: bool = False,
    top_k: int = 10,
    rrf_k: int = 60,
) -> dict[str, Any]:
    """Run retrieval evaluation for a specific match mode and filter condition."""
    hits_at_1 = 0
    hits_at_3 = 0
    hits_at_5 = 0
    hits_at_10 = 0
    reciprocal_ranks: list[float] = []
    per_question_results: list[dict[str, Any]] = []

    total = len(questions)

    for q in questions:
        query_text = q["question"]
        target_doc = (q.get("source_doc") or q.get("doc_id") or "").lower()
        evidence_phrase = (q.get("evidence_phrase") or "").strip().lower()

        filter_doc = target_doc if (use_doc_filter and target_doc and target_doc != "none") else None

        # Execute dense, sparse, and hybrid search
        q_vec = embedder.embed_query(query_text)
        dense_hits = vector_store.search(q_vec, top_k=top_k * 2, filter_doc_id=filter_doc)
        bm25_hits = bm25.search(query_text, top_k=top_k * 2, filter_doc_id=filter_doc)
        results = hybrid_search(dense_hits, bm25_hits, rrf_k=rrf_k, top_k=top_k)

        matched_rank: int | None = None
        for rank, chunk in enumerate(results, start=1):
            if match_mode == "chunk":
                # Chunk-level hit: top-k chunk contains the exact evidence phrase
                if evidence_phrase and evidence_phrase in chunk.content.lower():
                    matched_rank = rank
                    break
            else:
                # Document-level hit: top-k chunk belongs to target document
                if target_doc and (target_doc in chunk.doc_id.lower() or chunk.doc_id.lower() in target_doc):
                    matched_rank = rank
                    break

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
                "evidence_phrase": evidence_phrase,
                "matched_rank": matched_rank,
                "top_retrievals": [
                    {
                        "chunk_id": c.chunk_id,
                        "doc_id": c.doc_id,
                        "section": c.section,
                        "page": c.page,
                        "score": round(c.score, 4),
                        "snippet": c.content[:100].strip(),
                    }
                    for c in results[:3]
                ],
            }
        )

    recall_at_1 = round(hits_at_1 / total, 4) if total > 0 else 0.0
    recall_at_3 = round(hits_at_3 / total, 4) if total > 0 else 0.0
    recall_at_5 = round(hits_at_5 / total, 4) if total > 0 else 0.0
    recall_at_10 = round(hits_at_10 / total, 4) if total > 0 else 0.0
    mrr = round(sum(reciprocal_ranks) / total, 4) if total > 0 else 0.0

    return {
        "match_mode": match_mode,
        "use_doc_filter": use_doc_filter,
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


def run_full_retrieval_eval(
    questions_file: Path,
    output_file: Path | None = None,
    top_k: int = 10,
) -> dict[str, Any]:
    """Execute complete retrieval evaluation comparing document-level vs chunk-level, unfiltered vs filtered."""
    with open(questions_file, encoding="utf-8") as f:
        questions = [json.loads(line) for line in f if line.strip()]

    answerable = [q for q in questions if q.get("type") != "unanswerable"]
    if not answerable:
        logger.warning("No answerable questions found in %s", questions_file)
        return {}

    logger.info("Evaluating retrieval over %d answerable questions from %s", len(answerable), questions_file)

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

    # 1. Old baseline: Document-level matching, unfiltered
    doc_unfiltered = evaluate_retrieval_mode(
        answerable, vector_store, embedder, bm25,
        match_mode="document", use_doc_filter=False, top_k=top_k, rrf_k=settings.RRF_K,
    )

    # 2. New standard: Chunk-level phrase matching, unfiltered
    chunk_unfiltered = evaluate_retrieval_mode(
        answerable, vector_store, embedder, bm25,
        match_mode="chunk", use_doc_filter=False, top_k=top_k, rrf_k=settings.RRF_K,
    )

    # 3. New standard: Chunk-level phrase matching, with doc_id filter
    chunk_filtered = evaluate_retrieval_mode(
        answerable, vector_store, embedder, bm25,
        match_mode="chunk", use_doc_filter=True, top_k=top_k, rrf_k=settings.RRF_K,
    )

    # Print side-by-side comparison
    print("\n" + "=" * 80)
    print("RETRIEVAL EVALUATION: DOCUMENT-LEVEL VS CHUNK-LEVEL (SIDE-BY-SIDE)")
    print("=" * 80)
    print(f"{'Metric':<14} | {'Old Doc-Level':<18} | {'New Chunk-Level (No Filter)':<28} | {'New Chunk-Level (Filtered)':<26}")
    print("-" * 80)
    print(f"{'Recall@1':<14} | {doc_unfiltered['recall_at_1']:<18.2%} | {chunk_unfiltered['recall_at_1']:<28.2%} | {chunk_filtered['recall_at_1']:<26.2%}")
    print(f"{'Recall@3':<14} | {doc_unfiltered['recall_at_3']:<18.2%} | {chunk_unfiltered['recall_at_3']:<28.2%} | {chunk_filtered['recall_at_3']:<26.2%}")
    print(f"{'Recall@5':<14} | {doc_unfiltered['recall_at_5']:<18.2%} | {chunk_unfiltered['recall_at_5']:<28.2%} | {chunk_filtered['recall_at_5']:<26.2%}")
    print(f"{'Recall@10':<14} | {doc_unfiltered['recall_at_10']:<18.2%} | {chunk_unfiltered['recall_at_10']:<28.2%} | {chunk_filtered['recall_at_10']:<26.2%}")
    print(f"{'MRR':<14} | {doc_unfiltered['mrr']:<18.4f} | {chunk_unfiltered['mrr']:<28.4f} | {chunk_filtered['mrr']:<26.4f}")
    print("=" * 80)

    comparison_results = {
        "dataset_file": str(questions_file),
        "total_answerable_questions": len(answerable),
        "old_document_level_unfiltered": doc_unfiltered,
        "new_chunk_level_unfiltered": chunk_unfiltered,
        "new_chunk_level_filtered": chunk_filtered,
        # Default top-level metrics reflecting the new chunk-level evaluation
        "recall_at_1": chunk_unfiltered["recall_at_1"],
        "recall_at_3": chunk_unfiltered["recall_at_3"],
        "recall_at_5": chunk_unfiltered["recall_at_5"],
        "recall_at_10": chunk_unfiltered["recall_at_10"],
        "mrr": chunk_unfiltered["mrr"],
    }

    if output_file:
        out_p = Path(output_file)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(comparison_results, f, indent=2)
        print(f"\nSaved retrieval comparison metrics to {out_p}")

    return comparison_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run retrieval evaluation with side-by-side comparison.")
    parser.add_argument("--questions", default="eval/datasets/eval_questions_40.jsonl", help="Input questions JSONL")
    parser.add_argument("--out", default="results/retrieval_metrics.json", help="Output JSON path")
    parser.add_argument("--top-k", type=int, default=10, help="Max Top-K for retrieval evaluation")
    args = parser.parse_args()

    run_full_retrieval_eval(
        questions_file=Path(args.questions),
        output_file=Path(args.out),
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
