"""Retrieval Ablation Study: evaluates variants one change at a time on CPU.

Variants tested:
1. Baseline: BGE-small (384-dim), no prefix, hybrid retrieval, top-10, strict matching.
2. Baseline + Tolerant Match: same as baseline with boundary-tolerant matching.
3. Variant (a): BGE query-instruction prefix ("Represent this sentence for searching relevant passages: ")
4. Variant (d): Cross-encoder re-ranker ON (ms-marco-MiniLM-L-6-v2) on top-20 hybrid pool
5. Variant (c): bge-base-en-v1.5 (768-dim) dense embeddings
6. Variant (b): Chunk size comparison (250 vs 400 vs 600 words)

Evaluates on all 34 answerable questions.
Note: With n=34, exactly 1 question corresponds to 2.94 percentage points.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import time
from typing import Any

import numpy as np

from privatedocs.config import get_settings
from privatedocs.ingestion.chunking import chunk_loaded_documents
from privatedocs.ingestion.loaders import load_document
from privatedocs.retrieval.bm25 import BM25Index
from privatedocs.retrieval.embedder import CPUTextEmbedder
from privatedocs.retrieval.hybrid import hybrid_search
from privatedocs.retrieval.reranker import CrossEncoderReranker
from privatedocs.retrieval.vector_store import SearchResult, VectorStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eval_ablation")

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def is_chunk_match(evidence_phrase: str, chunk_content: str, tolerant: bool = False) -> bool:
    """Check if chunk contains the evidence phrase (strict or boundary-tolerant)."""
    if not evidence_phrase:
        return False
    lower_ev = evidence_phrase.strip().lower()
    lower_content = chunk_content.lower()

    if lower_ev in lower_content:
        return True

    if tolerant:
        # If phrase was chopped at boundary or contains table markdown pipes
        clean_ev = lower_ev.replace("|", " ").replace("-", " ")
        words = [w for w in clean_ev.split() if len(w) > 2]
        if len(words) >= 3:
            # Check if 80% of content words appear in chunk
            matches = sum(1 for w in words if w in lower_content)
            if matches / len(words) >= 0.80:
                return True
    return False


def run_evaluation_variant(
    questions: list[dict[str, Any]],
    embedder: CPUTextEmbedder,
    bm25: BM25Index,
    vector_store: VectorStore | None = None,
    query_prefix: str = "",
    reranker: CrossEncoderReranker | None = None,
    tolerant_matching: bool = False,
    top_k: int = 10,
    custom_chunks: list | None = None,
    custom_embeddings: np.ndarray | None = None,
) -> dict[str, Any]:
    """Evaluate retrieval metrics for a single variant."""
    total = len(questions)
    hits_at_1 = 0
    hits_at_5 = 0
    hits_at_10 = 0
    reciprocal_ranks: list[float] = []

    use_in_memory_dense = custom_chunks is not None or vector_store is None
    active_chunks = custom_chunks or bm25.chunks

    chunk_embeddings_np = None
    if use_in_memory_dense:
        if custom_embeddings is not None:
            chunk_embeddings_np = custom_embeddings
        else:
            texts = [c.content for c in active_chunks]
            raw_embs = embedder.embed_texts(texts)
            chunk_embeddings_np = np.array(raw_embs, dtype=np.float32)

    for q in questions:
        raw_query = q["question"]
        query_text = f"{query_prefix}{raw_query}" if query_prefix else raw_query
        ev_phrase = q.get("evidence_phrase", "")

        # 1. Dense candidates
        if use_in_memory_dense:
            q_vec = np.array(embedder.embed_query(query_text), dtype=np.float32)
            # Dot product (both are L2 normalized)
            sim_scores = np.dot(chunk_embeddings_np, q_vec)
            top_indices = np.argsort(-sim_scores)[: top_k * 2]
            dense_hits = [
                SearchResult(
                    chunk_id=active_chunks[i].chunk_id,
                    doc_id=active_chunks[i].doc_id,
                    content=active_chunks[i].content,
                    source=active_chunks[i].source,
                    page=active_chunks[i].page,
                    section=active_chunks[i].section,
                    score=float(sim_scores[i]),
                    metadata=active_chunks[i].metadata,
                )
                for i in top_indices
            ]
        else:
            q_vec = embedder.embed_query(query_text)
            dense_hits = vector_store.search(q_vec, top_k=top_k * 2)

        # 2. BM25 candidates
        bm25_hits = bm25.search(raw_query, top_k=top_k * 2)

        # 3. Hybrid fusion
        candidates = hybrid_search(dense_hits, bm25_hits, rrf_k=60, top_k=top_k * 2)

        # 4. Optional Re-ranking
        if reranker and reranker.enabled:
            candidates = reranker.rerank(raw_query, candidates, top_k=top_k)
        else:
            candidates = candidates[:top_k]

        matched_rank = None
        for rank, chunk in enumerate(candidates, start=1):
            if is_chunk_match(ev_phrase, chunk.content, tolerant=tolerant_matching):
                matched_rank = rank
                break

        if matched_rank is not None:
            rr = 1.0 / matched_rank
            if matched_rank <= 1:
                hits_at_1 += 1
            if matched_rank <= 5:
                hits_at_5 += 1
            if matched_rank <= 10:
                hits_at_10 += 1
        else:
            rr = 0.0

        reciprocal_ranks.append(rr)

    return {
        "total": total,
        "hits_at_1": hits_at_1,
        "hits_at_5": hits_at_5,
        "hits_at_10": hits_at_10,
        "recall_at_1": round(hits_at_1 / total, 4),
        "recall_at_5": round(hits_at_5 / total, 4),
        "recall_at_10": round(hits_at_10 / total, 4),
        "mrr": round(sum(reciprocal_ranks) / total, 4),
    }


def main() -> None:
    questions_path = Path("eval/datasets/eval_questions_40.jsonl")
    with open(questions_path, encoding="utf-8") as f:
        all_q = [json.loads(line) for line in f if line.strip()]
    questions = [q for q in all_q if q.get("type") != "unanswerable"]

    n = len(questions)
    pt_per_q = 100.0 / n
    logger.info("Running retrieval ablation on %d answerable questions (1 question = %.2f%%)", n, pt_per_q)

    settings = get_settings()
    vector_store = VectorStore(
        collection_name=settings.QDRANT_COLLECTION_NAME,
        host=settings.QDRANT_HOST,
        port=settings.QDRANT_PORT,
    )
    bm25 = BM25Index()
    bm25.load(Path("data/bm25_index.pkl"))

    bge_small = CPUTextEmbedder(model_name="BAAI/bge-small-en-v1.5", device="cpu")

    results_table: list[tuple[str, dict[str, Any]]] = []

    def log_result(name: str, res: dict[str, Any]) -> None:
        results_table.append((name, res))
        logger.info(
            "--> %-30s | Recall@1: %5.1f%% | Recall@5: %5.1f%% | Recall@10: %5.1f%% | MRR: %.4f",
            name,
            res["recall_at_1"] * 100.0,
            res["recall_at_5"] * 100.0,
            res["recall_at_10"] * 100.0,
            res["mrr"],
        )

    # 1. Baseline (Strict)
    logger.info("Evaluating [1] Baseline (bge-small, chunk_size=400, strict matching)...")
    res_base = run_evaluation_variant(questions, bge_small, bm25, vector_store=vector_store, tolerant_matching=False)
    log_result("Baseline (strict)", res_base)

    # 2. Baseline with Boundary-Tolerant Matching
    logger.info("Evaluating [2] Baseline with Boundary-Tolerant Matching...")
    res_base_tol = run_evaluation_variant(questions, bge_small, bm25, vector_store=vector_store, tolerant_matching=True)
    log_result("Baseline + Tolerant Match", res_base_tol)

    # 3. Variant (a): BGE Query-Instruction Prefix
    logger.info("Evaluating [3] Variant (a): BGE Query-Instruction Prefix...")
    res_prefix = run_evaluation_variant(
        questions, bge_small, bm25, vector_store=vector_store,
        query_prefix=BGE_QUERY_PREFIX, tolerant_matching=False
    )
    log_result("Variant (a): BGE Prefix", res_prefix)

    # 4. Variant (d): Cross-Encoder Re-ranker ON
    logger.info("Evaluating [4] Variant (d): Re-ranker ON (ms-marco-MiniLM-L-6-v2)...")
    reranker = CrossEncoderReranker(enabled=True, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    res_rerank = run_evaluation_variant(
        questions, bge_small, bm25, vector_store=vector_store,
        reranker=reranker, tolerant_matching=False
    )
    log_result("Variant (d): Re-ranker ON", res_rerank)

    # 5. Variant (c): bge-base-en-v1.5
    logger.info("Evaluating [5] Variant (c): bge-base-en-v1.5...")
    bge_base_cache = Path("data/cache_bge_base_embeddings.npy")
    bge_base = CPUTextEmbedder(model_name="BAAI/bge-base-en-v1.5", device="cpu")
    if bge_base_cache.exists():
        logger.info("Loading cached bge-base embeddings from %s...", bge_base_cache)
        bge_base_embs = np.load(bge_base_cache)
    else:
        logger.info("Computing bge-base embeddings for %d chunks...", len(bm25.chunks))
        texts = [c.content for c in bm25.chunks]
        raw_embs = bge_base.embed_texts(texts)
        bge_base_embs = np.array(raw_embs, dtype=np.float32)
        bge_base_cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(bge_base_cache, bge_base_embs)
        logger.info("Saved bge-base embeddings to %s", bge_base_cache)

    res_base_model = run_evaluation_variant(
        questions, bge_base, bm25, vector_store=None,
        tolerant_matching=False, custom_chunks=bm25.chunks,
        custom_embeddings=bge_base_embs,
    )
    log_result("Variant (c): bge-base", res_base_model)

    # 6. Variant (b): Chunk Sizes 250 / 400 / 600
    logger.info("Evaluating [6] Variant (b): Chunk Sizes 250, 400, 600...")
    raw_files = sorted(Path("data/raw_docs").glob("*.*"))
    loaded_docs: list = []
    for f in raw_files:
        loaded_docs.extend(load_document(f))

    # Pre-evaluated 400 is baseline
    for csize in [250, 400, 600]:
        if csize == 400:
            log_result("Variant (b): Chunk Size 400 (Baseline)", res_base)
            continue

        c_overlap = int(csize * 0.15)
        chunks_var = chunk_loaded_documents(
            loaded_docs, strategy="recursive", chunk_size=csize, chunk_overlap=c_overlap
        )
        bm25_var = BM25Index()
        bm25_var.index_chunks(chunks_var)

        # For chunk size comparison, evaluate with BM25 + dense (fast pseudo or bge-small)
        # Using bge-small on custom chunks
        cache_csize = Path(f"data/cache_chunks_{csize}_bgesmall.npy")
        if cache_csize.exists():
            csize_embs = np.load(cache_csize)
        else:
            texts = [c.content for c in chunks_var]
            csize_embs = np.array(bge_small.embed_texts(texts), dtype=np.float32)
            np.save(cache_csize, csize_embs)

        res_csize = run_evaluation_variant(
            questions, bge_small, bm25_var, vector_store=None,
            tolerant_matching=False, custom_chunks=chunks_var,
            custom_embeddings=csize_embs,
        )
        log_result(f"Variant (b): Chunk Size {csize}", res_csize)

    # Print summary table
    print("\n" + "=" * 80)
    print("RETRIEVAL ABLATION RESULTS (CPU ONLY)")
    print(f"Sample Size n={n} answerable questions | 1 question = {pt_per_q:.2f} percentage points")
    print("=" * 80)
    print(f"{'Variant / Configuration':<32} | {'Recall@1':<10} | {'Recall@5':<10} | {'Recall@10':<11} | {'MRR':<8}")
    print("-" * 80)
    for name, m in results_table:
        print(f"{name:<32} | {m['recall_at_1']:<10.2%} | {m['recall_at_5']:<10.2%} | {m['recall_at_10']:<11.2%} | {m['mrr']:<8.4f}")
    print("=" * 80)

    # Save to results/retrieval_ablation.json
    out_p = Path("results/retrieval_ablation.json")
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(
            {
                "n": n,
                "points_per_question": pt_per_q,
                "variants": [
                    {"variant": name, **m} for name, m in results_table
                ],
            },
            f,
            indent=2,
        )
    print(f"Saved ablation results to {out_p}")


if __name__ == "__main__":
    main()
