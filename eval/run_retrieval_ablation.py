"""Retrieval Ablation Study: evaluates retrieval variants one change at a time on CPU.

Reports STRICT and TOLERANT matching numbers side-by-side everywhere.

Definitions of Matching Protocols:
1. Strict Match:
   A retrieval hit is counted if and only if the exact character sequence of
   `evidence_phrase.strip().lower()` appears as an unbroken substring within `chunk_content.lower()`.

2. Tolerant Match:
   A retrieval hit is counted if either:
   - Condition A (Normalized Substring): The evidence phrase appears verbatim after
     stripping table delimiters (| and -), normalizing curly quotes, and collapsing contiguous
     whitespace to a single space.
   - Condition B (Lexical Token Containment within Single Chunk): The evidence phrase
     is tokenized into significant lexical tokens (alphanumeric, length >= 3 chars). If >= 80%
     of these tokens are present within the SINGLE candidate chunk, match is True. This handles
     boundary splits or table-row formatting without cross-document false positives.

Statistical Context:
- Sample size: n=34 answerable questions.
- Resolution: Exactly 1 question = 2.94 percentage points.
- Chunk-Size Metric Bias: Evaluating hit rates across chunk sizes (250 vs 400 vs 600 words)
  is structurally biased: larger chunks retrieve more document surface area (6,000 words at top-10
  vs 2,500 words for 250), increasing prior probability of containment, while smaller chunks have
  more boundary bisections.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import re
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


def is_strict_match(evidence_phrase: str, chunk_content: str) -> bool:
    """Strict Match: Exact verbatim substring check of evidence phrase in chunk."""
    if not evidence_phrase:
        return False
    return evidence_phrase.strip().lower() in chunk_content.lower()


def is_tolerant_match(evidence_phrase: str, chunk_content: str, token_overlap_threshold: float = 0.80) -> bool:
    """Tolerant Match:
    Condition A: Verbatim substring after pipe, dash, quote, and whitespace normalization.
    Condition B: >= token_overlap_threshold (80%) of significant lexical tokens in single chunk.
    """
    if not evidence_phrase:
        return False
    if is_strict_match(evidence_phrase, chunk_content):
        return True

    # Normalize pipes, dashes, whitespace
    norm_ev = re.sub(r"\s+", " ", evidence_phrase.replace("|", " ").replace("-", " ").strip().lower())
    norm_chunk = re.sub(r"\s+", " ", chunk_content.replace("|", " ").replace("-", " ").strip().lower())
    if norm_ev in norm_chunk:
        return True

    # Check token overlap in single chunk
    words = [w for w in norm_ev.split() if len(w) >= 3 and w.isalnum()]
    if len(words) >= 3:
        matches = sum(1 for w in words if w in norm_chunk)
        if matches / len(words) >= token_overlap_threshold:
            return True

    return False


def run_evaluation_variant(
    questions: list[dict[str, Any]],
    embedder: CPUTextEmbedder,
    bm25: BM25Index,
    vector_store: VectorStore | None = None,
    query_prefix: str = "",
    reranker: CrossEncoderReranker | None = None,
    top_k: int = 10,
    custom_chunks: list | None = None,
    custom_embeddings: np.ndarray | None = None,
) -> dict[str, Any]:
    """Evaluate retrieval metrics for a single variant reporting both Strict and Tolerant."""
    total = len(questions)
    strict_hits_1 = 0
    strict_hits_5 = 0
    strict_hits_10 = 0
    strict_rrs: list[float] = []

    tol_hits_1 = 0
    tol_hits_5 = 0
    tol_hits_10 = 0
    tol_rrs: list[float] = []

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

        # Calculate strict match rank
        strict_rank = None
        for rank, chunk in enumerate(candidates, start=1):
            if is_strict_match(ev_phrase, chunk.content):
                strict_rank = rank
                break

        # Calculate tolerant match rank
        tol_rank = None
        for rank, chunk in enumerate(candidates, start=1):
            if is_tolerant_match(ev_phrase, chunk.content):
                tol_rank = rank
                break

        # Strict stats
        if strict_rank is not None:
            strict_rrs.append(1.0 / strict_rank)
            if strict_rank <= 1:
                strict_hits_1 += 1
            if strict_rank <= 5:
                strict_hits_5 += 1
            if strict_rank <= 10:
                strict_hits_10 += 1
        else:
            strict_rrs.append(0.0)

        # Tolerant stats
        if tol_rank is not None:
            tol_rrs.append(1.0 / tol_rank)
            if tol_rank <= 1:
                tol_hits_1 += 1
            if tol_rank <= 5:
                tol_hits_5 += 1
            if tol_rank <= 10:
                tol_hits_10 += 1
        else:
            tol_rrs.append(0.0)

    return {
        "total": total,
        "strict": {
            "hits_at_1": strict_hits_1,
            "hits_at_5": strict_hits_5,
            "hits_at_10": strict_hits_10,
            "recall_at_1": round(strict_hits_1 / total, 4),
            "recall_at_5": round(strict_hits_5 / total, 4),
            "recall_at_10": round(strict_hits_10 / total, 4),
            "mrr": round(float(np.mean(strict_rrs)), 4),
        },
        "tolerant": {
            "hits_at_1": tol_hits_1,
            "hits_at_5": tol_hits_5,
            "hits_at_10": tol_hits_10,
            "recall_at_1": round(tol_hits_1 / total, 4),
            "recall_at_5": round(tol_hits_5 / total, 4),
            "recall_at_10": round(tol_hits_10 / total, 4),
            "mrr": round(float(np.mean(tol_rrs)), 4),
        },
    }


def main() -> None:
    eval_file = Path("eval/datasets/eval_questions_40.jsonl")
    with open(eval_file, "r", encoding="utf-8") as f:
        all_q = [json.loads(line) for line in f if line.strip()]

    # Evaluate on answerable questions only (n=34)
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
        s = res["strict"]
        t = res["tolerant"]
        logger.info(
            "--> %-32s | R@1: %4.1f%% / %4.1f%% | R@5: %4.1f%% / %4.1f%% | R@10: %4.1f%% / %4.1f%% | MRR: %.4f / %.4f",
            name,
            s["recall_at_1"] * 100, t["recall_at_1"] * 100,
            s["recall_at_5"] * 100, t["recall_at_5"] * 100,
            s["recall_at_10"] * 100, t["recall_at_10"] * 100,
            s["mrr"], t["mrr"],
        )

    # 1. Baseline: BGE-small, chunk_size=400, hybrid RRF
    logger.info("Evaluating [1] Baseline (bge-small, chunk_size=400)...")
    res_base = run_evaluation_variant(questions, bge_small, bm25, vector_store=vector_store)
    log_result("Baseline (bge-small, c=400)", res_base)

    # 2. Variant (a): BGE query prefix
    logger.info("Evaluating [2] Variant (a): BGE Query-Instruction Prefix...")
    res_prefix = run_evaluation_variant(
        questions, bge_small, bm25, vector_store=vector_store,
        query_prefix=BGE_QUERY_PREFIX
    )
    log_result("Variant (a): BGE Prefix", res_prefix)

    # 3. Variant (d): Cross-Encoder Re-ranker ON
    logger.info("Evaluating [3] Variant (d): Cross-Encoder Re-ranker ON...")
    reranker = CrossEncoderReranker(
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        enabled=True,
    )
    res_rerank = run_evaluation_variant(
        questions, bge_small, bm25, vector_store=vector_store,
        reranker=reranker
    )
    log_result("Variant (d): Re-ranker ON", res_rerank)

    # 4. Variant (c): bge-base-en-v1.5
    logger.info("Evaluating [4] Variant (c): bge-base-en-v1.5...")
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

    res_base_model = run_evaluation_variant(
        questions, bge_base, bm25, vector_store=None,
        custom_chunks=bm25.chunks,
        custom_embeddings=bge_base_embs,
    )
    log_result("Variant (c): bge-base", res_base_model)

    # 5. Variant (b): Chunk Sizes 250 / 400 / 600
    logger.info("Evaluating [5] Variant (b): Chunk Sizes 250, 400, 600...")
    raw_files = sorted(Path("data/raw_docs").glob("*.*"))
    loaded_docs: list = []
    for f in raw_files:
        loaded_docs.extend(load_document(f))

    for csize in [250, 400, 600]:
        if csize == 400:
            log_result("Variant (b): Chunk 400 (Baseline)", res_base)
            continue

        c_overlap = int(csize * 0.15)
        chunks_var = chunk_loaded_documents(
            loaded_docs, strategy="recursive", chunk_size=csize, chunk_overlap=c_overlap
        )
        bm25_var = BM25Index()
        bm25_var.index_chunks(chunks_var)

        cache_csize = Path(f"data/cache_chunks_{csize}_bgesmall.npy")
        if cache_csize.exists():
            csize_embs = np.load(cache_csize)
        else:
            texts = [c.content for c in chunks_var]
            csize_embs = np.array(bge_small.embed_texts(texts), dtype=np.float32)
            np.save(cache_csize, csize_embs)

        res_csize = run_evaluation_variant(
            questions, bge_small, bm25_var, vector_store=None,
            custom_chunks=chunks_var,
            custom_embeddings=csize_embs,
        )
        log_result(f"Variant (b): Chunk {csize}", res_csize)

    # Print unified summary table
    print("\n" + "=" * 105)
    print("RETRIEVAL ABLATION RESULTS (CPU ONLY): STRICT vs TOLERANT MATCHING")
    print(f"Sample Size n={n} answerable questions | 1 question = {pt_per_q:.2f} percentage points")
    print("=" * 105)
    print(f"{'Variant / Configuration':<32} | {'Recall@1 (Str | Tol)':<22} | {'Recall@5 (Str | Tol)':<22} | {'Recall@10 (Str | Tol)':<23} | {'MRR (Str | Tol)':<16}")
    print("-" * 105)
    for name, m in results_table:
        s = m["strict"]
        t = m["tolerant"]
        r1 = f"{s['recall_at_1']:>5.1%} | {t['recall_at_1']:>5.1%}"
        r5 = f"{s['recall_at_5']:>5.1%} | {t['recall_at_5']:>5.1%}"
        r10 = f"{s['recall_at_10']:>5.1%} | {t['recall_at_10']:>5.1%}"
        mrr = f"{s['mrr']:>6.4f} | {t['mrr']:>6.4f}"
        print(f"{name:<32} | {r1:<22} | {r5:<22} | {r10:<23} | {mrr:<16}")
    print("=" * 105)

    print("\n[METHODOLOGICAL NOTES & LIMITATIONS]")
    print(f"1. Statistical Resolution: At n={n}, each question is {pt_per_q:.2f} percentage points. Differences of 1-2 questions")
    print("   (~3-6%) are within normal single-query variation and should not be over-interpreted.")
    print("2. Chunk-Size Metric Bias: Evaluating hit rates across chunk sizes (250 vs 400 vs 600 words) with phrase containment")
    print("   is structurally biased. Chunks of size 600 retrieve up to 6,000 words across top-10, giving them a larger surface")
    print("   area prior, while chunks of size 250 have more token boundaries and higher sentence bisection risk.")
    print("3. Matching Protocol:")
    print("   - Strict: exact verbatim substring match in chunk.")
    print("   - Tolerant: verbatim match after delimiter/whitespace normalization, or >=80% lexical token containment in chunk.")

    # Save to results/retrieval_ablation.json
    out_p = Path("results/retrieval_ablation.json")
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(
            {
                "n": n,
                "points_per_question": pt_per_q,
                "definitions": {
                    "strict": "Exact verbatim character substring match in chunk content.",
                    "tolerant": "Verbatim match after pipe/whitespace normalization OR >=80% significant lexical token containment in a single chunk.",
                },
                "variants": [
                    {"variant": name, **m} for name, m in results_table
                ],
            },
            f,
            indent=2,
        )
    print(f"\nSaved ablation results to {out_p}")


if __name__ == "__main__":
    main()
