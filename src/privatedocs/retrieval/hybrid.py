"""Reciprocal Rank Fusion (RRF) for hybrid retrieval merging dense and sparse results."""

from __future__ import annotations

from privatedocs.retrieval.vector_store import SearchResult


def reciprocal_rank_fusion(
    ranked_lists: list[list[SearchResult]],
    rrf_k: int = 60,
    top_k: int = 5,
) -> list[SearchResult]:
    """Fuse multiple ranked search result lists using standard Reciprocal Rank Fusion.

    RRF formula: score(d) = sum_{list} (1 / (rrf_k + rank(d)))
    """
    scores: dict[str, float] = {}
    doc_map: dict[str, SearchResult] = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, start=1):
            chunk_id = item.chunk_id
            if chunk_id not in scores:
                scores[chunk_id] = 0.0
                doc_map[chunk_id] = item
            scores[chunk_id] += 1.0 / (rrf_k + rank)

    # Sort items by fused RRF score descending
    sorted_ids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)

    fused_results: list[SearchResult] = []
    for cid in sorted_ids[:top_k]:
        original_item = doc_map[cid]
        fused_item = SearchResult(
            chunk_id=original_item.chunk_id,
            doc_id=original_item.doc_id,
            content=original_item.content,
            source=original_item.source,
            page=original_item.page,
            section=original_item.section,
            score=scores[cid],
            metadata=dict(original_item.metadata, rrf_score=scores[cid]),
        )
        fused_results.append(fused_item)

    return fused_results


def hybrid_search(
    dense_results: list[SearchResult],
    bm25_results: list[SearchResult],
    rrf_k: int = 60,
    top_k: int = 5,
) -> list[SearchResult]:
    """Combine dense and BM25 results with Reciprocal Rank Fusion."""
    return reciprocal_rank_fusion(
        ranked_lists=[dense_results, bm25_results],
        rrf_k=rrf_k,
        top_k=top_k,
    )
