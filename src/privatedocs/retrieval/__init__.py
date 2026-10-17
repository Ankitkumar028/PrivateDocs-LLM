"""Retrieval module for dense vector search, BM25, and hybrid fusion."""

from privatedocs.retrieval.bm25 import BM25Index
from privatedocs.retrieval.embedder import CPUTextEmbedder
from privatedocs.retrieval.hybrid import hybrid_search, reciprocal_rank_fusion
from privatedocs.retrieval.reranker import CrossEncoderReranker
from privatedocs.retrieval.vector_store import SearchResult, VectorStore

__all__ = [
    "BM25Index",
    "CPUTextEmbedder",
    "CrossEncoderReranker",
    "SearchResult",
    "VectorStore",
    "hybrid_search",
    "reciprocal_rank_fusion",
]
