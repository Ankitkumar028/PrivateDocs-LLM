"""Unit tests for BM25 and Reciprocal Rank Fusion hybrid retrieval."""

from privatedocs.ingestion.chunking import DocumentChunk
from privatedocs.retrieval.bm25 import BM25Index
from privatedocs.retrieval.hybrid import hybrid_search, reciprocal_rank_fusion
from privatedocs.retrieval.vector_store import SearchResult


def test_bm25_index_and_search():
    index = BM25Index()
    chunks = [
        DocumentChunk(
            chunk_id="c1",
            doc_id="doc1",
            content="AMD MI300X offers 192GB of high bandwidth memory HBM3.",
            source="amd_specs.txt",
            page=1,
        ),
        DocumentChunk(
            chunk_id="c2",
            doc_id="doc2",
            content="FastAPI is an asynchronous web framework for Python.",
            source="fastapi_docs.txt",
            page=1,
        ),
        DocumentChunk(
            chunk_id="c3",
            doc_id="doc3",
            content="Qdrant is a vector similarity search engine written in Rust.",
            source="qdrant_docs.txt",
            page=1,
        ),
    ]

    index.index_chunks(chunks)

    # Search for AMD MI300X
    results = index.search("MI300X memory", top_k=2)
    assert len(results) > 0
    assert results[0].chunk_id == "c1"
    assert results[0].doc_id == "doc1"
    assert results[0].score > 0.0

    # Search for non-existent token
    no_results = index.search("nonexistenttermxyz", top_k=2)
    assert len(no_results) == 0


def test_reciprocal_rank_fusion():
    item_a = SearchResult(chunk_id="cA", doc_id="dA", content="A", source="sA", page=1, section="m", score=0.9, metadata={})
    item_b = SearchResult(chunk_id="cB", doc_id="dB", content="B", source="sB", page=1, section="m", score=0.8, metadata={})
    item_c = SearchResult(chunk_id="cC", doc_id="dC", content="C", source="sC", page=1, section="m", score=0.7, metadata={})

    # List 1: A, B
    list1 = [item_a, item_b]
    # List 2: B, C
    list2 = [item_b, item_c]

    # Item B appears in both lists, so its fused score should be highest!
    # Rank 2 in list 1: 1 / (60 + 2) = 1/62
    # Rank 1 in list 2: 1 / (60 + 1) = 1/61
    # B total = 1/62 + 1/61 ≈ 0.016129 + 0.016393 = 0.03252
    # A total = 1 / (60 + 1) = 1/61 ≈ 0.016393
    # C total = 1 / (60 + 2) = 1/62 ≈ 0.016129
    fused = reciprocal_rank_fusion([list1, list2], rrf_k=60, top_k=3)

    assert len(fused) == 3
    assert fused[0].chunk_id == "cB", "Document present in both lists must be ranked #1"
    assert fused[1].chunk_id == "cA"
    assert fused[2].chunk_id == "cC"


def test_hybrid_search_wrapper():
    dense_hits = [
        SearchResult(chunk_id="c1", doc_id="d1", content="Text 1", source="s1", page=1, section="", score=0.88, metadata={}),
    ]
    bm25_hits = [
        SearchResult(chunk_id="c2", doc_id="d2", content="Text 2", source="s2", page=1, section="", score=1.45, metadata={}),
    ]

    combined = hybrid_search(dense_hits, bm25_hits, rrf_k=60, top_k=2)
    assert len(combined) == 2
    chunk_ids = [c.chunk_id for c in combined]
    assert "c1" in chunk_ids
    assert "c2" in chunk_ids
