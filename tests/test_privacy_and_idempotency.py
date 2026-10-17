"""Tests for idempotent re-ingestion and privacy-preserving logging."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi.testclient import TestClient

from privatedocs.api.auth import AuthDB, get_auth_db
from privatedocs.api.main import app, services
from privatedocs.generation.llm_client import LLMClient
from privatedocs.ingestion.chunking import chunk_loaded_documents
from privatedocs.ingestion.cleaning import clean_documents
from privatedocs.ingestion.loaders import load_document
from privatedocs.retrieval.bm25 import BM25Index
from privatedocs.retrieval.embedder import CPUTextEmbedder
from privatedocs.retrieval.vector_store import VectorStore
from tests.mock_llm_server import start_mock_server_thread


def test_idempotent_reingestion_no_duplicate_chunks():
    """Verify that ingesting the exact same document twice creates zero duplicate vectors in Qdrant."""
    vector_store = VectorStore(collection_name="test_idempotency_collection", in_memory=True)
    embedder = CPUTextEmbedder()
    embedder._fallback_mode = True  # Fast deterministic test mode

    # Load real test document
    doc_path = Path("data/sample_documents/sec_10k_sample.html")
    docs = load_document(doc_path)
    cleaned = clean_documents(docs)
    chunks = chunk_loaded_documents(cleaned, strategy="fixed", chunk_size=50, chunk_overlap=10)

    assert len(chunks) > 0, "Should produce at least one chunk"
    initial_chunk_count = len(chunks)

    # First ingestion
    embeddings_1 = embedder.embed_texts([c.content for c in chunks])
    vector_store.upsert_chunks(chunks, embeddings_1)

    points_1 = vector_store.client.count(collection_name="test_idempotency_collection").count
    assert points_1 == initial_chunk_count, f"Expected {initial_chunk_count} points, got {points_1}"

    # Second ingestion of identical chunks (simulating repeated index run)
    embeddings_2 = embedder.embed_texts([c.content for c in chunks])
    vector_store.upsert_chunks(chunks, embeddings_2)

    points_2 = vector_store.client.count(collection_name="test_idempotency_collection").count
    assert points_2 == initial_chunk_count, (
        f"Idempotency violation! Point count grew from {points_1} to {points_2}. "
        "Deterministic UUID point IDs must overwrite, not duplicate."
    )


def test_privacy_logging_audit_zero_prompt_leakage(caplog):
    """Audit application logs during /v1/ask: sensitive question text must NEVER appear in logs."""
    auth_db = AuthDB(db_path=":memory:")
    user_key = auth_db.create_api_key(role="user", description="Audit Key")

    vector_store = VectorStore(collection_name="audit_collection", in_memory=True)
    embedder = CPUTextEmbedder()
    embedder._fallback_mode = True
    bm25 = BM25Index()

    server, port = start_mock_server_thread()
    try:
        client = LLMClient(
            base_url=f"http://127.0.0.1:{port}/v1",
            api_key="mock",
            model="mock",
            timeout_seconds=5.0,
            max_retries=1,
        )

        services.vector_store = vector_store
        services.embedder = embedder
        services.bm25 = bm25
        services.llm_client = client

        app.dependency_overrides[get_auth_db] = lambda: auth_db

        sensitive_canary = "TOP_SECRET_CANARY_PROPRIETARY_MNA_MERGER_PRICE_42_BILLION"

        with caplog.at_level(logging.INFO):
            with TestClient(app) as test_client:
                response = test_client.post(
                    "/v1/ask",
                    headers={"X-API-Key": user_key},
                    json={"question": f"What is the {sensitive_canary}?"},
                )
                assert response.status_code == 200

        # Assert canary is NOT present in any logged record
        leaked_records = [
            rec.getMessage()
            for rec in caplog.records
            if sensitive_canary in rec.getMessage()
        ]

        assert len(leaked_records) == 0, (
            f"Privacy leak detected! Sensitive prompt leaked into application logs: {leaked_records}"
        )

        # Confirm request access log does record safe operational metrics
        access_logs = [rec.getMessage() for rec in caplog.records if "path=/v1/ask" in rec.getMessage()]
        assert len(access_logs) > 0, "Access log middleware should log path and status"
        for log in access_logs:
            assert "status=200" in log
            assert sensitive_canary not in log
    finally:
        server.shutdown()
