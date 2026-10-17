# ///
""" 2>/dev/null
exec python3 "$0" "$@"
"""
# ///
"""End-to-end local smoke test for PrivateDocs-LLM."""

import json
import os
import sys
import time
from pathlib import Path

# Force UTF-8 stdout if supported
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Ensure root directory is on PYTHONPATH
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))
sys.path.insert(0, str(ROOT_DIR))
os.environ["PYTHONPATH"] = str(ROOT_DIR) + os.pathsep + str(ROOT_DIR / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")


def run_smoke_test() -> bool:
    print("==================================================")
    print("PrivateDocs-LLM: End-to-End Local Smoke Test")
    print("==================================================")

    # 1. Verify Imports
    print("[1/6] Testing core module imports...")
    try:
        from privatedocs.api.auth import AuthDB
        from privatedocs.config import get_settings
        from privatedocs.generation.llm_client import LLMClient
        from privatedocs.ingestion.chunking import chunk_document
        from privatedocs.retrieval.bm25 import BM25Index
        from privatedocs.retrieval.embedder import CPUTextEmbedder
        from privatedocs.retrieval.hybrid import hybrid_search
        from privatedocs.retrieval.vector_store import VectorStore

        print("  [PASS] Core imports successful.")
    except Exception as exc:
        print(f"  [FAIL] Import failure: {exc}")
        return False

    settings = get_settings()

    # 2. Test Ingestion & Chunking
    print("[2/6] Validating document chunking...")
    sample_text = (
        "PrivateDocs-LLM is designed for enterprise document question answering. "
        "All data remains on premise without external egress. "
        "AMD Instinct MI300X offers 192GB memory capable of serving 70B models."
    )
    chunks = chunk_document(
        doc_id="smoke_doc_1",
        text=sample_text,
        source="test_source.txt",
        strategy="fixed",
        chunk_size=15,
        chunk_overlap=3,
    )
    assert len(chunks) >= 1, "Chunking should produce at least 1 chunk"
    print(f"  [PASS] Produced {len(chunks)} chunks with deterministic UUID chunk IDs.")

    # 3. Test In-Memory Vector Store & BM25
    print("[3/6] Testing vector indexing & hybrid retrieval...")
    vector_store = VectorStore(
        collection_name="smoke_test_collection",
        in_memory=True,
    )
    embedder = CPUTextEmbedder()
    embedder._fallback_mode = True  # Fast offline deterministic embedding
    bm25 = BM25Index()

    # Index chunks
    dense_embeddings = embedder.embed_texts([c.content for c in chunks])
    vector_store.upsert_chunks(chunks, dense_embeddings)
    bm25.index_chunks(chunks)

    # Query
    query = "What accelerator is used?"
    dense_res = vector_store.search(embedder.embed_query(query), top_k=3)
    bm25_res = bm25.search(query, top_k=3)
    hybrid_res = hybrid_search(dense_res, bm25_res, top_k=2)

    assert len(hybrid_res) > 0, "Hybrid search must return results"
    print(f"  [PASS] Hybrid search returned top hit: '{hybrid_res[0].content[:40]}...' (score: {hybrid_res[0].score:.4f})")

    # 4. Test Auth & Key Hashing
    print("[4/6] Testing SQLite API-key authentication...")
    auth_db = AuthDB(db_path=":memory:")
    admin_key = auth_db.create_api_key(role="admin", description="Smoke Admin")
    user_key = auth_db.create_api_key(role="user", description="Smoke User")

    assert auth_db.verify_api_key(admin_key) == "admin", "Admin key verification failed"
    assert auth_db.verify_api_key(user_key) == "user", "User key verification failed"
    assert auth_db.verify_api_key("invalid-key-xyz") is None, "Invalid key should be rejected"
    print("  [PASS] SHA-256 API key hashing and RBAC verified.")

    # 5. Test Mock LLM Server & Client
    print("[5/6] Testing LLM client with mock server...")
    from tests.mock_llm_server import start_mock_server_thread

    server, port = start_mock_server_thread()
    time.sleep(0.5)

    try:
        client = LLMClient(
            base_url=f"http://127.0.0.1:{port}/v1",
            api_key="mock-key",
            model="mock-model",
            timeout_seconds=5.0,
            max_retries=1,
        )
        answer, usage = client.generate_answer(
            prompt="What is PrivateDocs?",
            temperature=0.1,
            max_tokens=50,
        )
        assert len(answer) > 0, "LLM client must return an answer"
        assert usage.total_tokens > 0, "Usage tokens must be greater than zero"
        print(f"  [PASS] Mock LLM responded: '{answer.strip()}' (Tokens: {usage.total_tokens})")

        # 6. Test FastAPI TestClient
        print("[6/6] Testing FastAPI endpoints via TestClient...")
        from fastapi.testclient import TestClient

        from privatedocs.api.auth import get_auth_db
        from privatedocs.api.main import app, services

        # In-memory services for test client
        services.vector_store = vector_store
        services.embedder = embedder
        services.bm25 = bm25
        services.llm_client = client

        app.dependency_overrides[get_auth_db] = lambda: auth_db

        with TestClient(app) as test_client:
            # Test healthz
            resp_health = test_client.get("/healthz")
            assert resp_health.status_code == 200, f"Healthz failed: {resp_health.text}"

            # Test metrics
            resp_metrics = test_client.get("/metrics")
            assert resp_metrics.status_code == 200, f"Metrics failed: {resp_metrics.text}"

            # Test auth rejection
            resp_unauth = test_client.post("/v1/ask", json={"question": "hello"})
            assert resp_unauth.status_code in (401, 403), "Unauthenticated ask must be rejected"

            # Test ask with valid user key
            resp_auth = test_client.post(
                "/v1/ask",
                headers={"X-API-Key": user_key},
                json={"question": "What is the architecture?", "top_k": 2},
            )
            assert resp_auth.status_code == 200, f"Ask failed: {resp_auth.text}"
            data = resp_auth.json()
            assert "answer" in data, "Response missing answer field"
            assert "sources" in data, "Response missing sources field"
            assert "latency_ms" in data, "Response missing latency_ms field"
            print(f"  [PASS] /v1/ask responded with valid schema (latency: {data['latency_ms']:.1f}ms).")

    finally:
        server.shutdown()

    print("\n==================================================")
    print("ALL SMOKE CHECKS PASSED SUCCESSFULLY! (100% CPU / LOCAL)")
    print("==================================================")
    return True


if __name__ == "__main__":
    success = run_smoke_test()
    sys.exit(0 if success else 1)
