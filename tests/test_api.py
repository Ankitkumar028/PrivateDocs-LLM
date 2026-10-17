"""Unit and integration tests for FastAPI endpoints with mock LLM server."""

import io
import time

import pytest
from fastapi.testclient import TestClient

from privatedocs.api.auth import AuthDB, get_auth_db
from privatedocs.api.main import app, services
from privatedocs.api.rate_limit import RateLimiter, get_rate_limiter
from privatedocs.generation.llm_client import LLMClient
from privatedocs.retrieval.vector_store import VectorStore
from tests.mock_llm_server import start_mock_server_thread


@pytest.fixture(scope="module")
def mock_llm():
    """Start mock OpenAI-compatible server in background."""
    server, port = start_mock_server_thread()
    time.sleep(0.5)
    yield f"http://127.0.0.1:{port}/v1"
    server.shutdown()


@pytest.fixture
def test_client(mock_llm):
    # Setup test auth DB
    test_auth_db = AuthDB(db_path=":memory:")
    test_auth_db.add_key("admin-test-key", role="admin", description="Admin Test")
    test_auth_db.add_key("user-test-key", role="user", description="User Test")

    # Wire into FastAPI dependency overrides
    app.dependency_overrides[get_auth_db] = lambda: test_auth_db

    # Configure services for in-memory, fast embedder, and mock LLM
    from privatedocs.retrieval.embedder import CPUTextEmbedder
    services.embedder = CPUTextEmbedder()
    services.embedder._fallback_mode = True
    services.vector_store = VectorStore(
        collection_name="test_api_collection",
        in_memory=True,
    )
    services.llm_client = LLMClient(
        base_url=mock_llm,
        api_key="mock-key",
        model="mock-model",
        timeout_seconds=5.0,
        max_retries=1,
    )

    with TestClient(app) as client:
        yield client

    app.dependency_overrides.clear()


def test_healthz(test_client):
    response = test_client.get("/healthz")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_metrics(test_client):
    response = test_client.get("/metrics")
    assert response.status_code == 200
    assert "ask_requests_total" in response.text


def test_readyz(test_client):
    response = test_client.get("/readyz")
    assert response.status_code == 200
    data = response.json()
    assert data["ready"] is True
    assert data["qdrant_healthy"] is True
    assert data["llm_healthy"] is True


def test_ask_unauthorized(test_client):
    response = test_client.post("/v1/ask", json={"question": "Where is the data stored?"})
    assert response.status_code == 401


def test_ask_authorized(test_client):
    response = test_client.post(
        "/v1/ask",
        headers={"X-API-Key": "user-test-key"},
        json={"question": "What is the memory capacity of MI300X?", "top_k": 3},
    )
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data
    assert "latency_ms" in data
    assert "tokens" in data
    assert len(data["answer"]) > 0

    # Token accounting verification
    assert "prompt_tokens" in data
    assert "completion_tokens" in data
    assert "total_tokens" in data
    assert data["prompt_tokens"] > 0, "Prompt tokens must be reported and > 0"
    assert data["completion_tokens"] > 0, "Completion tokens must be reported and > 0"
    assert data["total_tokens"] == data["prompt_tokens"] + data["completion_tokens"], (
        "total_tokens must equal prompt_tokens + completion_tokens"
    )
    assert data["tokens"] == data["total_tokens"]


def test_ingest_requires_admin(test_client):
    file_content = b"Confidential memo regarding hardware infrastructure."
    files = {"file": ("memo.txt", io.BytesIO(file_content), "text/plain")}

    # Attempt with user key -> 403 Forbidden
    resp_user = test_client.post(
        "/v1/ingest",
        headers={"X-API-Key": "user-test-key"},
        files=files,
    )
    assert resp_user.status_code == 403

    # Attempt with admin key -> 200 OK
    files = {"file": ("memo.txt", io.BytesIO(file_content), "text/plain")}
    resp_admin = test_client.post(
        "/v1/ingest",
        headers={"X-API-Key": "admin-test-key"},
        files=files,
    )
    assert resp_admin.status_code == 200
    data = resp_admin.json()
    assert data["status"] == "success"
    assert data["chunks_indexed"] >= 1


def test_rate_limiting(test_client):
    # Set low rate limit for test
    strict_limiter = RateLimiter(requests_per_minute=2)
    app.dependency_overrides[get_rate_limiter] = lambda: strict_limiter

    headers = {"X-API-Key": "user-test-key"}
    # Req 1: ok
    r1 = test_client.post("/v1/ask", headers=headers, json={"question": "q1"})
    assert r1.status_code == 200

    # Req 2: ok
    r2 = test_client.post("/v1/ask", headers=headers, json={"question": "q2"})
    assert r2.status_code == 200

    # Req 3: rejected 429
    r3 = test_client.post("/v1/ask", headers=headers, json={"question": "q3"})
    assert r3.status_code == 429
    assert "Retry-After" in r3.headers
