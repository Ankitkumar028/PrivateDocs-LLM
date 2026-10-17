"""FastAPI Gateway for PrivateDocs-LLM platform."""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, HTTPException, Request, Response, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response as RawResponse
from fastapi.responses import StreamingResponse

from privatedocs.api.auth import get_current_user_role, require_admin_role
from privatedocs.api.metrics import (
    ASK_LATENCY_SECONDS,
    ASK_REQUESTS_TOTAL,
    LLM_TOKENS_GENERATED_TOTAL,
    LLM_TOKENS_PROMPT_TOTAL,
    LLM_TTFT_SECONDS,
    RETRIEVAL_LATENCY_SECONDS,
    get_prometheus_metrics,
)
from privatedocs.api.rate_limit import check_rate_limit
from privatedocs.api.schemas import (
    AskRequest,
    AskResponse,
    DocumentInfo,
    HealthCheckResponse,
    IngestResponse,
    ReadinessResponse,
    SourceCitation,
)
from privatedocs.config import get_settings
from privatedocs.generation.llm_client import LLMClient
from privatedocs.generation.prompts import build_rag_prompt
from privatedocs.ingestion.chunking import chunk_loaded_documents
from privatedocs.ingestion.cleaning import clean_documents
from privatedocs.ingestion.loaders import load_document
from privatedocs.retrieval.bm25 import BM25Index
from privatedocs.retrieval.embedder import CPUTextEmbedder
from privatedocs.retrieval.hybrid import hybrid_search
from privatedocs.retrieval.reranker import CrossEncoderReranker
from privatedocs.retrieval.vector_store import SearchResult, VectorStore

# Configure privacy-conscious logging: NEVER log full query or text at INFO level
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("privatedocs.gateway")

# Global application services container
class AppServices:
    vector_store: VectorStore
    embedder: CPUTextEmbedder
    bm25: BM25Index
    reranker: CrossEncoderReranker
    llm_client: LLMClient


services = AppServices()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize platform dependencies and models at startup if not already injected."""
    settings = get_settings()
    logger.info("Initializing PrivateDocs-LLM core services (environment=%s)...", settings.ENVIRONMENT)

    if not hasattr(services, "embedder") or services.embedder is None:
        services.embedder = CPUTextEmbedder(
            model_name=settings.EMBEDDING_MODEL_NAME,
            device=settings.EMBEDDING_DEVICE,
        )
    if not hasattr(services, "vector_store") or services.vector_store is None:
        services.vector_store = VectorStore(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
            in_memory=settings.QDRANT_IN_MEMORY,
        )
    if not hasattr(services, "bm25") or services.bm25 is None:
        services.bm25 = BM25Index()
        bm25_cache = Path("data/bm25_index.pkl")
        if bm25_cache.exists():
            services.bm25.load(bm25_cache)

    if not hasattr(services, "reranker") or services.reranker is None:
        services.reranker = CrossEncoderReranker(
            enabled=settings.RERANK_ENABLED,
            model_name=settings.RERANK_MODEL,
        )
    if not hasattr(services, "llm_client") or services.llm_client is None:
        services.llm_client = LLMClient(
            base_url=settings.LLM_BASE_URL,
            api_key=settings.LLM_API_KEY,
            model=settings.LLM_MODEL,
            timeout_seconds=settings.LLM_TIMEOUT_SECONDS,
            max_retries=settings.LLM_MAX_RETRIES,
        )

    logger.info("Services initialized. Gateway is ready.")
    yield
    logger.info("Shutting down PrivateDocs-LLM services...")


app = FastAPI(
    title="PrivateDocs-LLM API Gateway",
    version="0.1.0",
    description="Confidential self-hosted LLM & RAG platform benchmarked on AMD MI300X",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def privacy_access_log_middleware(request: Request, call_next):
    """Privacy-preserving request logging: logs IDs, latency, and status, NOT payload contents."""
    start_time = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start_time) * 1000.0

    # Never log query or request body at INFO
    logger.info(
        "Request: method=%s path=%s status=%d duration=%.1fms client_ip=%s",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
        request.client.host if request.client else "unknown",
    )
    return response


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------

@app.get("/healthz", response_model=HealthCheckResponse, tags=["Health"])
async def healthz() -> HealthCheckResponse:
    """Liveness probe."""
    return HealthCheckResponse(status="healthy", version="0.1.0")


@app.get("/readyz", response_model=ReadinessResponse, tags=["Health"])
async def readyz(response: Response) -> ReadinessResponse:
    """Readiness probe checking Qdrant and LLM connectivity."""
    qdrant_ok = services.vector_store.check_health()
    llm_ok = services.llm_client.check_health()
    is_ready = qdrant_ok and llm_ok

    if not is_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return ReadinessResponse(
        ready=is_ready,
        qdrant_healthy=qdrant_ok,
        llm_healthy=llm_ok,
        details={
            "qdrant": "ok" if qdrant_ok else "unreachable",
            "llm": "ok" if llm_ok else "unreachable",
        },
    )


@app.get("/metrics", tags=["Observability"])
async def metrics() -> RawResponse:
    """Prometheus metrics scrape target."""
    data, content_type = get_prometheus_metrics()
    return RawResponse(content=data, media_type=content_type)


@app.get(
    "/v1/documents",
    response_model=list[DocumentInfo],
    dependencies=[Depends(get_current_user_role)],
    tags=["Documents"],
)
async def list_documents() -> list[DocumentInfo]:
    """List indexed documents and chunk statistics."""
    docs = services.vector_store.list_documents()
    return [DocumentInfo(**d) for d in docs]


def _retrieve_chunks(
    question: str,
    top_k: int,
    mode: str,
    filter_doc_id: str | None = None,
) -> tuple[list[SearchResult], float]:
    """Execute dense, sparse, or hybrid retrieval and measure retrieval latency."""
    t0 = time.perf_counter()

    if mode == "dense":
        query_vec = services.embedder.embed_query(question)
        hits = services.vector_store.search(query_vec, top_k=top_k, filter_doc_id=filter_doc_id)
    elif mode == "bm25":
        hits = services.bm25.search(question, top_k=top_k, filter_doc_id=filter_doc_id)
    else:  # default 'hybrid'
        query_vec = services.embedder.embed_query(question)
        dense_hits = services.vector_store.search(query_vec, top_k=top_k * 2, filter_doc_id=filter_doc_id)
        bm25_hits = services.bm25.search(question, top_k=top_k * 2, filter_doc_id=filter_doc_id)
        settings = get_settings()
        hits = hybrid_search(dense_hits, bm25_hits, rrf_k=settings.RRF_K, top_k=top_k)

    # Apply optional cross-encoder reranker
    if services.reranker.enabled:
        hits = services.reranker.rerank(question, hits, top_k=top_k)

    retrieval_sec = time.perf_counter() - t0
    RETRIEVAL_LATENCY_SECONDS.observe(retrieval_sec)
    return hits, retrieval_sec


@app.post(
    "/v1/ask",
    response_model=AskResponse,
    dependencies=[Depends(get_current_user_role), Depends(check_rate_limit)],
    tags=["Query"],
)
async def ask(payload: AskRequest) -> Any:
    """Ask a question against indexed documents with source citations."""
    start_time = time.perf_counter()
    settings = get_settings()
    mode = payload.mode or settings.RETRIEVAL_MODE
    filter_doc = payload.doc_id or payload.filter_doc_id

    # 1. Retrieve relevant chunks
    hits, _ = _retrieve_chunks(payload.question, payload.top_k, mode, filter_doc_id=filter_doc)

    chunk_dicts = [
        {
            "chunk_id": h.chunk_id,
            "doc_id": h.doc_id,
            "source": h.source,
            "page": h.page,
            "section": h.section,
            "content": h.content,
        }
        for h in hits
    ]

    # 2. Build prompt
    messages = build_rag_prompt(payload.question, chunk_dicts)

    # 3. Stream or Generate
    if payload.stream:
        def stream_generator():
            try:
                # First send citations event
                citations = [
                    {
                        "chunk_id": h.chunk_id,
                        "doc_id": h.doc_id,
                        "source": h.source,
                        "page": h.page,
                        "section": h.section,
                        "score": round(h.score, 4),
                        "snippet": h.content[:200],
                    }
                    for h in hits
                ]
                yield f"event: citations\ndata: {json.dumps(citations)}\n\n"

                # Stream tokens
                for delta, ttft in services.llm_client.stream_answer(
                    messages=messages,
                    temperature=settings.LLM_TEMPERATURE,
                    max_tokens=settings.LLM_MAX_TOKENS,
                ):
                    if ttft is not None:
                        LLM_TTFT_SECONDS.observe(ttft)
                    yield f"event: token\ndata: {json.dumps({'delta': delta})}\n\n"

                yield "event: done\ndata: {}\n\n"
                ASK_REQUESTS_TOTAL.labels(status="200").inc()
            except Exception as exc:
                ASK_REQUESTS_TOTAL.labels(status="500").inc()
                logger.error("Streaming generation failure: %s", exc)
                yield f"event: error\ndata: {json.dumps({'error': str(exc)})}\n\n"

        return StreamingResponse(stream_generator(), media_type="text/event-stream")

    try:
        answer, usage = services.llm_client.generate_answer(
            messages=messages,
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.LLM_MAX_TOKENS,
        )

        total_latency_sec = time.perf_counter() - start_time
        ASK_LATENCY_SECONDS.observe(total_latency_sec)
        ASK_REQUESTS_TOTAL.labels(status="200").inc()
        LLM_TOKENS_GENERATED_TOTAL.inc(usage.completion_tokens)
        LLM_TOKENS_PROMPT_TOTAL.inc(usage.prompt_tokens)

        sources = [
            SourceCitation(
                chunk_id=h.chunk_id,
                doc_id=h.doc_id,
                source=h.source,
                page=h.page,
                section=h.section,
                score=h.score,
                snippet=h.content[:300] + ("..." if len(h.content) > 300 else ""),
                metadata=h.metadata,
            )
            for h in hits
        ]

        return AskResponse(
            answer=answer,
            sources=sources,
            latency_ms=round(total_latency_sec * 1000.0, 2),
            tokens=usage.total_tokens,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
        )

    except Exception as exc:
        ASK_REQUESTS_TOTAL.labels(status="500").inc()
        logger.error("Failed handling /v1/ask: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Inference error: {exc}",
        ) from exc


@app.post(
    "/v1/ingest",
    response_model=IngestResponse,
    dependencies=[Depends(require_admin_role)],
    tags=["Ingestion"],
)
async def ingest_document(
    file: UploadFile = File(...),
) -> IngestResponse:
    """Upload and index a document (admin role required)."""
    settings = get_settings()
    filename = file.filename or "uploaded_file"
    suffix = Path(filename).suffix

    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    try:
        doc_id = Path(filename).stem
        docs = load_document(tmp_path, doc_id=doc_id)
        cleaned = clean_documents(docs)
        chunks = chunk_loaded_documents(
            cleaned,
            strategy=settings.CHUNKING_STRATEGY,
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
        )

        texts = [c.content for c in chunks]
        embeddings = services.embedder.embed_texts(texts)
        count = services.vector_store.upsert_chunks(chunks, embeddings)
        services.bm25.index_chunks(chunks)

        logger.info("Admin ingested document '%s' with %d chunks.", doc_id, count)
        return IngestResponse(doc_id=doc_id, chunks_indexed=count)
    finally:
        if tmp_path.exists():
            os.remove(tmp_path)
