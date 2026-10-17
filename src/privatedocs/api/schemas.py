"""Pydantic request and response schemas for FastAPI endpoints."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    """Payload for POST /v1/ask."""

    question: str = Field(..., min_length=1, description="User question to answer against documents")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of context chunks to retrieve")
    mode: Literal["dense", "bm25", "hybrid"] | None = Field(
        default=None,
        description="Retrieval mode override (default: configured RETRIEVAL_MODE)",
    )
    doc_id: str | None = Field(
        default=None,
        description="Optional doc_id or company filter for targeted retrieval",
    )
    filter_doc_id: str | None = Field(
        default=None,
        description="Alias for doc_id filter",
    )
    stream: bool = Field(default=False, description="Whether to stream response tokens via SSE")


class SourceCitation(BaseModel):
    """Citation of retrieved chunk."""

    chunk_id: str
    doc_id: str
    source: str
    page: int
    section: str
    score: float
    snippet: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class AskResponse(BaseModel):
    """Response payload for POST /v1/ask."""

    answer: str
    sources: list[SourceCitation]
    latency_ms: float
    tokens: int  # Backwards compatibility alias for total_tokens
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    ttft_ms: float | None = None


class DocumentInfo(BaseModel):
    """Summary of indexed document."""

    doc_id: str
    source: str
    chunk_count: int
    total_pages: int


class IngestResponse(BaseModel):
    """Response payload for POST /v1/ingest."""

    doc_id: str
    chunks_indexed: int
    status: str = "success"


class HealthCheckResponse(BaseModel):
    """Response payload for /healthz."""

    status: str = "healthy"
    version: str = "0.1.0"


class ReadinessResponse(BaseModel):
    """Response payload for /readyz."""

    ready: bool
    qdrant_healthy: bool
    llm_healthy: bool
    details: dict[str, str] = Field(default_factory=dict)
