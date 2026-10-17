"""Qdrant client wrapper for dense vector persistence and similarity retrieval."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

if TYPE_CHECKING:
    from privatedocs.ingestion.chunking import DocumentChunk

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Standardized retrieval search result across dense and sparse retrievers."""

    chunk_id: str
    doc_id: str
    content: str
    source: str
    page: int
    section: str
    score: float
    metadata: dict[str, Any]


class VectorStore:
    """Encapsulates Qdrant operations with support for in-memory or remote server."""

    def __init__(
        self,
        collection_name: str = "private_documents",
        host: str = "localhost",
        port: int = 6333,
        url: str | None = None,
        api_key: str | None = None,
        in_memory: bool = False,
        dimension: int = 384,
    ) -> None:
        self.collection_name = collection_name
        self.dimension = dimension

        if in_memory:
            logger.info("Initializing in-memory Qdrant client...")
            self.client = QdrantClient(location=":memory:")
        elif url:
            self.client = QdrantClient(url=url, api_key=api_key, timeout=2)
        else:
            self.client = QdrantClient(host=host, port=port, api_key=api_key, timeout=2)

        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Create Qdrant collection if not already present."""
        try:
            collections = self.client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)
            if not exists:
                logger.info("Creating Qdrant collection '%s' (dim=%d)...", self.collection_name, self.dimension)
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=qmodels.VectorParams(
                        size=self.dimension,
                        distance=qmodels.Distance.COSINE,
                    ),
                )
        except Exception as exc:
            logger.warning("Could not check/create Qdrant collection '%s': %s", self.collection_name, exc)

    def check_health(self) -> bool:
        """Verify Qdrant server responsiveness."""
        try:
            self.client.get_collections()
            return True
        except Exception as exc:
            logger.warning("Qdrant health check failed: %s", exc)
            return False

    def upsert_chunks(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
    ) -> int:
        """Upsert document chunks and vectors idempotently by deterministic chunk ID."""
        if not chunks:
            return 0

        points = []
        for chunk, vector in zip(chunks, embeddings, strict=True):
            points.append(
                qmodels.PointStruct(
                    id=chunk.chunk_id,
                    vector=vector,
                    payload={
                        "chunk_id": chunk.chunk_id,
                        "doc_id": chunk.doc_id,
                        "content": chunk.content,
                        "source": chunk.source,
                        "page": chunk.page,
                        "section": chunk.section,
                        "chunk_index": chunk.chunk_index,
                        "metadata": chunk.metadata,
                    },
                )
            )

        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True,
            )
        except Exception as exc:
            logger.warning("Could not upsert points to Qdrant (%s). Vector index may be offline.", exc)
        return len(points)

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        filter_doc_id: str | None = None,
    ) -> list[SearchResult]:
        """Perform dense cosine similarity search."""
        query_filter = None
        if filter_doc_id:
            query_filter = qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="doc_id",
                        match=qmodels.MatchValue(value=filter_doc_id),
                    )
                ]
            )

        try:
            hits = []
            if hasattr(self.client, "query_points"):
                try:
                    response = self.client.query_points(
                        collection_name=self.collection_name,
                        query=query_vector,
                        limit=top_k,
                        query_filter=query_filter,
                    )
                    hits = response.points
                except Exception as query_exc:
                    logger.debug("query_points failed (%s); trying fallback to search()...", query_exc)
                    if hasattr(self.client, "search"):
                        hits = self.client.search(
                            collection_name=self.collection_name,
                            query_vector=query_vector,
                            limit=top_k,
                            query_filter=query_filter,
                        )
            elif hasattr(self.client, "search"):
                hits = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vector,
                    limit=top_k,
                    query_filter=query_filter,
                )
        except Exception as exc:
            logger.error("Qdrant search error: %s", exc)
            return []

        results: list[SearchResult] = []
        for hit in hits:
            payload = hit.payload or {}
            results.append(
                SearchResult(
                    chunk_id=payload.get("chunk_id", str(hit.id)),
                    doc_id=payload.get("doc_id", "unknown"),
                    content=payload.get("content", ""),
                    source=payload.get("source", ""),
                    page=payload.get("page", 1),
                    section=payload.get("section", ""),
                    score=float(hit.score),
                    metadata=payload.get("metadata", {}),
                )
            )
        return results

    def list_documents(self) -> list[dict[str, Any]]:
        """List distinct documents indexed in the vector store."""
        try:
            # Scroll points to extract unique document metadata
            records, _ = self.client.scroll(
                collection_name=self.collection_name,
                limit=1000,
                with_payload=True,
                with_vectors=False,
            )
            doc_map: dict[str, dict[str, Any]] = {}
            for record in records:
                payload = record.payload or {}
                doc_id = payload.get("doc_id")
                if not doc_id:
                    continue
                if doc_id not in doc_map:
                    doc_map[doc_id] = {
                        "doc_id": doc_id,
                        "source": payload.get("source", ""),
                        "chunk_count": 0,
                        "pages": set(),
                    }
                doc_map[doc_id]["chunk_count"] += 1
                doc_map[doc_id]["pages"].add(payload.get("page", 1))

            result = []
            for doc in doc_map.values():
                result.append(
                    {
                        "doc_id": doc["doc_id"],
                        "source": doc["source"],
                        "chunk_count": doc["chunk_count"],
                        "total_pages": len(doc["pages"]),
                    }
                )
            return result
        except Exception as exc:
            logger.error("Failed to list documents from Qdrant: %s", exc)
            return []
