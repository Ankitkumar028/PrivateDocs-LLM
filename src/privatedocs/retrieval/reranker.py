"""Optional cross-encoder reranking behind feature flag."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from privatedocs.retrieval.vector_store import SearchResult

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Optional cross-encoder model to re-score and re-rank candidate chunks."""

    def __init__(
        self,
        enabled: bool = False,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ) -> None:
        self.enabled = enabled
        self.model_name = model_name
        self._model: CrossEncoder | None = None

    def _load_model(self) -> None:
        if not self.enabled or self._model is not None:
            return
        try:
            from sentence_transformers import CrossEncoder

            logger.info("Loading cross-encoder reranker model '%s'...", self.model_name)
            self._model = CrossEncoder(self.model_name)
        except Exception as exc:
            logger.warning("Could not load CrossEncoder model '%s': %s", self.model_name, exc)
            self.enabled = False

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Rerank search results if enabled, else return original results."""
        if not self.enabled or not results:
            return results[:top_k]

        self._load_model()
        if not self._model:
            return results[:top_k]

        pairs = [(query, r.content) for r in results]
        try:
            scores = self._model.predict(pairs)  # type: ignore[arg-type]
            scored = sorted(
                zip(results, scores, strict=True),
                key=lambda x: x[1],
                reverse=True,
            )
            reranked: list[SearchResult] = []
            for item, score in scored[:top_k]:
                reranked.append(
                    SearchResult(
                        chunk_id=item.chunk_id,
                        doc_id=item.doc_id,
                        content=item.content,
                        source=item.source,
                        page=item.page,
                        section=item.section,
                        score=float(score),
                        metadata=dict(item.metadata, rerank_score=float(score)),
                    )
                )
            return reranked
        except Exception as exc:
            logger.error("Cross-encoder reranking failed, returning original order: %s", exc)
            return results[:top_k]
