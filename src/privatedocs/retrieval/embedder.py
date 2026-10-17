"""CPU-based sentence-transformers wrapper for dense vector representations."""

from __future__ import annotations

import hashlib
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class CPUTextEmbedder:
    """Generates dense vector embeddings using sentence-transformers strictly on CPU."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        device: str = "cpu",
        dimension: int = 384,
        fallback_mode: bool = False,
    ) -> None:
        import os

        self.model_name = model_name
        self.device = device
        self.dimension = dimension
        self._model: SentenceTransformer | None = None
        self._fallback_mode = fallback_mode or os.getenv("FAST_EMBEDDER", "false").lower() in ("true", "1")

    def _load_model(self) -> None:
        """Lazy load sentence-transformers model on CPU."""
        if self._model is not None or self._fallback_mode:
            return

        try:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading embedding model '%s' on %s...", self.model_name, self.device)
            self._model = SentenceTransformer(self.model_name, device=self.device)
            if hasattr(self._model, "get_embedding_dimension"):
                dim = self._model.get_embedding_dimension()
            else:
                dim = self._model.get_sentence_embedding_dimension()
            if dim is not None:
                self.dimension = dim
            logger.info("Embedding model loaded successfully (dimension=%d).", self.dimension)
        except Exception as exc:
            logger.warning(
                "Could not load sentence-transformers model '%s' (%s). "
                "Enabling deterministic pseudo-embedding fallback for local dev / offline tests.",
                self.model_name,
                exc,
            )
            self._fallback_mode = True

    def _pseudo_embed(self, text: str) -> list[float]:
        """Generate deterministic pseudo-embedding vector for offline / test environments."""
        vec = [0.0] * self.dimension
        words = text.lower().split()
        for idx, word in enumerate(words):
            h = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16)
            pos = h % self.dimension
            vec[pos] += 1.0 / (idx + 1.0)

        # L2 normalize
        norm = sum(x * x for x in vec) ** 0.5
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of text documents."""
        if not texts:
            return []

        self._load_model()
        if self._model is not None and not self._fallback_mode:
            try:
                embeddings = self._model.encode(
                    texts,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                return [emb.tolist() for emb in embeddings]
            except Exception as exc:
                logger.error("Embedding generation failed, falling back: %s", exc)

        return [self._pseudo_embed(t) for t in texts]

    def embed_query(self, query: str) -> list[float]:
        """Embed a single search query."""
        results = self.embed_texts([query])
        return results[0]
