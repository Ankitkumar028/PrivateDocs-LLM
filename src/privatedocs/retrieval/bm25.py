"""BM25 sparse keyword index wrapper using rank-bm25."""

from __future__ import annotations

import logging
import pickle
import re
from pathlib import Path
from typing import TYPE_CHECKING

from rank_bm25 import BM25Okapi

from privatedocs.retrieval.vector_store import SearchResult

if TYPE_CHECKING:
    from privatedocs.ingestion.chunking import DocumentChunk

logger = logging.getLogger(__name__)


def tokenize_text(text: str) -> list[str]:
    """Tokenize text into lowercase alphanumeric tokens."""
    return re.findall(r"\b\w+\b", text.lower())


class BM25Index:
    """In-memory keyword index using BM25Okapi for sparse lexical retrieval."""

    def __init__(self) -> None:
        self.chunks: list[DocumentChunk] = []
        self.corpus_tokens: list[list[str]] = []
        self.bm25: BM25Okapi | None = None
        self._chunk_map: dict[str, DocumentChunk] = {}

    def index_chunks(self, chunks: list[DocumentChunk]) -> int:
        """Add chunks to the index (idempotent by chunk_id) and build BM25Okapi."""
        for chunk in chunks:
            if chunk.chunk_id not in self._chunk_map:
                self.chunks.append(chunk)
                self._chunk_map[chunk.chunk_id] = chunk
                self.corpus_tokens.append(tokenize_text(chunk.content))

        if self.corpus_tokens:
            self.bm25 = BM25Okapi(self.corpus_tokens)
            logger.info("Indexed %d total chunks in BM25 index.", len(self.chunks))

        return len(chunks)

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Perform BM25 search and return top-k scored SearchResult objects."""
        if not self.bm25 or not self.chunks:
            return []

        tokens = tokenize_text(query)
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)

        # Pair scores with chunk objects
        scored_pairs = sorted(
            enumerate(scores),
            key=lambda x: x[1],
            reverse=True,
        )

        results: list[SearchResult] = []
        for idx, score in scored_pairs[:top_k]:
            if score <= 0.0:
                continue
            chunk = self.chunks[idx]
            results.append(
                SearchResult(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    content=chunk.content,
                    source=chunk.source,
                    page=chunk.page,
                    section=chunk.section,
                    score=float(score),
                    metadata=chunk.metadata,
                )
            )

        return results

    def save(self, file_path: Path | str) -> None:
        """Persist BM25 index and chunks to disk."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"chunks": self.chunks, "corpus_tokens": self.corpus_tokens}, f)

    def load(self, file_path: Path | str) -> bool:
        """Load BM25 index and chunks from disk."""
        path = Path(file_path)
        if not path.exists():
            return False
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.chunks = data.get("chunks", [])
        self.corpus_tokens = data.get("corpus_tokens", [])
        self._chunk_map = {c.chunk_id: c for c in self.chunks}
        if self.corpus_tokens:
            self.bm25 = BM25Okapi(self.corpus_tokens)
        return True
