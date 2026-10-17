"""Document indexer orchestrating loading, cleaning, chunking, and storage."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from privatedocs.config import get_settings
from privatedocs.ingestion.chunking import chunk_loaded_documents
from privatedocs.ingestion.cleaning import clean_documents
from privatedocs.ingestion.loaders import load_document
from privatedocs.retrieval.bm25 import BM25Index
from privatedocs.retrieval.embedder import CPUTextEmbedder
from privatedocs.retrieval.vector_store import VectorStore

if TYPE_CHECKING:
    from privatedocs.ingestion.chunking import DocumentChunk

logger = logging.getLogger(__name__)


class DocumentIndexer:
    """End-to-end indexer coordinating embedding, Qdrant upserts, and BM25 index."""

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        embedder: CPUTextEmbedder | None = None,
        bm25: BM25Index | None = None,
    ) -> None:
        settings = get_settings()
        self.vector_store = vector_store or VectorStore(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
            in_memory=settings.QDRANT_IN_MEMORY,
        )
        self.embedder = embedder or CPUTextEmbedder(
            model_name=settings.EMBEDDING_MODEL_NAME,
            device=settings.EMBEDDING_DEVICE,
        )
        self.bm25 = bm25 or BM25Index()
        self.settings = settings

    def index_chunks(self, chunks: list[DocumentChunk]) -> int:
        """Embed and upsert given chunks into Qdrant and BM25."""
        if not chunks:
            return 0

        # Dense embedding
        texts = [c.content for c in chunks]
        embeddings = self.embedder.embed_texts(texts)

        # Upsert into Qdrant
        count = self.vector_store.upsert_chunks(chunks, embeddings)

        # Update BM25
        self.bm25.index_chunks(chunks)

        # Persist BM25 cache
        cache_path = Path("data/bm25_index.pkl")
        try:
            self.bm25.save(cache_path)
        except Exception as exc:
            logger.warning("Could not persist BM25 cache to %s: %s", cache_path, exc)

        logger.info("Successfully indexed %d chunks (idempotent).", count)
        return count

    def index_file(self, file_path: Path | str, doc_id: str | None = None) -> int:
        """Load, clean, chunk, and index a single document file."""
        path = Path(file_path)
        logger.info("Indexing file: %s (doc_id=%s)", path, doc_id)

        docs = load_document(path, doc_id=doc_id)
        cleaned = clean_documents(docs)
        chunks = chunk_loaded_documents(
            cleaned,
            strategy=self.settings.CHUNKING_STRATEGY,
            chunk_size=self.settings.CHUNK_SIZE,
            chunk_overlap=self.settings.CHUNK_OVERLAP,
        )
        return self.index_chunks(chunks)

    def index_directory(self, dir_path: Path | str) -> int:
        """Scan directory and index all supported documents."""
        path = Path(dir_path)
        if not path.exists():
            logger.warning("Directory does not exist: %s", path)
            return 0

        total_indexed = 0
        supported_exts = {".pdf", ".html", ".htm", ".txt", ".md"}
        for file in path.glob("**/*"):
            if file.is_file() and file.suffix.lower() in supported_exts:
                try:
                    count = self.index_file(file)
                    total_indexed += count
                except Exception as exc:
                    logger.error("Failed indexing %s: %s", file, exc)

        return total_indexed


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Ingest and index documents into PrivateDocs-LLM.")
    parser.add_argument("--path", default="data/sample_docs", help="Path to document file or directory")
    parser.add_argument("--doc-id", default=None, help="Explicit document ID for single file")
    parser.add_argument("--in-memory", action="store_true", help="Use in-memory Qdrant instance")
    args = parser.parse_args()

    target = Path(args.path)
    # Ensure sample dir exists with a demo file if empty
    if not target.exists():
        target.mkdir(parents=True, exist_ok=True)
        demo_file = target / "sample_internal_memo.txt"
        demo_file.write_text(
            "PrivateDocs-LLM confidential infrastructure memo.\n"
            "All model weights and vector embeddings remain strictly local.\n"
            "Self-hosted vLLM on AMD MI300X provides 192GB unified HBM3 memory.\n"
            "This enables serving 70B parameter models at full FP16 or high-throughput FP8.\n",
            encoding="utf-8",
        )
        print(f"Created sample document for indexing at {demo_file}")

    settings = get_settings()
    vstore = VectorStore(
        collection_name=settings.QDRANT_COLLECTION_NAME,
        in_memory=args.in_memory or settings.QDRANT_IN_MEMORY,
        host=settings.QDRANT_HOST,
        port=settings.QDRANT_PORT,
        url=settings.QDRANT_URL,
        api_key=settings.QDRANT_API_KEY,
    )
    indexer = DocumentIndexer(vector_store=vstore)
    if target.is_dir():
        count = indexer.index_directory(target)
    else:
        count = indexer.index_file(target, doc_id=args.doc_id)

    print(f"Indexing complete! Total chunks indexed: {count}")


if __name__ == "__main__":
    main()
