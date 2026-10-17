"""Ingest the real SEC and regulatory corpus, measuring performance and verifying loader integrity."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
import tracemalloc
from pathlib import Path

from privatedocs.config import get_settings
from privatedocs.ingestion.chunking import chunk_loaded_documents
from privatedocs.ingestion.cleaning import clean_documents
from privatedocs.ingestion.loaders import load_document
from privatedocs.retrieval.bm25 import BM25Index
from privatedocs.retrieval.embedder import CPUTextEmbedder
from privatedocs.retrieval.vector_store import VectorStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ingest_real_corpus")


def get_working_set_mb() -> float:
    """Get process resident memory (WorkingSet64) on Windows or Unix."""
    try:
        pid = os.getpid()
        cmd = f'powershell -Command "(Get-Process -Id {pid}).WorkingSet64"'
        out = subprocess.check_output(cmd, shell=True).decode().strip()
        return round(int(out) / (1024 * 1024), 2)
    except Exception:
        return 0.0


def main() -> None:
    tracemalloc.start()
    start_time = time.perf_counter()
    initial_rss = get_working_set_mb()

    settings = get_settings()
    data_dir = Path("data/raw_docs")
    if not data_dir.exists():
        raise FileNotFoundError(f"Raw docs directory not found at {data_dir}")

    files = sorted([f for f in data_dir.iterdir() if f.suffix.lower() in (".html", ".htm", ".pdf")])
    if not files:
        raise ValueError(f"No documents found in {data_dir}")

    logger.info("Found %d documents to ingest from %s", len(files), data_dir)

    # Initialize storage & models
    vector_store = VectorStore(
        collection_name=settings.QDRANT_COLLECTION_NAME,
        host=settings.QDRANT_HOST,
        port=settings.QDRANT_PORT,
        url=settings.QDRANT_URL,
        api_key=settings.QDRANT_API_KEY,
        in_memory=settings.QDRANT_IN_MEMORY,
    )
    embedder = CPUTextEmbedder(
        model_name=settings.EMBEDDING_MODEL_NAME,
        device=settings.EMBEDDING_DEVICE,
    )
    bm25 = BM25Index()

    total_docs = len(files)
    total_sections_pages = 0
    total_chunks = 0
    empty_or_garbled: list[dict[str, str]] = []
    file_summaries: list[dict[str, object]] = []

    for idx, file_path in enumerate(files, start=1):
        file_start = time.perf_counter()
        logger.info("[%d/%d] Loading %s...", idx, total_docs, file_path.name)

        loaded_docs = load_document(file_path)
        if not loaded_docs:
            logger.warning("File %s produced 0 loaded documents!", file_path.name)
            empty_or_garbled.append({"file": file_path.name, "issue": "0 documents loaded"})
            continue

        total_sections_pages += len(loaded_docs)

        # Audit each loaded section/page
        for d in loaded_docs:
            content_stripped = d.content.strip()
            if len(content_stripped) == 0:
                empty_or_garbled.append(
                    {"file": file_path.name, "section_or_page": d.section or f"p.{d.page}", "issue": "Empty content"}
                )
            elif len(content_stripped) < 20 and not any(kw in (d.section or "").lower() for kw in ("reserve", "mine")):
                empty_or_garbled.append(
                    {
                        "file": file_path.name,
                        "section_or_page": d.section or f"p.{d.page}",
                        "issue": f"Trivial/garbled content ({len(content_stripped)} chars): {content_stripped!r}",
                    }
                )

        cleaned_docs = clean_documents(loaded_docs)
        chunks = chunk_loaded_documents(
            cleaned_docs,
            strategy=settings.CHUNKING_STRATEGY,
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
        )

        logger.info(
            "  -> %s: %d sections/pages, %d chunks",
            file_path.name,
            len(loaded_docs),
            len(chunks),
        )

        # Embed and upsert in batches
        BATCH_SIZE = 128
        for b_start in range(0, len(chunks), BATCH_SIZE):
            chunk_batch = chunks[b_start : b_start + BATCH_SIZE]
            texts = [c.content for c in chunk_batch]
            embeddings = embedder.embed_texts(texts)
            vector_store.upsert_chunks(chunk_batch, embeddings)

        # Update BM25
        bm25.index_chunks(chunks)
        total_chunks += len(chunks)

        file_elapsed = round(time.perf_counter() - file_start, 2)
        file_summaries.append(
            {
                "file": file_path.name,
                "sections_or_pages": len(loaded_docs),
                "chunks": len(chunks),
                "elapsed_seconds": file_elapsed,
            }
        )

    # Persist BM25 Index
    bm25_path = Path("data/bm25_index.pkl")
    bm25.save(bm25_path)
    bm25_size_bytes = bm25_path.stat().st_size
    bm25_size_mb = round(bm25_size_bytes / (1024 * 1024), 2)

    total_time = round(time.perf_counter() - start_time, 2)
    final_rss = get_working_set_mb()
    curr_traced, peak_traced = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    report = {
        "status": "success",
        "document_count": total_docs,
        "total_sections_or_pages": total_sections_pages,
        "chunk_count": total_chunks,
        "ingestion_time_seconds": total_time,
        "bm25_index_size_bytes": bm25_size_bytes,
        "bm25_index_size_mb": bm25_size_mb,
        "memory_use": {
            "initial_process_rss_mb": initial_rss,
            "final_process_rss_mb": final_rss,
            "peak_python_heap_mb": round(peak_traced / (1024 * 1024), 2),
        },
        "empty_or_garbled_count": len(empty_or_garbled),
        "empty_or_garbled": empty_or_garbled,
        "file_details": file_summaries,
    }

    report_path = Path("data/ingestion_report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print("REAL CORPUS INGESTION REPORT")
    print("=" * 60)
    print(f"Document Count:            {report['document_count']}")
    print(f"Total Sections / Pages:    {report['total_sections_or_pages']}")
    print(f"Total Chunk Count:         {report['chunk_count']:,}")
    print(f"Ingestion Time:            {report['ingestion_time_seconds']} seconds")
    print(f"BM25 Index Size:           {report['bm25_index_size_mb']} MB ({report['bm25_index_size_bytes']:,} bytes)")
    print(f"Initial Process RSS:       {report['memory_use']['initial_process_rss_mb']} MB")
    print(f"Final Process RSS:         {report['memory_use']['final_process_rss_mb']} MB")
    print(f"Peak Python Heap:          {report['memory_use']['peak_python_heap_mb']} MB")
    print(f"Empty/Garbled Issues:      {report['empty_or_garbled_count']}")
    if empty_or_garbled:
        print("\nEmpty or Garbled Sections/Pages:")
        for item in empty_or_garbled:
            print(f"  - {item}")
    else:
        print("\nZero empty or garbled sections/pages detected across all documents!")
    print("=" * 60)


if __name__ == "__main__":
    main()
