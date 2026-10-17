"""Pluggable document chunking strategies with deterministic content hashing."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any

from privatedocs.ingestion.loaders import LoadedDocument


@dataclass
class DocumentChunk:
    """A granular chunk of document text ready for vector and keyword indexing."""

    chunk_id: str
    doc_id: str
    content: str
    source: str
    page: int
    section: str = "main"
    chunk_index: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


def compute_chunk_id(doc_id: str, page: int, chunk_index: int, content: str) -> str:
    """Generate a deterministic UUID string for Qdrant and storage idempotency."""
    import uuid

    raw_key = f"{doc_id}:{page}:{chunk_index}:{content.strip()}"
    h = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:32]
    return str(uuid.UUID(hex=h))


def chunk_fixed(
    text: str,
    chunk_size: int = 400,
    chunk_overlap: int = 50,
) -> list[str]:
    """Token/word-based fixed chunking with sliding overlap."""
    words = text.split()
    if not words:
        return []

    if len(words) <= chunk_size:
        return [" ".join(words)]

    chunks: list[str] = []
    step = max(1, chunk_size - chunk_overlap)
    for i in range(0, len(words), step):
        chunk_words = words[i : i + chunk_size]
        chunks.append(" ".join(chunk_words))
        if i + chunk_size >= len(words):
            break

    return chunks


def chunk_recursive(
    text: str,
    chunk_size: int = 400,
    chunk_overlap: int = 50,
    separators: list[str] | None = None,
) -> list[str]:
    """Hierarchical recursive text splitting by paragraphs, sentences, and words."""
    if separators is None:
        separators = ["\n\n", "\n", ". ", "; ", ", ", " "]

    words = text.split()
    if len(words) <= chunk_size:
        return [text] if text.strip() else []

    # Find the highest-priority separator present in text
    sep = ""
    for s in separators:
        if s in text:
            sep = s
            break

    if not sep:
        return chunk_fixed(text, chunk_size, chunk_overlap)

    splits = text.split(sep)
    chunks: list[str] = []
    current_acc: list[str] = []
    current_word_count = 0

    for split in splits:
        split_words = len(split.split())
        if current_word_count + split_words > chunk_size and current_acc:
            chunk_str = sep.join(current_acc).strip()
            if chunk_str:
                chunks.append(chunk_str)

            # Keep overlap if possible
            overlap_words = 0
            overlap_acc: list[str] = []
            for item in reversed(current_acc):
                item_wc = len(item.split())
                if overlap_words + item_wc <= chunk_overlap:
                    overlap_acc.insert(0, item)
                    overlap_words += item_wc
                else:
                    break
            current_acc = overlap_acc
            current_word_count = overlap_words

        current_acc.append(split)
        current_word_count += split_words

    if current_acc:
        final_chunk = sep.join(current_acc).strip()
        if final_chunk:
            chunks.append(final_chunk)

    return chunks


def chunk_sentence_window(
    text: str,
    window_size: int = 3,
    step_size: int = 1,
) -> list[str]:
    """Splits into sentences and constructs sliding sentence windows."""
    # Regex split on sentence boundaries
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return []

    if len(sentences) <= window_size:
        return [" ".join(sentences)]

    chunks: list[str] = []
    for i in range(0, len(sentences), step_size):
        window = sentences[i : i + window_size]
        chunks.append(" ".join(window))
        if i + window_size >= len(sentences):
            break

    return chunks


def chunk_document(
    doc_id: str,
    text: str,
    source: str,
    page: int = 1,
    section: str = "main",
    strategy: str = "fixed",
    chunk_size: int = 400,
    chunk_overlap: int = 50,
) -> list[DocumentChunk]:
    """Apply chosen chunking strategy to document text and return DocumentChunk objects."""
    if strategy == "recursive":
        raw_chunks = chunk_recursive(text, chunk_size, chunk_overlap)
    elif strategy == "sentence_window":
        raw_chunks = chunk_sentence_window(text, window_size=max(2, chunk_size // 100))
    else:  # default 'fixed'
        raw_chunks = chunk_fixed(text, chunk_size, chunk_overlap)

    chunks: list[DocumentChunk] = []
    for idx, content in enumerate(raw_chunks):
        if not content.strip():
            continue
        chunk_id = compute_chunk_id(doc_id, page, idx, content)
        chunks.append(
            DocumentChunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                content=content,
                source=source,
                page=page,
                section=section,
                chunk_index=idx,
                metadata={
                    "char_length": len(content),
                    "word_count": len(content.split()),
                    "strategy": strategy,
                },
            )
        )

    return chunks


def chunk_loaded_documents(
    docs: list[LoadedDocument],
    strategy: str = "fixed",
    chunk_size: int = 400,
    chunk_overlap: int = 50,
) -> list[DocumentChunk]:
    """Transform list of LoadedDocument pages into indexed DocumentChunk objects."""
    all_chunks: list[DocumentChunk] = []
    for doc in docs:
        chunks = chunk_document(
            doc_id=doc.doc_id,
            text=doc.content,
            source=doc.source,
            page=doc.page,
            section=doc.section,
            strategy=strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        all_chunks.extend(chunks)
    return all_chunks
