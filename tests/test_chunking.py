"""Unit tests for document chunking strategies and deterministic IDs."""

from privatedocs.ingestion.chunking import (
    chunk_document,
    chunk_fixed,
    chunk_loaded_documents,
    chunk_recursive,
    chunk_sentence_window,
    compute_chunk_id,
)
from privatedocs.ingestion.loaders import LoadedDocument


def test_compute_chunk_id_deterministic():
    id1 = compute_chunk_id("docA", 1, 0, "Some confidential paragraph.")
    id2 = compute_chunk_id("docA", 1, 0, "Some confidential paragraph.")
    id3 = compute_chunk_id("docA", 1, 0, "Different confidential paragraph.")

    assert id1 == id2, "Identical content and coordinates must produce identical chunk_id"
    assert id1 != id3, "Different content must produce different chunk_id"
    assert len(id1) == 36


def test_chunk_fixed_overlap():
    words = [f"word{i}" for i in range(100)]
    text = " ".join(words)

    chunks = chunk_fixed(text, chunk_size=30, chunk_overlap=10)
    assert len(chunks) == 5

    # Check overlap between first and second chunk
    chunk0_words = chunks[0].split()
    chunk1_words = chunks[1].split()
    overlap_words = set(chunk0_words[-10:]).intersection(set(chunk1_words[:10]))
    assert len(overlap_words) == 10


def test_chunk_recursive():
    text = (
        "Section 1: Overview\n\n"
        "This is paragraph one detailing company revenue and expenses.\n\n"
        "Section 2: Architecture\n\n"
        "This is paragraph two detailing AMD MI300X memory bandwidth and ROCm stack."
    )
    chunks = chunk_recursive(text, chunk_size=20, chunk_overlap=5)
    assert len(chunks) >= 2
    for c in chunks:
        assert len(c.strip()) > 0


def test_chunk_sentence_window():
    text = (
        "Sentence one describes revenue. "
        "Sentence two describes EBITDA margins. "
        "Sentence three highlights capital expenditures. "
        "Sentence four covers risk factors."
    )
    windows = chunk_sentence_window(text, window_size=2, step_size=1)
    assert len(windows) >= 3
    assert "Sentence one" in windows[0]
    assert "Sentence two" in windows[0]
    assert "Sentence two" in windows[1]
    assert "Sentence three" in windows[1]


def test_chunk_document_metadata():
    text = "PrivateDocs test content for metadata verification."
    chunks = chunk_document(
        doc_id="sec_doc_99",
        text=text,
        source="filing.pdf",
        page=4,
        section="item_1a",
        strategy="fixed",
    )
    assert len(chunks) == 1
    c = chunks[0]
    assert c.doc_id == "sec_doc_99"
    assert c.source == "filing.pdf"
    assert c.page == 4
    assert c.section == "item_1a"
    assert "char_length" in c.metadata


def test_chunk_loaded_documents():
    docs = [
        LoadedDocument(doc_id="d1", content="Page one content.", source="d1.txt", page=1),
        LoadedDocument(doc_id="d1", content="Page two content.", source="d1.txt", page=2),
    ]
    chunks = chunk_loaded_documents(docs, strategy="fixed", chunk_size=50, chunk_overlap=5)
    assert len(chunks) == 2
    assert chunks[0].page == 1
    assert chunks[1].page == 2
