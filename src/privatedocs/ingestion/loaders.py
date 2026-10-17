"""Document loaders for PDF, HTML, TXT, and Markdown files."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup
from pypdf import PdfReader

logger = logging.getLogger(__name__)


@dataclass
class LoadedDocument:
    """Represents an ingested document page or section with metadata."""

    doc_id: str
    content: str
    source: str
    page: int = 1
    section: str = "main"
    metadata: dict[str, Any] = field(default_factory=dict)


def load_pdf(file_path: Path, doc_id: str | None = None) -> list[LoadedDocument]:
    """Extract pages from a PDF file preserving page numbers and metadata."""
    path = Path(file_path)
    document_id = doc_id or path.stem
    documents: list[LoadedDocument] = []

    try:
        reader = PdfReader(str(path))
        for page_idx, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            documents.append(
                LoadedDocument(
                    doc_id=document_id,
                    content=text,
                    source=path.name,
                    page=page_idx,
                    section="page",
                    metadata={"total_pages": len(reader.pages)},
                )
            )
    except Exception as exc:
        logger.error("Failed to parse PDF %s: %s", path, exc)
        raise

    return documents


def load_html(file_path: Path, doc_id: str | None = None) -> list[LoadedDocument]:
    """Parse HTML documents, stripping script/style tags."""
    path = Path(file_path)
    document_id = doc_id or path.stem

    try:
        html_content = path.read_text(encoding="utf-8", errors="replace")
        soup = BeautifulSoup(html_content, "html.parser")

        # Strip scripts, styles, navigation, footer
        for element in soup(["script", "style", "nav", "footer", "header"]):
            element.decompose()

        text = soup.get_text(separator="\n")
        title = soup.title.string if soup.title and soup.title.string else document_id

        return [
            LoadedDocument(
                doc_id=document_id,
                content=text,
                source=path.name,
                page=1,
                section="html_body",
                metadata={"title": title.strip()},
            )
        ]
    except Exception as exc:
        logger.error("Failed to parse HTML %s: %s", path, exc)
        raise


def load_text(file_path: Path, doc_id: str | None = None) -> list[LoadedDocument]:
    """Load plain text or Markdown file."""
    path = Path(file_path)
    document_id = doc_id or path.stem

    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        return [
            LoadedDocument(
                doc_id=document_id,
                content=text,
                source=path.name,
                page=1,
                section="full_text",
                metadata={"file_size": path.stat().st_size},
            )
        ]
    except Exception as exc:
        logger.error("Failed to parse text file %s: %s", path, exc)
        raise


def load_document(file_path: Path | str, doc_id: str | None = None) -> list[LoadedDocument]:
    """Universal loader dispatching by file extension."""
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return load_pdf(path, doc_id)
    elif suffix in (".html", ".htm"):
        return load_html(path, doc_id)
    elif suffix in (".txt", ".md", ".markdown"):
        return load_text(path, doc_id)
    else:
        # Default fallback to text loader
        logger.info("Unrecognized extension %s; attempting text loader for %s", suffix, path)
        return load_text(path, doc_id)
