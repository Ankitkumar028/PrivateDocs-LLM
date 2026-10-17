"""Text cleaning, whitespace normalization, and empty page detection."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from privatedocs.ingestion.loaders import LoadedDocument

logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """Normalize whitespace, remove recurrent headers/footers, and clean control chars."""
    if not text:
        return ""

    # Replace null bytes and non-printable control characters (except newline, tab)
    cleaned = re.sub(r"[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]", "", text)

    # Normalize horizontal whitespace (spaces, tabs)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)

    # Collapse multiple consecutive blank lines to at most two
    cleaned = re.sub(r"\n\s*\n\s*\n+", "\n\n", cleaned)

    # Strip repeated page header/footer patterns like "Page X of Y" or standalone digits
    cleaned = re.sub(r"(?i)\bpage\s+\d+\s+of\s+\d+\b", "", cleaned)

    return cleaned.strip()


def clean_documents(docs: list[LoadedDocument], min_char_length: int = 20) -> list[LoadedDocument]:
    """Clean document contents and log pages producing near-empty text."""
    cleaned_docs: list[LoadedDocument] = []

    for doc in docs:
        cleaned_content = clean_text(doc.content)
        if len(cleaned_content) < min_char_length:
            logger.warning(
                "Document parse warning: doc_id='%s' source='%s' page=%d produced near-empty text (%d chars). "
                "Possible scan, image-only, or parse failure.",
                doc.doc_id,
                doc.source,
                doc.page,
                len(cleaned_content),
            )

        if cleaned_content:  # Retain only non-empty documents
            doc.content = cleaned_content
            cleaned_docs.append(doc)

    return cleaned_docs
