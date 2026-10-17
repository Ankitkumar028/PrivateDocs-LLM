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


SEC_ITEM_TITLES: dict[str, str] = {
    "ITEM 1": "Business",
    "ITEM 1A": "Risk Factors",
    "ITEM 1B": "Unresolved Staff Comments",
    "ITEM 1C": "Cybersecurity",
    "ITEM 2": "Properties",
    "ITEM 3": "Legal Proceedings",
    "ITEM 4": "Mine Safety Disclosures",
    "ITEM 5": "Market for Registrant's Common Equity and Related Stockholder Matters",
    "ITEM 6": "[Reserved]",
    "ITEM 7": "Management's Discussion and Analysis of Financial Condition and Results of Operations",
    "ITEM 7A": "Quantitative and Qualitative Disclosures About Market Risk",
    "ITEM 8": "Financial Statements and Supplementary Data",
    "ITEM 9": "Changes in and Disagreements with Accountants on Accounting and Financial Disclosure",
    "ITEM 9A": "Controls and Procedures",
    "ITEM 9B": "Other Information",
    "ITEM 9C": "Disclosure Regarding Foreign Jurisdictions that Prevent Inspections",
    "ITEM 10": "Directors, Executive Officers and Corporate Governance",
    "ITEM 11": "Executive Compensation",
    "ITEM 12": "Security Ownership of Certain Beneficial Owners and Management",
    "ITEM 13": "Certain Relationships and Related Transactions and Director Independence",
    "ITEM 14": "Principal Accountant Fees and Services",
    "ITEM 15": "Exhibits, Financial Statement Schedules",
    "ITEM 16": "Form 10-K Summary",
}


def load_html(file_path: Path, doc_id: str | None = None) -> list[LoadedDocument]:
    """Parse HTML documents, stripping script/style tags, formatting tables, and extracting section headings."""
    import re

    path = Path(file_path)
    document_id = doc_id or path.stem

    try:
        raw_bytes = path.read_bytes()
        html_content = ""
        for enc in ("utf-8", "cp1252", "latin-1"):
            try:
                html_content = raw_bytes.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        if not html_content:
            html_content = raw_bytes.decode("utf-8", errors="replace")

        soup = BeautifulSoup(html_content, "html.parser")

        # Strip scripts, styles, navigation, footer, header
        for element in soup(["script", "style", "nav", "footer", "header"]):
            element.decompose()

        # Handle inline XBRL tags (decomposing hidden metadata, unwrapping data tags)
        for ix_tag in soup.find_all(re.compile(r"^ix:(?:header|hidden)$")):
            ix_tag.decompose()
        for ix_tag in soup.find_all(re.compile(r"^ix:")):
            ix_tag.unwrap()

        # Format HTML tables: detect inline heading tables (e.g. 2 cells: 'Item 1A.' and 'Risk Factors')
        heading_re = re.compile(r"^ITEM\s+[0-9]{1,2}[A-Z]?\.?$", re.I)
        for table in soup.find_all("table"):
            rows: list[str] = []
            for tr in table.find_all("tr"):
                cells = [
                    re.sub(r"[\xa0\s]+", " ", c.get_text(separator=" ", strip=True)).strip()
                    for c in tr.find_all(["th", "td"])
                ]
                cells = [c for c in cells if c]
                if len(cells) == 2 and heading_re.match(cells[0]) and not cells[1].isdigit():
                    rows.append(f"{cells[0]} - {cells[1]}")
                elif cells:
                    rows.append(" | ".join(cells))
            if rows:
                table_text = "\n" + "\n".join(rows) + "\n"
                table.replace_with(soup.new_string(table_text))

        # Separate block-level elements with newlines to prevent inline tag concatenation
        for tag in soup.find_all(["p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "br", "hr", "li"]):
            tag.append(soup.new_string("\n"))

        title = soup.title.string.strip() if soup.title and soup.title.string else document_id
        raw_text = soup.get_text(separator=" ")
        # Normalize non-breaking and zero-width spaces
        raw_text = re.sub(r"[\xa0\u200b\u200e\u200f]+", " ", raw_text)

        # Detect SEC Item section headings (e.g., ITEM 1, ITEM 1A, ITEM 7, etc.)
        sec_item_pattern = re.compile(
            r"^(ITEM\s+[0-9]{1,2}[A-Z]?)(?:[\.\:\s\-]+([^\|\n\r]{2,120}))?$",
            re.I,
        )

        lines = raw_text.splitlines()
        sections_data: list[tuple[str, str, list[str]]] = []
        curr_item: str | None = None
        curr_title: str | None = None
        curr_lines: list[str] = []

        for line in lines:
            stripped = line.strip()
            # If line is not a table row with pipes, test for SEC Item heading
            m = sec_item_pattern.match(stripped) if "|" not in stripped else None
            if m:
                code = m.group(1).upper().replace("  ", " ")
                title_match = (m.group(2) or "").strip()
                title_clean = re.sub(r"[\.\:\s\-]+$", "", title_match).strip()

                if code != curr_item:
                    if curr_lines:
                        sections_data.append((curr_item or "Cover & General", curr_title or "", curr_lines))
                        curr_lines = []
                    curr_item = code
                    curr_title = title_clean
                    continue
                elif title_clean and not curr_title:
                    curr_title = title_clean

            if stripped:
                curr_lines.append(stripped)

        if curr_lines:
            sections_data.append((curr_item or "Cover & General", curr_title or "", curr_lines))

        # If multiple SEC items were found, produce LoadedDocument for each non-trivial section
        if len(sections_data) >= 2:
            documents: list[LoadedDocument] = []
            for code, s_title, s_lines in sections_data:
                content = "\n".join(s_lines).strip()
                if len(content) < 50 and code != "Cover & General":
                    continue

                canonical = SEC_ITEM_TITLES.get(code)
                if canonical:
                    section_label = f"{code.title()} - {canonical}"
                elif s_title:
                    section_label = f"{code.title()} - {s_title.title()}"
                else:
                    section_label = code.title() if code != "Cover & General" else "Cover & General"

                documents.append(
                    LoadedDocument(
                        doc_id=document_id,
                        content=content,
                        source=path.name,
                        page=0,
                        section=section_label,
                        metadata={
                            "title": title,
                            "has_page": False,
                            "sec_item": code.title() if code != "Cover & General" else "Cover",
                        },
                    )
                )

            if documents:
                logger.info("Parsed %d SEC sections from HTML %s", len(documents), path.name)
                return documents

        # Fallback for non-SEC HTML: return full body or single document
        return [
            LoadedDocument(
                doc_id=document_id,
                content=raw_text.strip(),
                source=path.name,
                page=0,
                section="main",
                metadata={"title": title, "has_page": False},
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
