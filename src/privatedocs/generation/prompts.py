"""Prompt templates and formatting for confidential document Q&A."""

from __future__ import annotations

from typing import Any

SYSTEM_PROMPT = """You are PrivateDocs Assistant, a secure question-answering assistant for confidential company documents.

STRICT INSTRUCTIONS:
1. Answer the user's question ONLY using the factual information provided in the Document Context below.
2. If the context does not contain enough information to answer the question, or if no relevant context is provided, you MUST reply exactly:
   "I don't know based on the provided documents."
3. Every single factual statement or claim MUST include a citation indicating the source:
   - For PDF documents with page numbers, cite in the exact bracket format: [doc_id, p.<page>] (e.g. [confidential_memo, p.2]).
   - For HTML documents or SEC filings with section headings (where no page exists), cite in the exact bracket format: [doc_id, <section>] (e.g. [sec_apple_10k, Item 1A]).
4. Do NOT speculate, extrapolate, or use external knowledge not present in the Document Context.
5. Keep your tone objective, professional, and concise.
"""


def format_context(chunks: list[dict[str, Any]]) -> str:
    """Format retrieved document chunks into prompt context with clear headers."""
    if not chunks:
        return "No relevant documents found."

    formatted_parts = []
    for chunk in chunks:
        doc_id = chunk.get("doc_id", "unknown_doc")
        page = chunk.get("page", 0)
        section = chunk.get("section", "main")
        content = chunk.get("content", "").strip()

        # Build location label: Page if page > 0, otherwise Section
        if page and page > 0:
            if section and section not in ("page", "main", "html_body", "full_text"):
                loc_label = f"Page: {page} | Section: {section}"
            else:
                loc_label = f"Page: {page}"
        else:
            loc_label = f"Section: {section}"

        formatted_parts.append(f"--- Document: {doc_id} | {loc_label} ---\n{content}\n")

    return "\n".join(formatted_parts)


def build_rag_prompt(question: str, chunks: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Build OpenAI-compatible messages payload with system instructions and context."""
    context_str = format_context(chunks)

    user_message = f"""Document Context:
{context_str}

User Question: {question}

Provide a factual answer with strict citations ([doc_id, p.<page>] or [doc_id, <section>]) based solely on the Document Context above:"""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]
