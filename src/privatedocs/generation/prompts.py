"""Prompt templates and formatting for confidential document Q&A."""

from __future__ import annotations

from typing import Any

SYSTEM_PROMPT = """You are PrivateDocs Assistant, a secure question-answering assistant for confidential company documents.

STRICT INSTRUCTIONS:
1. Answer the user's question ONLY using the factual information provided in the Document Context below.
2. If the context does not contain enough information to answer the question, or if no relevant context is provided, you MUST reply exactly:
   "I don't know based on the provided documents."
3. Every single factual statement or claim MUST include a citation indicating the source document and page number in the exact bracket format: [doc_id, page] (e.g. [sec_apple_10k, 24]).
4. Do NOT speculate, extrapolate, or use external knowledge not present in the Document Context.
5. Keep your tone objective, professional, and concise.
"""

CONTEXT_CHUNK_TEMPLATE = """--- Document: {doc_id} | Page: {page} | Section: {section} ---
{content}
"""


def format_context(chunks: list[dict[str, Any]]) -> str:
    """Format retrieved document chunks into prompt context with clear headers."""
    if not chunks:
        return "No relevant documents found."

    formatted_parts = []
    for chunk in chunks:
        doc_id = chunk.get("doc_id", "unknown_doc")
        page = chunk.get("page", 1)
        section = chunk.get("section", "general")
        content = chunk.get("content", "").strip()

        formatted_parts.append(
            CONTEXT_CHUNK_TEMPLATE.format(
                doc_id=doc_id,
                page=page,
                section=section,
                content=content,
            )
        )

    return "\n".join(formatted_parts)


def build_rag_prompt(question: str, chunks: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Build OpenAI-compatible messages payload with system instructions and context."""
    context_str = format_context(chunks)

    user_message = f"""Document Context:
{context_str}

User Question: {question}

Provide a factual answer with strict citations [doc_id, page] based solely on the Document Context above:"""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]
