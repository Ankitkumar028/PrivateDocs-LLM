"""Generation module for prompt construction and OpenAI-compatible LLM communication."""

from privatedocs.generation.llm_client import LLMClient, LLMGenerationError, LLMTimeoutError
from privatedocs.generation.prompts import build_rag_prompt

__all__ = ["LLMClient", "LLMGenerationError", "LLMTimeoutError", "build_rag_prompt"]
