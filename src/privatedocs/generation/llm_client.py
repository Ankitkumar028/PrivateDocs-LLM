"""OpenAI-compatible HTTP LLM client with retries, timeouts, and streaming support."""

from __future__ import annotations

import logging
import time
from collections.abc import Generator
from dataclasses import dataclass
from typing import Any

from openai import APIConnectionError, APITimeoutError, OpenAI, RateLimitError

logger = logging.getLogger(__name__)


class LLMClientError(Exception):
    """Base exception for LLM client errors."""


class LLMConnectionError(LLMClientError):
    """Raised when the LLM server is unreachable or offline."""


class LLMTimeoutError(LLMClientError):
    """Raised when the LLM server request exceeds configured timeout."""


class LLMGenerationError(LLMClientError):
    """Raised when the LLM server returns an error during generation."""


@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class GenerationResult:
    content: str
    usage: TokenUsage
    latency_sec: float
    ttft_sec: float | None = None


class LLMClient:
    """Robust client communicating with any OpenAI-compatible server (vLLM, Ollama, mock)."""

    def __init__(
        self,
        base_url: str = "http://localhost:8080/v1",
        api_key: str = "local-dev-key",
        model: str = "meta-llama/Llama-3.1-8B-Instruct",
        timeout_seconds: float = 60.0,
        max_retries: int = 3,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or "EMPTY"
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

        self._client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout_seconds,
            max_retries=0,  # We handle retries explicitly with backoff and error classification
        )

    def check_health(self) -> bool:
        """Check if LLM server is reachable and responsive."""
        try:
            models = self._client.models.list()
            return len(models.data) >= 0
        except Exception as exc:
            logger.warning("LLM server health check failed at %s: %s", self.base_url, exc)
            return False

    def generate_answer(
        self,
        messages: list[dict[str, str]] | str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        prompt: str | None = None,
    ) -> tuple[str, TokenUsage]:
        """Generate complete answer synchronously with exponential backoff retries."""
        target_input = prompt if prompt is not None else messages
        if target_input is None:
            raise ValueError("Either 'messages' or 'prompt' must be provided.")

        if isinstance(target_input, str):
            formatted_messages = [{"role": "user", "content": target_input}]
        else:
            formatted_messages = target_input

        attempt = 0
        last_exception: Exception | None = None

        while attempt <= self.max_retries:
            try:
                raw_response: Any = self._client.chat.completions.create(
                    model=self.model,
                    messages=formatted_messages,  # type: ignore[arg-type]
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=False,
                )

                content = raw_response.choices[0].message.content or ""
                prompt_tokens = raw_response.usage.prompt_tokens if raw_response.usage else 0
                completion_tokens = raw_response.usage.completion_tokens if raw_response.usage else 0
                total_tokens = raw_response.usage.total_tokens if raw_response.usage else prompt_tokens + completion_tokens

                usage = TokenUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                )
                return content, usage

            except (APIConnectionError, ConnectionRefusedError) as exc:
                last_exception = LLMConnectionError(
                    f"Could not connect to LLM server at {self.base_url}. Ensure the server is running. Error: {exc}"
                )
            except APITimeoutError as exc:
                last_exception = LLMTimeoutError(
                    f"LLM request timed out after {self.timeout_seconds}s at {self.base_url}. Error: {exc}"
                )
            except RateLimitError as exc:
                last_exception = LLMGenerationError(f"Rate limit exceeded on LLM server: {exc}")
            except Exception as exc:
                last_exception = LLMGenerationError(f"LLM generation failed: {exc}")

            attempt += 1
            if attempt <= self.max_retries:
                backoff = 0.5 * (2 ** (attempt - 1))
                logger.info("Retrying LLM request (attempt %d/%d) in %.1fs...", attempt, self.max_retries, backoff)
                time.sleep(backoff)

        raise last_exception or LLMGenerationError("Unknown LLM failure")

    def stream_answer(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> Generator[tuple[str, float | None], None, None]:
        """Stream response tokens with TTFT tracking. Yields (token_text, ttft_seconds)."""
        start_time = time.perf_counter()
        first_token_yielded = False

        try:
            raw_stream: Any = self._client.chat.completions.create(
                model=self.model,
                messages=messages,  # type: ignore[arg-type]
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )

            for chunk in raw_stream:
                if not getattr(chunk, "choices", None):
                    continue
                delta = chunk.choices[0].delta.content or ""
                if not delta:
                    continue

                ttft = None
                if not first_token_yielded:
                    ttft = time.perf_counter() - start_time
                    first_token_yielded = True

                yield delta, ttft

        except (APIConnectionError, ConnectionRefusedError) as exc:
            raise LLMConnectionError(
                f"LLM connection lost during streaming to {self.base_url}: {exc}"
            ) from exc
        except APITimeoutError as exc:
            raise LLMTimeoutError(
                f"LLM streaming timed out after {self.timeout_seconds}s: {exc}"
            ) from exc
        except Exception as exc:
            raise LLMGenerationError(f"Error during LLM streaming: {exc}") from exc
