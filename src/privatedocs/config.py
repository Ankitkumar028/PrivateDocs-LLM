"""Application configuration management using pydantic-settings."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for PrivateDocs-LLM."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Environment
    ENVIRONMENT: str = "local"
    LOG_LEVEL: str = "INFO"

    # LLM Settings (OpenAI-compatible)
    LLM_BASE_URL: str = "http://localhost:8080/v1"
    LLM_API_KEY: str = "local-dev-key"
    LLM_MODEL: str = "meta-llama/Llama-3.1-8B-Instruct"
    LLM_TIMEOUT_SECONDS: float = 60.0
    LLM_MAX_RETRIES: int = 3
    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_TOKENS: int = 1024

    # Embedder Settings
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-small-en-v1.5"
    EMBEDDING_DEVICE: str = "cpu"

    # Qdrant Vector Store
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_URL: str | None = None
    QDRANT_API_KEY: str | None = None
    QDRANT_COLLECTION_NAME: str = "private_documents"
    QDRANT_IN_MEMORY: bool = False

    # Ingestion & Chunking
    CHUNKING_STRATEGY: str = "fixed"  # fixed | recursive | sentence_window
    CHUNK_SIZE: int = 400
    CHUNK_OVERLAP: int = 50

    # Retrieval
    RETRIEVAL_MODE: str = "hybrid"  # dense | bm25 | hybrid
    TOP_K: int = 5
    RRF_K: int = 60
    RERANK_ENABLED: bool = False
    RERANK_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # FastAPI Gateway
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_KEYS_DB_PATH: str = "data/auth_keys.db"
    RATE_LIMIT_PER_MINUTE: int = 60
    ADMIN_API_KEY: str = "admin-secret-dev-key-change-in-prod"
    USER_API_KEY: str = "user-secret-dev-key-change-in-prod"

    # Streamlit UI
    STREAMLIT_PORT: int = 8501
    API_BASE_URL: str = "http://localhost:8000"

    # GPU / Cloud
    VLLM_IMAGE: str = "rocm/vllm:rocm6.2_mi300_ubuntu22.04_py3.10_vllm_0.6.3"
    HF_TOKEN: str | None = None
    HF_HUB_ENABLE_HF_TRANSFER: int = 1
    GPU_HOURLY_RATE: float = 1.99
    GPU_MAX_BUDGET_HOURS: float = 35.0


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached Settings instance."""
    return Settings()
