"""
config/settings.py  —  LOCAL / OLLAMA version

Switch between local and Bedrock by changing LLM_BACKEND in .env:
  LLM_BACKEND=ollama    →  uses Ollama (this file's defaults)
  LLM_BACKEND=bedrock   →  uses AWS Bedrock (swap to bedrock_client.py)
"""
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional


class Settings(BaseSettings):

    # ── Application ───────────────────────────────────────────────────────────
    APP_NAME: str = "FinServ Regulatory Compliance Assistant"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: str = Field(default="development", env="ENVIRONMENT")
    DEBUG: bool = Field(default=False, env="DEBUG")
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")

    # ── Backend switch ────────────────────────────────────────────────────────
    # Set to "ollama" for local dev, "bedrock" for AWS
    LLM_BACKEND: str = Field(default="ollama", env="LLM_BACKEND")

    # ── Ollama (local) ────────────────────────────────────────────────────────
    OLLAMA_BASE_URL: str = Field(default="http://localhost:11434", env="OLLAMA_BASE_URL")

    # Ollama model names (as shown in `ollama list`)
    LLM_PRIMARY_MODEL: str = Field(default="mistral", env="LLM_PRIMARY_MODEL")
    LLM_COMPLEX_MODEL: str = Field(default="mixtral", env="LLM_COMPLEX_MODEL")
    EMBEDDING_MODEL: str = Field(default="nomic-embed-text", env="EMBEDDING_MODEL")

    # ── AWS / Bedrock (cloud) — ignored when LLM_BACKEND=ollama ──────────────
    AWS_REGION: str = Field(default="us-east-1", env="AWS_REGION")
    AWS_PROFILE: Optional[str] = Field(default=None, env="AWS_PROFILE")
    BEDROCK_PRIMARY_MODEL: str = Field(
        default="mistral.mistral-7b-instruct-v0:2", env="BEDROCK_PRIMARY_MODEL"
    )
    BEDROCK_COMPLEX_MODEL: str = Field(
        default="mistral.mixtral-8x7b-instruct-v0:1", env="BEDROCK_COMPLEX_MODEL"
    )
    BEDROCK_EMBEDDING_MODEL: str = Field(
        default="amazon.titan-embed-text-v2:0", env="BEDROCK_EMBEDDING_MODEL"
    )

    # ── LLM behaviour ─────────────────────────────────────────────────────────
    LLM_TEMPERATURE: float = Field(default=0.1, env="LLM_TEMPERATURE")
    LLM_MAX_TOKENS: int = Field(default=2048, env="LLM_MAX_TOKENS")
    LLM_TIMEOUT_SECONDS: int = Field(default=120, env="LLM_TIMEOUT_SECONDS")

    # ── Vector store (Qdrant) ─────────────────────────────────────────────────
    QDRANT_URL: str = Field(default="http://localhost:6333", env="QDRANT_URL")
    QDRANT_MODE: str = Field(default="disk", env="QDRANT_MODE")
    QDRANT_API_KEY: Optional[str] = Field(default=None, env="QDRANT_API_KEY")
    QDRANT_COLLECTION: str = Field(default="regulatory_docs", env="QDRANT_COLLECTION")

    # nomic-embed-text  →  768 dims
    # Titan v2          → 1536 dims
    # MUST match the embedding model you're using
    EMBEDDING_DIMENSION: int = Field(default=768, env="EMBEDDING_DIMENSION")

    RETRIEVAL_TOP_K: int = Field(default=10, env="RETRIEVAL_TOP_K")
    RERANK_TOP_K: int = Field(default=5, env="RERANK_TOP_K")

    # ── PostgreSQL + Redis ────────────────────────────────────────────────────
    POSTGRES_URL: str = Field(
        default="postgresql://finserv:finserv@localhost:5432/compliance",
        env="POSTGRES_URL"
    )
    REDIS_URL: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")
    CACHE_TTL_SECONDS: int = Field(default=3600, env="CACHE_TTL_SECONDS")

    # ── Reranker (always local — CPU, ~10ms) ──────────────────────────────────
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # ── Ingestion ─────────────────────────────────────────────────────────────
    CHUNK_SIZE_TOKENS: int = Field(default=400, env="CHUNK_SIZE_TOKENS")
    CHUNK_OVERLAP_TOKENS: int = Field(default=50, env="CHUNK_OVERLAP_TOKENS")
    DOCS_DIR: str = Field(default="sample_docs", env="DOCS_DIR")

    # ── Agent ─────────────────────────────────────────────────────────────────
    MAX_REFLECTION_CYCLES: int = Field(default=2, env="MAX_REFLECTION_CYCLES")
    CONFIDENCE_THRESHOLD_HUMAN_REVIEW: float = Field(default=0.4, env="CONFIDENCE_THRESHOLD_HUMAN_REVIEW")

    # ── Security ──────────────────────────────────────────────────────────────
    API_KEY_HEADER: str = "X-API-Key"
    API_KEYS: list[str] = Field(default=["dev-key-1234"], env="API_KEYS")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


settings = Settings()


# patch: add QDRANT_MODE if not already present
