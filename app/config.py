from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # App
    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    app_version: str = "1.0.0"

    # Security
    internal_api_key: str = "change-me-in-production"

    # Gemini
    gemini_api_key: str = ""
    gemini_embedding_model: str = "models/gemini-embedding-2"
    gemini_llm_model: str = "gemini-3.6-flash"

    # Database (PostgreSQL + pgvector)
    database_url: str = ""

    # RAG tuning
    similarity_threshold: float = 0.5
    top_k_default: int = 5
    chunking_strategy: str = "semantic"
    chunk_size: int = 800
    chunk_overlap: int = 150
    semantic_breakpoint_percentile: float = 90.0
    min_chunk_size: int = 350
    max_chunk_size: int = 1200

    # LLM token budget
    max_context_tokens: int = 1_048_576   # model context window (gemini-3.6-flash)
    max_output_tokens: int = 8192         # max tokens reserved for the response

    # Logging
    log_level: str = "DEBUG"

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
