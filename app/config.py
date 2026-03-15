"""Centralised application configuration."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DatabaseConfig:
    dsn: str = "postgresql://postgres:postgres@localhost:5432/auction"
    pool_min: int = 1
    pool_max: int = 10


@dataclass(frozen=True)
class OllamaConfig:
    base_url: str = "http://127.0.0.1:11434"
    timeout: int = 60
    chat_model: str = "qwen3.5:9b"
    embed_model: str = "nomic-embed-text"
    num_predict: int = 4096


@dataclass(frozen=True)
class QueryConfig:
    max_rows: int = 500
    default_limit: int = 20
    max_limit: int = 200
    min_limit: int = 10
    mileage_tolerance_km: int = 25_000
    year_tolerance: int = 5
    min_keyword_length: int = 3
    min_fallback_rows: int = 3


@dataclass(frozen=True)
class Settings:
    db: DatabaseConfig = field(default_factory=DatabaseConfig)
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    query: QueryConfig = field(default_factory=QueryConfig)
    log_level: str = "INFO"
    app_title: str = "Auction Intelligence API"
    app_version: str = "1.0.0"
    max_agent_iterations: int = 6


settings = Settings()