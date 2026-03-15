"""Centralised application configuration — all values read from .env"""

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


# ── helpers ──────────────────────────────────────────────────

def _env(key: str, default: str = "") -> str:
    return os.getenv(key, default)


def _env_int(key: str, default: int = 0) -> int:
    return int(os.getenv(key, str(default)))


def _env_tuple(key: str) -> tuple:
    raw = os.getenv(key, "")
    return tuple(k.strip() for k in raw.split(",") if k.strip())


# ── config groups ────────────────────────────────────────────

@dataclass(frozen=True)
class DatabaseConfig:
    dsn: str = _env("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/auction")
    pool_min: int = _env_int("DB_POOL_MIN", 1)
    pool_max: int = _env_int("DB_POOL_MAX", 10)


@dataclass(frozen=True)
class OllamaConfig:
    base_url: str = _env("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    timeout: int = _env_int("OLLAMA_TIMEOUT", 120)
    chat_model: str = _env("OLLAMA_CHAT_MODEL", "qwen3.5:9b")
    embed_model: str = _env("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    embed_dimensions: int = _env_int("EMBED_DIMENSIONS", 768)
    num_predict: int = _env_int("OLLAMA_NUM_PREDICT", 4096)


@dataclass(frozen=True)
class QueryConfig:
    max_rows: int = _env_int("MAX_ROWS", 500)
    default_limit: int = _env_int("DEFAULT_LIMIT", 20)
    max_limit: int = _env_int("MAX_LIMIT", 200)
    min_limit: int = _env_int("MIN_LIMIT", 10)
    mileage_tolerance_km: int = _env_int("MILEAGE_TOLERANCE_KM", 25000)
    year_tolerance: int = _env_int("YEAR_TOLERANCE", 5)
    min_keyword_length: int = _env_int("MIN_KEYWORD_LENGTH", 3)
    min_fallback_rows: int = _env_int("MIN_FALLBACK_ROWS", 3)


@dataclass(frozen=True)
class Settings:
    db: DatabaseConfig = field(default_factory=DatabaseConfig)
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    query: QueryConfig = field(default_factory=QueryConfig)
    log_level: str = _env("LOG_LEVEL", "INFO")
    app_title: str = "Auction Intelligence API"
    app_version: str = "1.0.0"
    model_id: str = _env("MODEL_ID", "auction-intelligence-v1")
    max_agent_iterations: int = _env_int("MAX_AGENT_ITERATIONS", 6)
    api_keys: tuple = field(default_factory=lambda: _env_tuple("API_KEYS"))


settings = Settings()