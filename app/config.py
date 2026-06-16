from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    jwt_secret_key: str = "change-this"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 1440

    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_model: str = ""

    redis_url: str = "redis://localhost:6379/0"

    database_url: str = "sqlite+aiosqlite:///./data/docqa.db"

    chroma_persist_dir: str = "./data/chroma_db"

    embedding_model_path: str = "./models/bge-m3"
    rerank_model_path: str = "./models/bge-reranker-v2-m3"

    chunk_size: int = 500
    chunk_overlap: int = 100
    # 分块策略：auto（按类型路由）| fixed | markdown | recursive
    split_strategy: str = "auto"

    retrieve_top_k: int = 20
    rerank_top_k: int = 3

    cache_ttl_seconds: int = 1800
    cache_null_ttl_seconds: int = 60


@lru_cache
def get_settings() -> Settings:
    return Settings()
