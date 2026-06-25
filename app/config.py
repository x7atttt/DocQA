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
    # BGE-M3 encode 的 max_length（token）。应与 chunk_size 对齐：
    # chunk 默认 500 字符 ≈ 400-500 token，故 512 足够。
    # 设大了（如 8192）会让 CPU 推理极慢，GPU 也浪费算力。
    embedding_max_length: int = 512
    # Reranker 的 max_length（token）。query+chunk 拼接，768 较稳妥。
    rerank_max_length: int = 768

    chunk_size: int = 500
    chunk_overlap: int = 100
    # 分块策略：auto（按类型路由）| fixed | markdown | recursive
    split_strategy: str = "auto"

    retrieve_top_k: int = 20
    rerank_top_k: int = 3
    # 每用户最大会话数（超出拒绝创建，提示先删除旧会话）
    max_conversations: int = 10

    cache_ttl_seconds: int = 1800
    cache_null_ttl_seconds: int = 60

    # ============ MinerU 文档解析 API（PDF 高精度解析：OCR/表格/公式）============
    # 留空则 PDF 解析回退到本地 pymupdf4llm（不含 OCR/图片提取）
    mineru_token: str = ""
    mineru_base_url: str = "https://mineru.net/api/v4"
    # pipeline=免费/CPU可跑（PP-OCRv6）；vlm=高精度需GPU算力
    mineru_model_version: str = "pipeline"
    # 轮询总超时（秒），超时自动回退 pymupdf4llm
    mineru_timeout: int = 180


@lru_cache
def get_settings() -> Settings:
    return Settings()
