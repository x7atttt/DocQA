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
    # Hybrid 检索（dense+sparse RRF 融合）参数
    # dense 粗筛召回量：从 Chroma 取的候选数，需 > retrieve_top_k 以给 sparse 重排留余量
    dense_recall_top_k: int = 50
    # RRF（Reciprocal Rank Fusion）常数 k：score = Σ 1/(k + rank_i)，标准值 60
    # k 越大，排名靠后的项衰减越慢（对各路更均衡）；k 越小，头部项权重越大
    rrf_k: int = 60
    # 每用户最大会话数（超出拒绝创建，提示先删除旧会话）
    max_conversations: int = 10

    # ============ 文档增量更新 ============
    # 更新文档时，按分块 content_hash 集合 diff，仅重算变化块。
    # 变化块占比超此阈值（边界漂移严重）则降级全量重建。
    incremental_update_threshold: float = 0.5

    # ============ 会话记忆 / 历史管理 ============
    # 生成答案时历史的最大轮数上限（每轮 = user + assistant 两条消息）
    max_history_rounds: int = 5
    # query 改写用的历史轮数（指代消解只需近期上下文，比生成窗口小）
    rewrite_history_rounds: int = 3
    # 生成时历史的 token 预算上限（仅历史部分，不含 system prompt/文档）
    # 用 token 预算而非纯条数截断，避免长答案历史撑爆 context
    history_token_budget: int = 3500
    # 会话累计轮数达此阈值后异步生成摘要（压缩老对话）。
    # 应略大于 max_history_rounds：窗口外积攒几轮老对话再压缩，避免
    # 刚掉出窗口就触发（那些轮次还没积累够上下文）。默认 max_history_rounds(5)+3=8。
    # 两者差距过大（如 12）会出现"窗口外、摘要未触发"的真空区，老对话丢失。
    summarize_round_threshold: int = 8
    # 摘要生成的 max_tokens
    summary_max_tokens: int = 512
    # 对话历史 Redis 缓存 TTL（秒）
    history_cache_ttl_seconds: int = 7200

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
