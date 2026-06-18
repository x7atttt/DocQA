import asyncio
import logging
import threading

from FlagEmbedding import FlagReranker

from app.config import get_settings

settings = get_settings()
logger = logging.getLogger("docqa.rerank")

_reranker: FlagReranker | None = None
_lock = threading.Lock()


def _select_device() -> str:
    """选择推理设备：有 CUDA 用 cuda，否则 cpu。与 embedding_service 保持一致。"""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def get_reranker() -> FlagReranker:
    global _reranker
    if _reranker is None:
        with _lock:
            if _reranker is None:
                device = _select_device()
                use_fp16 = device.startswith("cuda")
                _reranker = FlagReranker(
                    settings.rerank_model_path,
                    use_fp16=use_fp16,
                    device=device,
                )
                logger.info(f"BGE-Reranker 加载完成 (device={device}, fp16={use_fp16})")
    return _reranker


def _rerank_sync(query: str, documents: list[str], top_k: int = 3) -> list[tuple[int, float]]:
    if not documents:
        return []
    model = get_reranker()
    pairs = [[query, doc] for doc in documents]
    # max_length 与 settings 对齐（默认 768），避免过长的 token 截断配置不一致
    scores = model.compute_score(pairs, normalize=True, max_length=settings.rerank_max_length)
    if not isinstance(scores, list):
        scores = [scores]
    scored = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    return scored[:top_k]


async def rerank(query: str, documents: list[str], top_k: int = 3) -> list[tuple[int, float]]:
    return await asyncio.to_thread(_rerank_sync, query, documents, top_k)
