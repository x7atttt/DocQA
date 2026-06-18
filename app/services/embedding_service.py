import asyncio
import logging
import threading
import warnings
from typing import Any

from FlagEmbedding import BGEM3FlagModel

from app.config import get_settings

settings = get_settings()

# 过滤 FlagEmbedding 内部的 XLMRobertaTokenizerFast 提示（库内部用 encode+pad
# 而非 __call__，属于库的实现细节，不影响功能）
warnings.filterwarnings("ignore", message=".*XLMRobertaTokenizerFast.*")
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

_model: BGEM3FlagModel | None = None
_lock = threading.Lock()


def _select_device() -> str:
    """选择推理设备：有 CUDA 用 cuda，否则 cpu。
    FlagEmbedding 的 BGEM3FlagModel 接受 device 字符串（cuda / cuda:0 / cpu）。
    """
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def get_embedding_model() -> BGEM3FlagModel:
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                device = _select_device()
                # use_fp16 仅在 GPU 上生效（CPU 上 fp16 反而更慢且精度差），故按设备自适应
                use_fp16 = device.startswith("cuda")
                _model = BGEM3FlagModel(
                    settings.embedding_model_path,
                    use_fp16=use_fp16,
                    device=device,
                )
                logging.getLogger("docqa.embedding").info(
                    f"BGE-M3 加载完成 (device={device}, fp16={use_fp16})"
                )
    return _model


def _encode_sync(texts: list[str], batch_size: int = 16) -> list[list[float]]:
    if not texts:
        return []
    model = get_embedding_model()
    # max_length 与 chunk_size 对齐：chunk 默认 500 字符，512 token 足够覆盖。
    # 之前用 8192 导致 BGE-M3 在 CPU 上每个 batch 推理极慢（17 chunks 21s），
    # 调到 512 后 CPU 提速约 5-7 倍，GPU 下更是 <1s。
    output: dict[str, Any] = model.encode(
        texts, batch_size=batch_size, max_length=settings.embedding_max_length, return_dense=True
    )
    return output["dense_vecs"].tolist()


async def encode_texts(texts: list[str], batch_size: int = 16) -> list[list[float]]:
    return await asyncio.to_thread(_encode_sync, texts, batch_size)


async def encode_single(text: str) -> list[float]:
    result = await encode_texts([text])
    return result[0]
