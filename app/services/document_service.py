import asyncio
import io
import logging
import os
import time
import uuid
import zipfile

import chromadb
import requests
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.core.exceptions import BizError
from app.core.response import ResponseCode
from app.models import Document
from app.services.embedding_service import encode_texts

settings = get_settings()
logger = logging.getLogger("docqa.document")

SUPPORTED_EXTS = {"pdf", "docx", "md"}

_chroma_client: chromadb.api.ClientAPI | None = None
_chroma_lock = asyncio.Lock()


def get_chroma_client() -> chromadb.api.ClientAPI:
    global _chroma_client
    if _chroma_client is None:
        os.makedirs(settings.chroma_persist_dir, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    return _chroma_client


def get_user_collection(user_id: int) -> chromadb.Collection:
    return get_chroma_client().get_or_create_collection(
        name=f"doc_user_{user_id}",
        metadata={"hnsw:space": "cosine"},
    )


def _parse_pdf_pymupdf_sync(file_path: str) -> str:
    """用 pymupdf4llm 解析 PDF → Markdown（版面感知，保留表格/多栏顺序/标题层级）。

    作为 MinerU 不可用时的回退方案。不含 OCR：扫描件（图片型 PDF）会返回空字符串。
    """
    import pymupdf4llm

    md = pymupdf4llm.to_markdown(file_path)  # write_images 默认 False，不提取图片
    return md.strip()


def _parse_pdf_mineru_sync(file_path: str) -> str:
    """用 MinerU 云 API 解析 PDF → Markdown（含 OCR/表格/公式/页眉页脚去除）。

    异步任务流程：申请上传 URL → PUT 上传到 OSS → 轮询任务状态 → 下载 zip → 取 .md。
    失败抛异常，由上层 _parse_pdf_sync 捕获并回退 pymupdf4llm。

    注意：MinerU 输出的表格是 HTML 格式（<table>），当前分块策略不专门处理，
    可能在大表格中间切断（已知限制）。
    """
    token = settings.mineru_token
    base = settings.mineru_base_url
    headers = {"Authorization": f"Bearer {token}", "Accept": "*/*"}
    file_name = os.path.basename(file_path)

    # 1. 申请上传 URL（file-urls/batch）
    resp = requests.post(
        f"{base}/file-urls/batch",
        headers={**headers, "Content-Type": "application/json"},
        json={
            "files": [{"name": file_name, "data_id": uuid.uuid4().hex[:12]}],
            "model_version": settings.mineru_model_version,
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    if data.get("code") != 0:
        raise RuntimeError(f"MinerU 申请上传 URL 失败: {data.get('msg', data)}")
    batch_id = data["data"]["batch_id"]
    upload_url = data["data"]["file_urls"][0]

    # 2. PUT 上传文件到 OSS（不带 Content-Type，避免签名不匹配）
    with open(file_path, "rb") as f:
        resp = requests.put(upload_url, data=f.read(), timeout=180)
    if resp.status_code != 200:
        raise RuntimeError(f"MinerU 文件上传失败: HTTP {resp.status_code}")

    # 3. 轮询任务状态（每 5s，超时由 mineru_timeout 控制）
    max_attempts = max(1, settings.mineru_timeout // 5)
    for i in range(max_attempts):
        resp = requests.get(f"{base}/extract-results/batch/{batch_id}", headers=headers, timeout=30)
        resp.raise_for_status()
        result = resp.json()
        er = result.get("data", {}).get("extract_result", [])
        status = er[0].get("state") if er else None
        if status == "done":
            zip_url = er[0].get("full_zip_url")
            if not zip_url:
                raise RuntimeError("MinerU 任务完成但无结果 URL")
            break
        if status in ("failed", "error"):
            raise RuntimeError(f"MinerU 解析失败: {er[0].get('err_msg', status)}")
        time.sleep(5)
    else:
        raise TimeoutError(f"MinerU 轮询超时（{settings.mineru_timeout}s）")

    # 4. 下载 zip，解压取 .md 文件
    resp = requests.get(zip_url, timeout=120)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        md_files = [n for n in zf.namelist() if n.endswith(".md")]
        if not md_files:
            raise RuntimeError("MinerU 结果 zip 中无 Markdown 文件")
        md_content = zf.read(md_files[0]).decode("utf-8")

    logger.info(f"MinerU 解析完成: {file_name} → {len(md_content)} 字符")
    return md_content.strip()


def _parse_pdf_sync(file_path: str) -> str:
    """PDF 解析统一入口：优先 MinerU（OCR/表格/公式），失败回退 pymupdf4llm。

    MinerU 是云服务，可能超时/限流/网络抖动；回退保证 PDF 解析始终可用。
    无 mineru_token 时直接走 pymupdf4llm。
    """
    if settings.mineru_token:
        try:
            return _parse_pdf_mineru_sync(file_path)
        except Exception as e:
            logger.warning(f"MinerU 解析失败，回退 pymupdf4llm: {e}")
    return _parse_pdf_pymupdf_sync(file_path)


def _parse_docx_sync(file_path: str) -> str:
    """用 MarkItDown 解析 DOCX → Markdown（mammoth 底层，保留表格结构）。"""
    from markitdown import MarkItDown

    md = MarkItDown()
    result = md.convert(file_path)
    return result.text_content.strip()


def _parse_markdown_sync(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def _chunk_text_sync(text: str, chunk_size: int, overlap: int) -> list[str]:
    if not text:
        return []
    chunks = []
    start = 0
    step = max(chunk_size - overlap, 1)
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start += step
    return [c for c in chunks if c.strip()]


def chunk_text(text: str, chunk_size: int | None = None, overlap: int | None = None) -> list[str]:
    """向后兼容的包装：等价于 split_text(strategy="fixed")。

    保留是为了兼容 tests/test_chunking.py 的旧用例；生产路径已改用 split_text。
    """
    from app.services.text_splitter import split_text

    return split_text(
        text,
        strategy="fixed",
        chunk_size=chunk_size if chunk_size is not None else settings.chunk_size,
        chunk_overlap=overlap if overlap is not None else settings.chunk_overlap,
    )


async def parse_file(file_path: str, ext: str) -> str:
    parsers = {"pdf": _parse_pdf_sync, "docx": _parse_docx_sync, "md": _parse_markdown_sync}
    return await asyncio.to_thread(parsers[ext], file_path)


async def process_document(
    file_path: str,
    filename: str,
    ext: str,
    file_size: int,
    user_id: int,
    db: AsyncSession,
    file_hash: str | None = None,
) -> Document:
    if ext not in SUPPORTED_EXTS:
        raise BizError(
            code=ResponseCode.UNSUPPORTED_FILE_TYPE,
            message=f"不支持的文件类型: {ext}",
            http_status=400,
        )

    text = await parse_file(file_path, ext)
    if not text:
        raise BizError(
            code=ResponseCode.DOC_PARSE_FAILED,
            message="无法解析文本内容（可能是扫描版 PDF，暂不支持 OCR）",
            http_status=400,
        )

    # 分块：按配置策略切分（auto/fixed/markdown/recursive），传 ext 供 auto 路由
    from app.services.text_splitter import split_text

    chunks = await asyncio.to_thread(
        split_text, text, settings.split_strategy, settings.chunk_size, settings.chunk_overlap, ext
    )
    if not chunks:
        raise BizError(code=ResponseCode.DOC_PARSE_FAILED, message="文档内容为空", http_status=400)

    embeddings = await encode_texts(chunks)

    document = Document(
        user_id=user_id,
        filename=filename,
        file_type=ext,
        chunk_count=len(chunks),
        file_size=file_size,
        file_hash=file_hash,
    )
    db.add(document)
    await db.commit()
    await db.refresh(document)

    try:
        async with _chroma_lock:
            collection = get_user_collection(user_id)
            ids = [f"{document.id}_chunk_{i}" for i in range(len(chunks))]
            metadatas = [
                {
                    "user_id": user_id,
                    "document_id": document.id,
                    "filename": filename,
                    "chunk_index": i,
                }
                for i in range(len(chunks))
            ]
            collection.add(ids=ids, documents=chunks, embeddings=embeddings, metadatas=metadatas)
    except Exception:
        await db.delete(document)
        await db.commit()
        raise

    return document


async def delete_document(document: Document, db: AsyncSession) -> None:
    import logging

    logger = logging.getLogger("docqa.document")
    document_id = document.id
    user_id = document.user_id
    await db.delete(document)
    await db.commit()

    try:
        async with _chroma_lock:
            collection = get_user_collection(user_id)
            # document_id 在 metadata 里存的是 int，where 过滤需用同类型
            collection.delete(where={"document_id": document_id})
            logger.info(f"已清理文档 {document_id} 在向量库中的向量 (user={user_id})")
    except Exception as e:
        # 不再静默吞异常：记录日志，便于发现向量库与元数据不一致
        logger.warning(f"清理文档 {document_id} 向量失败（向量库可能有残留）: {e}")


def save_upload_file(content: bytes, ext: str) -> tuple[str, str]:
    os.makedirs("data/uploads", exist_ok=True)
    file_id = uuid.uuid4().hex[:12]
    file_path = os.path.abspath(os.path.join("data/uploads", f"{file_id}.{ext}"))
    with open(file_path, "wb") as f:
        f.write(content)
    return file_path, file_id
