import asyncio
import hashlib
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
from app.core.database import async_session_factory
from app.core.exceptions import BizError
from app.core.response import ResponseCode
from app.models import Document
from app.services.embedding_service import encode_texts

settings = get_settings()
logger = logging.getLogger("docqa.document")

SUPPORTED_EXTS = {"pdf", "docx", "md"}

_chroma_client: chromadb.api.ClientAPI | None = None
_chroma_lock = asyncio.Lock()
# MinerU 云 API 并发限制：防批量上传时打到云 API 触发限流/静默降级
# Semaphore 在事件循环内生效，限制同时进行的 MinerU 解析数
_mineru_sem: asyncio.Semaphore | None = None


def _get_mineru_sem() -> asyncio.Semaphore:
    """延迟初始化 MinerU 并发信号量（asyncio.Semaphore 必须在事件循环内创建）。"""
    global _mineru_sem
    if _mineru_sem is None:
        _mineru_sem = asyncio.Semaphore(2)
    return _mineru_sem


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


def _compute_chunk_hash(chunk: str) -> str:
    """计算单个分块的内容哈希（sha256 截 16 位），用于增量更新 diff。"""
    return hashlib.sha256(chunk.encode("utf-8")).hexdigest()[:16]


def _build_chunk_metadata(
    user_id: int, doc_id: int, filename: str, chunks: list[str]
) -> list[dict]:
    """统一生成分块元数据列表（含 content_hash，供增量更新 diff 用）。

    抽出来消除 process_pending_document / process_document 两处的重复。
    """
    return [
        {
            "user_id": user_id,
            "document_id": doc_id,
            "filename": filename,
            "chunk_index": i,
            "content_hash": _compute_chunk_hash(chunk),
        }
        for i, chunk in enumerate(chunks)
    ]


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
    if ext == "pdf":
        # MinerU 云 API 限并发（防批量上传打到云 API 触发限流/静默降级）
        # 非并发瓶颈的本地解析（docx/md）不限流
        async with _get_mineru_sem():
            return await asyncio.to_thread(parsers[ext], file_path)
    return await asyncio.to_thread(parsers[ext], file_path)


async def create_pending_document(
    filename: str,
    ext: str,
    file_size: int,
    user_id: int,
    db: AsyncSession,
    file_hash: str | None = None,
) -> Document:
    """创建 pending 状态的文档记录（解析前入库，供前端轮询状态）。

    C2 异步上传：先建 status=pending 的记录立即返回，实际解析在后台
    process_pending_document 跑。前端轮询列表看到 pending→processing→done。
    """
    document = Document(
        user_id=user_id,
        filename=filename,
        file_type=ext,
        chunk_count=0,
        file_size=file_size,
        file_hash=file_hash,
        status="pending",
    )
    db.add(document)
    await db.commit()
    await db.refresh(document)
    return document


async def process_pending_document(
    document_id: int,
    file_path: str,
    ext: str,
    user_id: int,
    replace_id: int | None = None,
) -> None:
    """后台处理 pending 文档：解析 → 分块 → embedding → 入库 → 更新 status。

    独立于 HTTP 请求运行（BackgroundTasks 调起），内部新建独立 AsyncSession
    （不复用请求的 session，请求结束后 session 已关闭）。

    状态流转：pending → processing → done（成功）/ failed（异常，记录错误信息）。
    失败时清理已落盘的文件，status 置 failed，不影响其他文档。

    replace_id 非空时走增量更新：复用旧文档记录（不新建），后台 diff chunks。
    """
    from app.services.text_splitter import split_text

    # 独立 session（脱离请求生命周期）
    async with async_session_factory() as db:
        doc = await db.get(Document, document_id)
        if doc is None:
            logger.warning(f"后台处理：文档 {document_id} 不存在")
            return

        try:
            doc.status = "processing"
            await db.commit()

            text = await parse_file(file_path, ext)
            if not text:
                raise RuntimeError("解析内容为空（可能是扫描版 PDF 且 OCR 失败）")

            chunks = await asyncio.to_thread(
                split_text, text, settings.split_strategy,
                settings.chunk_size, settings.chunk_overlap, ext,
            )
            if not chunks:
                raise RuntimeError("分块结果为空")

            if replace_id is not None:
                # 增量更新：diff 新旧分块，仅重算变化块
                stats = await update_document_chunks(doc, chunks, user_id, db)
                logger.info(
                    f"文档 {document_id} ({doc.filename}) 增量更新完成，"
                    f"+{stats['added']} -{stats['removed']} 复用{stats['reused']}"
                    f"{'（降级全量）' if stats['degraded'] else ''}"
                )
            else:
                # 新建文档：全量 embedding + add
                embeddings = await encode_texts(chunks)
                async with _chroma_lock:
                    collection = get_user_collection(user_id)
                    ids = [f"{doc.id}_chunk_{i}" for i in range(len(chunks))]
                    metadatas = _build_chunk_metadata(user_id, doc.id, doc.filename, chunks)
                    collection.add(ids=ids, documents=chunks, embeddings=embeddings, metadatas=metadatas)
                doc.chunk_count = len(chunks)

            doc.status = "done"
            await db.commit()
            logger.info(f"文档 {document_id} ({doc.filename}) 处理完成，{doc.chunk_count} chunks")
        except Exception as e:
            logger.warning(f"文档 {document_id} 处理失败: {e}")
            doc.status = "failed"
            await db.commit()
            # 清理已落盘的文件
            try:
                os.remove(file_path)
            except OSError:
                pass


async def process_document(
    file_path: str,
    filename: str,
    ext: str,
    file_size: int,
    user_id: int,
    db: AsyncSession,
    file_hash: str | None = None,
) -> Document:
    """同步处理文档（解析+分块+embedding+入库），供测试和兼容旧调用。

    生产路径走 create_pending_document + process_pending_document（异步），
    本函数保留是为了 tests 和不需要异步的简单场景。
    """
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
        status="done",
    )
    db.add(document)
    await db.commit()
    await db.refresh(document)

    try:
        async with _chroma_lock:
            collection = get_user_collection(user_id)
            ids = [f"{document.id}_chunk_{i}" for i in range(len(chunks))]
            metadatas = _build_chunk_metadata(user_id, document.id, filename, chunks)
            collection.add(ids=ids, documents=chunks, embeddings=embeddings, metadatas=metadatas)
    except Exception:
        await db.delete(document)
        await db.commit()
        raise

    return document


async def update_document_chunks(
    document: Document, new_chunks: list[str], user_id: int, db: AsyncSession
) -> dict:
    """文档增量更新：按分块 content_hash 集合 diff，仅对变化块重算向量。

    流程：
      1. 取旧文档在 ChromaDB 的所有块（含 content_hash metadata）
      2. 对新分块算 content_hash，集合 diff：
         - 旧有新无（removed）→ 删除消失的块
         - 新有旧无（added）→ encode 新增块后 upsert
         - 都有且 hash 同 → 复用旧向量（核心省算力）
      3. 边界漂移检测：变化率 > incremental_update_threshold → 降级全量重建
         （改一处导致分块边界整体后移时，diff 无收益，不如直接重建）
      4. 旧文档无 content_hash（stage-14 之前的数据）→ removed 含全部旧块，
         变化率必然超阈值 → 自动降级全量，首次更新后补全 hash

    返回统计 dict（added/removed/reused/degraded），供调用方记录日志。
    """
    collection = get_user_collection(user_id)
    new_hashes = [_compute_chunk_hash(c) for c in new_chunks]
    new_hash_set = set(new_hashes)

    # 取旧块（含 metadata，用于读 content_hash 和定位 id）
    async with _chroma_lock:
        old = collection.get(
            where={"document_id": document.id},
            include=["metadatas"],
        )
    old_ids = old["ids"]
    old_metas = old["metadatas"]
    # 旧块可能无 content_hash（stage-14 前数据），用 .get 兜底为空串
    old_hash_to_ids: dict[str, list[str]] = {}
    for oid, meta in zip(old_ids, old_metas):
        h = meta.get("content_hash", "")
        old_hash_to_ids.setdefault(h, []).append(oid)
    old_hash_set = set(old_hash_to_ids.keys())

    # 集合 diff（顺序无关）
    added_hashes = new_hash_set - old_hash_set
    removed_hashes = old_hash_set - new_hash_set
    reused_count = len(new_hash_set & old_hash_set)

    # 边界漂移检测：变化率超阈值降级全量重建
    union_size = len(new_hash_set | old_hash_set)
    changed_ratio = (len(added_hashes) + len(removed_hashes)) / max(union_size, 1)
    if changed_ratio > settings.incremental_update_threshold:
        stats = await _full_rebuild_chunks(document, new_chunks, user_id, db)
        stats["degraded"] = True
        logger.info(
            f"文档 {document.id} 变化率 {changed_ratio:.0%} > 阈值，降级全量重建"
        )
        return stats

    # 增量路径：删除消失块
    removed_ids: list[str] = []
    for h in removed_hashes:
        removed_ids.extend(old_hash_to_ids.get(h, []))
    # 新增块：用 hash 找回对应的 chunk 文本（集合 diff 后需重建 id+metadata）
    added_chunks = [
        (i, c) for i, (c, h) in enumerate(zip(new_chunks, new_hashes)) if h in added_hashes
    ]

    async with _chroma_lock:
        if removed_ids:
            collection.delete(ids=removed_ids)
        if added_chunks:
            added_texts = [c for _, c in added_chunks]
            added_embeddings = await encode_texts(added_texts)
            # 新增块用位置 id（_chunk_{i}），与旧块 id 空间不冲突（旧块的对应位置
            # 要么已删除要么是复用块——复用块 id 仍在但 hash 不同会作为 added 重算）
            added_ids = [f"{document.id}_chunk_{i}" for i, _ in added_chunks]
            added_metas = [
                {
                    "user_id": user_id,
                    "document_id": document.id,
                    "filename": document.filename,
                    "chunk_index": i,
                    "content_hash": _compute_chunk_hash(c),
                }
                for i, c in added_chunks
            ]
            collection.upsert(
                ids=added_ids,
                embeddings=added_embeddings,
                documents=added_texts,
                metadatas=added_metas,
            )

    document.chunk_count = len(new_chunks)
    await db.commit()
    stats = {
        "added": len(added_chunks),
        "removed": len(removed_ids),
        "reused": reused_count,
        "degraded": False,
    }
    logger.info(
        f"文档 {document.id} 增量更新完成：+{stats['added']} -{stats['removed']} "
        f"复用{stats['reused']}（共 {len(new_chunks)} 块）"
    )
    return stats


async def _full_rebuild_chunks(
    document: Document, chunks: list[str], user_id: int, db: AsyncSession
) -> dict:
    """全量重建：删旧文档所有块 + 重新 add 全部新块（含 content_hash）。

    增量更新的降级路径（变化率超阈值或旧数据无 content_hash 时触发）。
    """
    async with _chroma_lock:
        collection = get_user_collection(user_id)
        # 删旧
        collection.delete(where={"document_id": document.id})
        # 重算全部 + 入库
        embeddings = await encode_texts(chunks)
        ids = [f"{document.id}_chunk_{i}" for i in range(len(chunks))]
        metadatas = _build_chunk_metadata(user_id, document.id, document.filename, chunks)
        collection.add(ids=ids, documents=chunks, embeddings=embeddings, metadatas=metadatas)

    document.chunk_count = len(chunks)
    await db.commit()
    return {"added": len(chunks), "removed": len(chunks), "reused": 0, "degraded": True}


async def delete_document(document: Document, db: AsyncSession) -> None:
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
