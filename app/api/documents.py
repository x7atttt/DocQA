from fastapi import APIRouter, Depends, File, Query, UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.core.database import get_db
from app.core.exceptions import BizError
from app.core.response import ResponseCode, success_response
from app.models import Document, User
from app.schemas.document import DocumentListData, DocumentOut
from app.services.document_service import (
    SUPPORTED_EXTS,
    delete_document,
    process_document,
    save_upload_file,
)

router = APIRouter()

MAX_FILE_SIZE = 20 * 1024 * 1024


@router.post("/upload")
async def upload(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    if not file.filename or "." not in file.filename:
        raise BizError(code=ResponseCode.UNSUPPORTED_FILE_TYPE, message="文件名非法", http_status=400)
    ext = file.filename.rsplit(".", 1)[-1].lower()
    if ext not in SUPPORTED_EXTS:
        raise BizError(
            code=ResponseCode.UNSUPPORTED_FILE_TYPE,
            message=f"仅支持 {', '.join(sorted(SUPPORTED_EXTS))}",
            http_status=400,
        )

    content = await file.read()
    if len(content) == 0:
        raise BizError(code=ResponseCode.DOC_UPLOAD_FAILED, message="文件为空", http_status=400)
    if len(content) > MAX_FILE_SIZE:
        raise BizError(code=ResponseCode.DOC_UPLOAD_FAILED, message="文件超过 20MB 限制", http_status=400)

    # 内容哈希查重（用户内）—— 在解析/向量化之前拦截，省 embedding 算力
    import hashlib

    file_hash = hashlib.sha256(content).hexdigest()
    existing = await db.execute(
        select(Document).where(
            Document.user_id == user.id,
            Document.file_hash == file_hash,
        )
    )
    if existing.scalar_one_or_none() is not None:
        raise BizError(
            code=ResponseCode.DOC_ALREADY_EXISTS,
            message="该文档已上传过，无需重复上传",
            http_status=409,
        )

    file_path, _ = save_upload_file(content, ext)
    try:
        document = await process_document(
            file_path=file_path,
            filename=file.filename,
            ext=ext,
            file_size=len(content),
            user_id=user.id,
            db=db,
            file_hash=file_hash,
        )
    except Exception:
        import os

        if os.path.exists(file_path):
            os.remove(file_path)
        raise

    return success_response(DocumentOut.model_validate(document).model_dump(mode="json"), "上传成功")


@router.get("/list")
async def list_documents(
    cursor: int | None = Query(default=None, description="上一页最后一条记录的ID"),
    limit: int = Query(default=20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    query = select(Document).where(Document.user_id == user.id).order_by(Document.id.desc())
    if cursor is not None:
        query = query.where(Document.id < cursor)
    query = query.limit(limit + 1)

    result = await db.execute(query)
    documents = result.scalars().all()
    has_next = len(documents) > limit
    documents = documents[:limit]
    next_cursor = documents[-1].id if has_next and documents else None

    data = DocumentListData(
        documents=[DocumentOut.model_validate(d) for d in documents],
        next_cursor=next_cursor,
        has_next=has_next,
    )
    return success_response(data.model_dump(mode="json"))


@router.delete("/{document_id}")
async def remove_document(
    document_id: int,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    result = await db.execute(
        select(Document).where(Document.id == document_id, Document.user_id == user.id)
    )
    document = result.scalar_one_or_none()
    if document is None:
        raise BizError(code=ResponseCode.DOC_NOT_FOUND, message="文档不存在", http_status=404)

    await delete_document(document, db)
    return success_response(None, "删除成功")
