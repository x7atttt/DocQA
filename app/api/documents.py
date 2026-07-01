from fastapi import APIRouter, BackgroundTasks, Depends, File, Query, UploadFile
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
    create_pending_document,
    delete_document,
    process_pending_document,
    save_upload_file,
)

router = APIRouter()

MAX_FILE_SIZE = 20 * 1024 * 1024


@router.post("/upload")
async def upload(
    file: UploadFile = File(...),
    replace_id: int | None = None,
    background_tasks: BackgroundTasks = None,
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

    # 增量更新：用户在前端确认"更新同名文档"后，带 replace_id 重传
    # 不删旧文档（保留旧 chunks 供 diff），复用旧记录、更新 hash、置 pending
    pending_doc_id: int
    if replace_id is not None:
        old_doc = await db.execute(
            select(Document).where(Document.id == replace_id, Document.user_id == user.id)
        )
        old = old_doc.scalar_one_or_none()
        if old is None:
            raise BizError(code=ResponseCode.DOC_NOT_FOUND, message="待更新的文档不存在", http_status=404)
        # 复用旧文档记录：更新 file_hash/size/status，后台增量 diff chunks
        old.file_hash = file_hash
        old.file_size = len(content)
        old.status = "pending"
        await db.commit()
        pending_doc_id = old.id
    else:
        # 同名检测：无 replace_id 时，若同名文档存在且内容不同 → 返回冲突，等用户确认
        # （同名≠同文档，不自动覆盖；让用户显式确认，避免误伤同名不同文档）
        same_name = await db.execute(
            select(Document).where(
                Document.user_id == user.id,
                Document.filename == file.filename,
            )
        )
        same_doc = same_name.scalar_one_or_none()
        if same_doc is not None:
            # 能走到这里说明 hash 不同（上面已拦截相同 hash），即内容变了
            raise BizError(
                code=ResponseCode.DOC_SAME_NAME_CONFLICT,
                message=f"已存在同名文档「{file.filename}」，是否更新为最新版本？",
                http_status=409,
                data={"existing_id": same_doc.id, "filename": same_doc.filename},
            )
        # 新建 pending 记录
        document = await create_pending_document(
            filename=file.filename,
            ext=ext,
            file_size=len(content),
            user_id=user.id,
            db=db,
            file_hash=file_hash,
        )
        pending_doc_id = document.id

    file_path, _ = save_upload_file(content, ext)
    # 后台处理：replace_id 时走增量 diff，否则全量入库
    background_tasks.add_task(
        process_pending_document,
        document_id=pending_doc_id,
        file_path=file_path,
        ext=ext,
        user_id=user.id,
        replace_id=replace_id,
    )
    # 返回当前文档记录（refresh 拿到最新状态）
    doc_out = await db.get(Document, pending_doc_id)
    return success_response(
        DocumentOut.model_validate(doc_out).model_dump(mode="json"),
        "已接收，后台处理中",
    )


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
