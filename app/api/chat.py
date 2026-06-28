import asyncio
import json

from fastapi import APIRouter, BackgroundTasks, Depends, Query, Request
from fastapi.responses import StreamingResponse
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.config import get_settings
from app.core.cache import (
    acquire_lock,
    get_cached_answer,
    release_lock,
    set_cached_answer,
)
from app.core.database import async_session_factory, get_db
from app.core.exceptions import BizError
from app.core.rate_limit import limiter
from app.core.response import ResponseCode, success_response
from app.models import Conversation, Message, User
from app.schemas.chat import (
    ChatAskRequest,
    ChatHistoryData,
    ConversationListData,
    ConversationOut,
    MessageOut,
    SourceItem,
)
from app.services.chat_service import sse, stream_graph

router = APIRouter()
settings = get_settings()


def _stream_cached(cached: dict, cache_tag: str):
    async def gen():
        if cached.get("sources"):
            yield sse("sources", cached["sources"])
        # 缓存命中时回放完整 reasoning（如有）
        reasoning = cached.get("reasoning")
        if reasoning:
            yield sse("reasoning", reasoning)
        if cached.get("answer"):
            for i in range(0, len(cached["answer"]), 4):
                yield sse("token", cached["answer"][i : i + 4])
        yield sse("done", {"status": "ok", "cache": cache_tag})

    return gen()


@router.post("/ask")
@limiter.limit("100/minute")
async def ask(
    request: Request,
    background_tasks: BackgroundTasks,
    body: ChatAskRequest,
    user: User = Depends(get_current_user),
):
    graph = request.app.state.graph
    question = body.question.strip()
    if not question:
        raise BizError(code=ResponseCode.EMPTY_QUESTION, message="问题不能为空", http_status=400)

    user_id = user.id
    thinking = bool(body.thinking)
    conversation_id = body.conversation_id

    # 取当前会话最近若干轮历史作为多轮上下文（正序：最旧在前）
    # 实际窗口由节点层按 token 预算 + 轮数截断
    history = await _load_recent_history(user_id, conversation_id)
    # 加载会话摘要（长对话老上下文压缩），注入 system prompt 做长期记忆
    summary = await _load_summary(conversation_id)

    # 缓存按会话隔离（key 含 conversation_id）
    hit, cached = await get_cached_answer(user_id, question, conversation_id)
    if hit and cached and cached.get("answer") and bool(cached.get("thinking", False)) == thinking:
        return StreamingResponse(
            _stream_cached(cached, "hit"),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    lock_token: str | None = await acquire_lock(user_id, question, conversation_id)

    if lock_token is None:
        for _ in range(8):
            await asyncio.sleep(0.3)
            hit2, cached2 = await get_cached_answer(user_id, question, conversation_id)
            if hit2 and cached2 and cached2.get("answer") and bool(cached2.get("thinking", False)) == thinking:
                return StreamingResponse(
                    _stream_cached(cached2, "wait"),
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
                )
        lock_token = await acquire_lock(user_id, question, conversation_id)
        if lock_token is None:
            lock_token = "no-redis"

    async def event_stream():
        answer_parts: list[str] = []
        reasoning_parts: list[str] = []
        sources: list[dict] = []
        error_msg: str | None = None

        try:
            async for event_name, payload in stream_graph(
                graph, user_id, question, history, thinking, summary
            ):
                yield sse(event_name, payload)
                if event_name == "token":
                    answer_parts.append(payload)
                elif event_name == "reasoning":
                    reasoning_parts.append(payload)
                elif event_name == "sources":
                    sources = payload
                elif event_name == "answer_final":
                    # 用流末的权威完整答案/推理覆盖（避免流式拼接遗漏）
                    if isinstance(payload, dict):
                        if payload.get("answer"):
                            answer_parts = [payload["answer"]]
                        if payload.get("reasoning"):
                            reasoning_parts = [payload["reasoning"]]
                elif event_name == "error":
                    error_msg = payload.get("message", "未知错误")
        except asyncio.CancelledError:
            raise
        finally:
            await asyncio.shield(
                _finalize(
                    user_id=user_id,
                    question=question,
                    answer_parts=answer_parts,
                    reasoning_parts=reasoning_parts,
                    sources=sources,
                    error_msg=error_msg,
                    lock_token=lock_token,
                    background_tasks=background_tasks,
                    thinking=thinking,
                    conversation_id=conversation_id,
                )
            )

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _load_recent_history(
    user_id: int,
    conversation_id: int | None,
    rounds: int | None = None,
) -> list[dict]:
    """读取指定会话最近 N 轮历史消息，返回正序（最旧在前）。

    默认取 summarize_round_threshold 轮（比生成窗口更宽），交由节点层按
    token 预算 + 轮数截断——既满足生成时的窗口控制，也为会话摘要预留
    看到窗口外老对话的能力。conversation_id 为 None 时取全局最近消息（兼容旧调用）。
    """
    if rounds is None:
        rounds = settings.summarize_round_threshold
    try:
        async with async_session_factory() as db:
            stmt = (
                select(Message)
                .join(Conversation, Message.conversation_id == Conversation.id)
                .where(Conversation.user_id == user_id)
                .order_by(Message.id.desc())
                .limit(rounds * 2)
            )
            if conversation_id is not None:
                stmt = stmt.where(Message.conversation_id == conversation_id)
            result = await db.execute(stmt)
            msgs = result.scalars().all()
        history = [
            {"role": m.role, "content": m.content}
            for m in reversed(msgs)
            if m.role in ("user", "assistant") and m.content
        ]
        return history
    except Exception:
        return []


async def _load_summary(conversation_id: int | None) -> str | None:
    """读取会话摘要（长对话老上下文压缩）。无会话或未生成摘要返回 None。"""
    if conversation_id is None:
        return None
    try:
        async with async_session_factory() as db:
            conv = await db.get(Conversation, conversation_id)
            return conv.summary if conv else None
    except Exception:
        return None


async def _finalize(
    user_id: int,
    question: str,
    answer_parts: list[str],
    reasoning_parts: list[str] | None,
    sources: list[dict],
    error_msg: str | None,
    lock_token: str | None,
    background_tasks: BackgroundTasks | None = None,
    thinking: bool = False,
    conversation_id: int | None = None,
) -> None:
    answer = "".join(answer_parts)
    reasoning = "".join(reasoning_parts) if reasoning_parts else ""
    if lock_token:
        await release_lock(user_id, question, lock_token, conversation_id)
    if error_msg or not answer:
        return
    try:
        await set_cached_answer(user_id, question, answer, sources, conversation_id, reasoning, thinking)
    except Exception:
        pass
    try:
        async with async_session_factory() as db:
            # 复用指定会话（校验 user_id 防越权），否则新建
            conv_id: int
            if conversation_id is not None:
                conv = await db.get(Conversation, conversation_id)
                if conv and conv.user_id == user_id:
                    # 首问时把"新对话"更新为问题摘要
                    if conv.title == "新对话":
                        conv.title = question[:20]
                    conv_id = conv.id
                else:
                    # 会话不存在或越权 → 新建
                    conv = Conversation(user_id=user_id, title=question[:20])
                    db.add(conv)
                    await db.flush()
                    conv_id = conv.id
            else:
                conv = Conversation(user_id=user_id, title=question[:20])
                db.add(conv)
                await db.flush()
                conv_id = conv.id
            db.add(
                Message(conversation_id=conv_id, role="user", content=question, sources=None)
            )
            db.add(
                Message(
                    conversation_id=conv_id,
                    role="assistant",
                    content=answer,
                    sources=json.dumps(sources, ensure_ascii=False) if sources else None,
                    reasoning=reasoning or None,
                )
            )
            await db.commit()
            # 消息落库后，异步判断是否需要生成/刷新会话摘要（达阈值才真正调 LLM）
            # 放进 BackgroundTasks：用户已收到流式回答，摘要生成不阻塞请求
            if background_tasks is not None:
                from app.services.summary_service import maybe_generate_summary

                background_tasks.add_task(maybe_generate_summary, conv_id)
    except Exception:
        pass


@router.get("/history")
async def history(
    conversation_id: int = Query(..., description="会话 ID"),
    cursor: int | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    # 校验会话归属当前用户
    conv = await db.get(Conversation, conversation_id)
    if not conv or conv.user_id != user.id:
        raise BizError(code=ResponseCode.CONVERSATION_NOT_FOUND, message="会话不存在", http_status=404)

    query = (
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.id.desc())
    )
    if cursor is not None:
        query = query.where(Message.id < cursor)
    query = query.limit(limit + 1)

    result = await db.execute(query)
    messages = result.scalars().all()
    has_next = len(messages) > limit
    messages = messages[:limit]
    next_cursor = messages[-1].id if has_next and messages else None

    out: list[MessageOut] = []
    for m in messages:
        sources_list: list[SourceItem] = []
        if m.sources:
            try:
                raw = json.loads(m.sources)
                sources_list = [SourceItem(**s) for s in raw]
            except Exception:
                sources_list = []
        out.append(
            MessageOut(
                id=m.id,
                role=m.role,
                content=m.content,
                sources=sources_list,
                reasoning=m.reasoning,
                created_at=m.created_at,
            )
        )

    data = ChatHistoryData(messages=out, next_cursor=next_cursor, has_next=has_next)
    return success_response(data.model_dump(mode="json"))


# ---------- 会话管理 ----------


@router.get("/conversations")
async def list_conversations(
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """列出当前用户所有会话，按 updated_at 倒序。"""
    result = await db.execute(
        select(Conversation)
        .where(Conversation.user_id == user.id)
        .order_by(Conversation.updated_at.desc())
    )
    convs = result.scalars().all()
    out = [ConversationOut.model_validate(c) for c in convs]
    data = ConversationListData(conversations=out, total=len(out))
    return success_response(data.model_dump(mode="json"))


@router.post("/conversations")
async def create_conversation(
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """新建空会话。达上限则拒绝创建。"""
    # 检查上限
    cnt_result = await db.execute(
        select(func.count()).select_from(Conversation).where(Conversation.user_id == user.id)
    )
    count = cnt_result.scalar() or 0
    if count >= settings.max_conversations:
        raise BizError(
            code=ResponseCode.CONVERSATION_LIMIT_EXCEEDED,
            message=f"会话已达上限（{settings.max_conversations} 个），请先删除旧会话",
            http_status=409,
        )
    conv = Conversation(user_id=user.id, title="新对话")
    db.add(conv)
    await db.commit()
    await db.refresh(conv)
    return success_response(ConversationOut.model_validate(conv).model_dump(mode="json"), "创建成功")


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: int,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """删除会话及其所有消息（外键无 CASCADE，手动先删 Message）。"""
    conv = await db.get(Conversation, conversation_id)
    if not conv or conv.user_id != user.id:
        raise BizError(code=ResponseCode.CONVERSATION_NOT_FOUND, message="会话不存在", http_status=404)
    # 先删该会话所有消息，再删会话（防孤儿消息）
    await db.execute(delete(Message).where(Message.conversation_id == conversation_id))
    await db.delete(conv)
    await db.commit()
    return success_response(None, "删除成功")
