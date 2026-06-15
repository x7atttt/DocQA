import asyncio
import json

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
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
from app.schemas.chat import ChatAskRequest, ChatHistoryData, MessageOut, SourceItem
from app.services.chat_service import sse, stream_graph

router = APIRouter()


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
    body: ChatAskRequest,
    user: User = Depends(get_current_user),
):
    graph = request.app.state.graph
    question = body.question.strip()
    if not question:
        raise BizError(code=ResponseCode.EMPTY_QUESTION, message="问题不能为空", http_status=400)

    user_id = user.id
    thinking = bool(body.thinking)

    # 取最近若干轮历史作为多轮上下文（正序：最旧在前）
    history = await _load_recent_history(user_id, rounds=5)

    # 缓存命中时需匹配 thinking 模式（开启/关闭 thinking 的答案不共享）
    hit, cached = await get_cached_answer(user_id, question)
    if hit and cached and cached.get("answer") and bool(cached.get("thinking", False)) == thinking:
        return StreamingResponse(
            _stream_cached(cached, "hit"),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    lock_token: str | None = await acquire_lock(user_id, question)

    if lock_token is None:
        for _ in range(8):
            await asyncio.sleep(0.3)
            hit2, cached2 = await get_cached_answer(user_id, question)
            if hit2 and cached2 and cached2.get("answer") and bool(cached2.get("thinking", False)) == thinking:
                return StreamingResponse(
                    _stream_cached(cached2, "wait"),
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
                )
        lock_token = await acquire_lock(user_id, question)
        if lock_token is None:
            lock_token = "no-redis"

    async def event_stream():
        answer_parts: list[str] = []
        reasoning_parts: list[str] = []
        sources: list[dict] = []
        error_msg: str | None = None

        try:
            async for event_name, payload in stream_graph(graph, user_id, question, history, thinking):
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
                    thinking=thinking,
                )
            )

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _load_recent_history(user_id: int, rounds: int = 5) -> list[dict]:
    """读取当前用户最近 N 轮历史消息，返回正序（最旧在前）。

    用于多轮上下文。注意：当前 history 端点是跨会话按时间倒序的消息流，
    这里同样按全局最近消息取，不区分 conversation。
    """
    try:
        async with async_session_factory() as db:
            # 取最近 rounds*2 条（user+assistant 各一），倒序取再反转成正序
            stmt = (
                select(Message)
                .join(Conversation, Message.conversation_id == Conversation.id)
                .where(Conversation.user_id == user_id)
                .order_by(Message.id.desc())
                .limit(rounds * 2)
            )
            result = await db.execute(stmt)
            msgs = result.scalars().all()
        # 反转为正序，并只保留 user/assistant 的文本内容
        history = [
            {"role": m.role, "content": m.content}
            for m in reversed(msgs)
            if m.role in ("user", "assistant") and m.content
        ]
        return history
    except Exception:
        return []


async def _finalize(
    user_id: int,
    question: str,
    answer_parts: list[str],
    reasoning_parts: list[str] | None,
    sources: list[dict],
    error_msg: str | None,
    lock_token: str | None,
    thinking: bool = False,
) -> None:
    answer = "".join(answer_parts)
    reasoning = "".join(reasoning_parts) if reasoning_parts else ""
    if lock_token:
        await release_lock(user_id, question, lock_token)
    if error_msg or not answer:
        return
    try:
        await set_cached_answer(user_id, question, answer, sources, reasoning, thinking)
    except Exception:
        pass
    try:
        async with async_session_factory() as db:
            conv = Conversation(user_id=user_id, title=question[:50])
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
    except Exception:
        pass


@router.get("/history")
async def history(
    cursor: int | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    query = (
        select(Message)
        .join(Conversation, Message.conversation_id == Conversation.id)
        .where(Conversation.user_id == user.id)
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
                created_at=m.created_at,
            )
        )

    data = ChatHistoryData(messages=out, next_cursor=next_cursor, has_next=has_next)
    return success_response(data.model_dump(mode="json"))
