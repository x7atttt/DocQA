"""会话摘要服务测试。

验证 maybe_generate_summary 的核心行为：
1. 轮数不足阈值 → 不调 LLM（省调用）
2. 达阈值 → 调 LLM 生成摘要并写回 Conversation.summary
3. SETNX 锁防重复生成（已被占锁时跳过）
4. LLM 异常时不阻断（降级忽略）

DB 用 conftest 的 test.db，mock LLM 控制 chat 返回。
Redis 用真 Redis（同 test_cache.py 风格），靠 flushdb 隔离。

阈值/窗口参数用 patch.object 直接改 summary_service 模块级 settings 对象的属性：
该模块顶层 `settings = get_settings()` 已绑定单例，改 env + 清 lru_cache 对已导入模块
无效，故直接 patch 对象属性（patch.object 退出时自动还原）。
"""

from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio

from app.core.cache import _summary_lock_key
from app.core.database import async_session_factory
from app.models import Conversation, Message


@pytest_asyncio.fixture(scope="function")
async def _db():
    """初始化测试库（建表），用完清空。比 client fixture 轻量（不起 ASGI）。"""
    from app.core.database import Base, engine, init_db

    await init_db()
    yield
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


async def _seed_conversation_with_rounds(user_id: int, rounds: int) -> int:
    """建一个会话并塞入指定轮数的 user+assistant 消息，返回 conv_id。"""
    async with async_session_factory() as db:
        conv = Conversation(user_id=user_id, title="测试长会话")
        db.add(conv)
        await db.flush()
        for i in range(rounds):
            db.add(Message(conversation_id=conv.id, role="user", content=f"第{i+1}个问题"))
            db.add(Message(conversation_id=conv.id, role="assistant", content=f"第{i+1}个回答"))
        await db.commit()
        return conv.id


async def _get_summary(conv_id: int) -> str | None:
    async with async_session_factory() as db:
        conv = await db.get(Conversation, conv_id)
        return conv.summary if conv else None


@pytest.mark.asyncio
async def test_summary_below_threshold_skips_llm(_db):
    """轮数不足阈值时不调 LLM，summary 保持 None。"""
    from app.services.summary_service import maybe_generate_summary

    conv_id = await _seed_conversation_with_rounds(user_id=9001, rounds=3)  # 远低于默认阈值12

    with patch("app.services.summary_service.chat", new_callable=AsyncMock) as mock_chat:
        await maybe_generate_summary(conv_id)

    assert not mock_chat.called  # 没调 LLM
    assert await _get_summary(conv_id) is None  # 未生成摘要


@pytest.mark.asyncio
async def test_summary_at_threshold_generates(_db):
    """达阈值时调 LLM 生成摘要，写回 Conversation.summary。"""
    from app.services import summary_service
    from app.services.summary_service import maybe_generate_summary

    conv_id = await _seed_conversation_with_rounds(user_id=9002, rounds=2)
    with (
        patch.object(summary_service.settings, "summarize_round_threshold", 2),
        patch.object(summary_service.settings, "max_history_rounds", 1),  # 窗口外 = 全部 - 最近1轮
        patch(
            "app.services.summary_service.chat",
            new_callable=AsyncMock,
            return_value="会话讨论了加密与备份方案",
        ),
    ):
        await maybe_generate_summary(conv_id)

    summary = await _get_summary(conv_id)
    assert summary is not None
    assert "加密" in summary


@pytest.mark.asyncio
async def test_summary_lock_prevents_dup(_db):
    """已被占锁时跳过生成（不调 LLM）。"""
    from app.services import summary_service
    from app.services.summary_service import maybe_generate_summary

    conv_id = await _seed_conversation_with_rounds(user_id=9003, rounds=2)

    # 先手动占锁（模拟已有 worker 在生成）
    import redis.asyncio as aioredis

    r = aioredis.from_url("redis://localhost:6379/0", decode_responses=True)
    await r.set(_summary_lock_key(conv_id), "existing-token", ex=60)
    await r.aclose()

    with (
        patch.object(summary_service.settings, "summarize_round_threshold", 2),
        patch.object(summary_service.settings, "max_history_rounds", 1),
        patch("app.services.summary_service.chat", new_callable=AsyncMock) as mock_chat,
    ):
        await maybe_generate_summary(conv_id)

    assert not mock_chat.called  # 锁被占，跳过
    assert await _get_summary(conv_id) is None


@pytest.mark.asyncio
async def test_summary_llm_exception_does_not_raise(_db):
    """LLM 异常时不抛错（降级忽略），summary 保持 None。"""
    from app.services import summary_service
    from app.services.summary_service import maybe_generate_summary

    conv_id = await _seed_conversation_with_rounds(user_id=9004, rounds=2)

    with (
        patch.object(summary_service.settings, "summarize_round_threshold", 2),
        patch.object(summary_service.settings, "max_history_rounds", 1),
        patch(
            "app.services.summary_service.chat",
            new_callable=AsyncMock,
            side_effect=Exception("LLM 超时"),
        ),
    ):
        # 不应抛异常
        await maybe_generate_summary(conv_id)

    assert await _get_summary(conv_id) is None
