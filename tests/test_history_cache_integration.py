"""对话历史缓存集成测试。

验证 _load_recent_history 的缓存优先 + 回填 + 失效链路（DB ↔ Redis）：
1. 首次读：DB 未命中缓存 → 查 DB → 回填 Redis
2. 二次读：命中 Redis（不查 DB）
3. 失效后：重新查 DB

复用 test_summary.py 的 _db fixture 模式（init_db + drop_all）。
"""

from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio

from app.api.chat import _load_history_from_db, _load_recent_history
from app.core.cache import get_history_cache, invalidate_history_cache
from app.core.database import async_session_factory
from app.models import Conversation, Message


@pytest_asyncio.fixture(scope="function")
async def _db():
    from app.core.database import Base, engine, init_db

    await init_db()
    yield
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


async def _seed(user_id: int, conv_id: int | None, rounds: int) -> int:
    """建会话（若 conv_id 为 None 则新建）+ 塞消息，返回 conv_id。"""
    async with async_session_factory() as db:
        if conv_id is None:
            conv = Conversation(user_id=user_id, title="t")
            db.add(conv)
            await db.flush()
            conv_id = conv.id
        for i in range(rounds):
            db.add(Message(conversation_id=conv_id, role="user", content=f"q{i}"))
            db.add(Message(conversation_id=conv_id, role="assistant", content=f"a{i}"))
        await db.commit()
        return conv_id


@pytest.mark.asyncio
async def test_load_history_caches_then_backfills(_db):
    """首次读查 DB 并回填缓存，二次读命中缓存。"""
    conv_id = await _seed(user_id=77001, conv_id=None, rounds=2)

    # 清空缓存确保首次 miss
    await invalidate_history_cache(conv_id)
    # 首次读：应查 DB
    with patch("app.api.chat._load_history_from_db", wraps=_load_history_from_db) as spy_db:
        history1 = await _load_recent_history(77001, conv_id)
        assert spy_db.called  # 首次查了 DB
    assert len(history1) == 4  # 2 轮 = 4 条

    # 缓存应已回填
    cached = await get_history_cache(conv_id)
    assert cached is not None
    assert len(cached) == 4

    # 二次读：应命中缓存（不再查 DB）
    with patch("app.api.chat._load_history_from_db", new_callable=AsyncMock) as mock_db:
        history2 = await _load_recent_history(77001, conv_id)
        assert not mock_db.called  # 命中缓存，没查 DB
    assert history2 == history1


@pytest.mark.asyncio
async def test_load_history_invalidation_forces_db_reload(_db):
    """失效缓存后，下次读重新查 DB。"""
    conv_id = await _seed(user_id=77002, conv_id=None, rounds=1)
    await _load_recent_history(77002, conv_id)  # 首次回填
    assert await get_history_cache(conv_id) is not None

    # 失效
    await invalidate_history_cache(conv_id)
    assert await get_history_cache(conv_id) is None

    # 再读 → 重新查 DB（验证缓存被清后能重新读到非空历史并回填）
    history = await _load_recent_history(77002, conv_id)
    assert len(history) == 2  # 1 轮 = 2 条
    assert await get_history_cache(conv_id) is not None  # 再次回填


@pytest.mark.asyncio
async def test_load_history_no_conversation_id_skips_cache(_db):
    """conversation_id 为 None（兼容旧调用）时不走缓存，直接查 DB。"""
    await _seed(user_id=77003, conv_id=None, rounds=1)

    # conversation_id=None：应直接查 DB，不读/不写缓存
    history = await _load_recent_history(77003, None)
    assert len(history) == 2
