import pytest

from app.core.cache import (
    _history_key,
    _lock_key,
    _null_key,
    _qa_key,
    acquire_lock,
    get_cached_answer,
    get_history_cache,
    invalidate_history_cache,
    release_lock,
    set_cached_answer,
    set_history_cache,
)


@pytest.mark.asyncio
async def test_cache_miss_then_hit():
    user_id = 999
    q = "测试问题"
    hit, data = await get_cached_answer(user_id, q)
    assert hit is False

    await set_cached_answer(user_id, q, "这是答案", [{"filename": "f.md"}])
    hit2, data2 = await get_cached_answer(user_id, q)
    assert hit2 is True
    assert data2["answer"] == "这是答案"
    assert data2["sources"][0]["filename"] == "f.md"


@pytest.mark.asyncio
async def test_null_cache_for_empty_answer():
    user_id = 998
    q = "空答案问题"
    await set_cached_answer(user_id, q, "", None)
    hit, data = await get_cached_answer(user_id, q)
    assert hit is True
    assert data["answer"] == ""


@pytest.mark.asyncio
async def test_lock_acquire_and_release():
    user_id = 997
    q = "锁测试"
    token_a = await acquire_lock(user_id, q, expire=5)
    assert token_a is not None
    token_b = await acquire_lock(user_id, q, expire=5)
    assert token_b is None
    await release_lock(user_id, q, token_a)
    token_c = await acquire_lock(user_id, q, expire=5)
    assert token_c is not None
    await release_lock(user_id, q, token_c)


@pytest.mark.asyncio
async def test_lock_release_with_wrong_token_does_not_delete():
    user_id = 996
    q = "token校验"
    token_a = await acquire_lock(user_id, q, expire=10)
    await release_lock(user_id, q, "wrong-token")
    import redis.asyncio as aioredis

    r = aioredis.from_url("redis://localhost:6379/0", decode_responses=True)
    async with r:
        still_locked = await r.exists(_lock_key(user_id, q))
    assert still_locked
    await release_lock(user_id, q, token_a)


@pytest.mark.asyncio
async def test_question_normalization():
    user_id = 995
    await set_cached_answer(user_id, "Hello World", "答案", None)
    hit, _ = await get_cached_answer(user_id, "  hello world  ")
    assert hit is True


@pytest.mark.asyncio
async def test_cache_isolation_per_user():
    await set_cached_answer(1, "共享问题", "用户1的答案", None)
    hit, data = await get_cached_answer(2, "共享问题")
    assert hit is False


# ============ 对话历史缓存测试 ============

@pytest.mark.asyncio
async def test_history_cache_miss_then_hit():
    """历史缓存 miss→set→hit 往返。"""
    conv_id = 88001
    cached = await get_history_cache(conv_id)
    assert cached is None  # 初始 miss

    history = [
        {"role": "user", "content": "问题1"},
        {"role": "assistant", "content": "回答1"},
    ]
    await set_history_cache(conv_id, history)
    cached = await get_history_cache(conv_id)
    assert cached == history  # 命中，内容一致


@pytest.mark.asyncio
async def test_history_cache_invalidation():
    """invalidate 后缓存失效（重新 miss）。"""
    conv_id = 88002
    await set_history_cache(conv_id, [{"role": "user", "content": "x"}])
    assert await get_history_cache(conv_id) is not None

    await invalidate_history_cache(conv_id)
    assert await get_history_cache(conv_id) is None  # 已失效


@pytest.mark.asyncio
async def test_history_cache_isolation_per_conversation():
    """会话间缓存隔离（不同 conv_id 互不污染）。"""
    await set_history_cache(88003, [{"role": "user", "content": "会话A"}])
    cached_b = await get_history_cache(88004)
    assert cached_b is None  # 会话B 读不到会话A 的缓存


@pytest.mark.asyncio
async def test_history_cache_key_format():
    """key 命名规范：history:conv_{id}。"""
    assert _history_key(123) == "history:conv_123"
