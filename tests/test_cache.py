import pytest

from app.core.cache import (
    _lock_key,
    _null_key,
    _qa_key,
    acquire_lock,
    get_cached_answer,
    release_lock,
    set_cached_answer,
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
