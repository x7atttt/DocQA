import hashlib
import json
import logging
import random
import uuid

import redis.asyncio as aioredis

from app.config import get_settings

settings = get_settings()
logger = logging.getLogger("docqa.cache")

_redis: aioredis.Redis | None = None
_available: bool | None = None


async def get_redis() -> aioredis.Redis | None:
    global _redis, _available
    if _available is False:
        return None
    if _redis is None:
        try:
            _redis = aioredis.from_url(settings.redis_url, decode_responses=True)
            await _redis.ping()
            _available = True
        except Exception as e:
            logger.warning(f"Redis 不可用，缓存与限流降级：{e}")
            _available = False
            _redis = None
            return None
    return _redis


async def ping_redis() -> None:
    client = await get_redis()
    if client is None:
        raise RuntimeError("Redis 连接失败")


def _normalize(question: str) -> str:
    return question.strip().lower()


def _qa_key(user_id: int, question: str, conversation_id: int | None = None) -> str:
    h = hashlib.sha1(_normalize(question).encode("utf-8")).hexdigest()[:16]
    cid = conversation_id or 0  # 无会话 id 用 0 兜底（兼容）
    return f"qa:user_{user_id}:conv_{cid}:{h}"


def _null_key(user_id: int, question: str, conversation_id: int | None = None) -> str:
    return "null:" + _qa_key(user_id, question, conversation_id)


def _lock_key(user_id: int, question: str, conversation_id: int | None = None) -> str:
    return "lock:" + _qa_key(user_id, question, conversation_id)


async def get_cached_answer(
    user_id: int, question: str, conversation_id: int | None = None
) -> tuple[bool, dict | None]:
    client = await get_redis()
    if client is None:
        return False, None
    try:
        raw = await client.get(_qa_key(user_id, question, conversation_id))
        if raw is not None:
            try:
                return True, json.loads(raw)
            except json.JSONDecodeError:
                return True, {"answer": raw, "sources": []}
        if await client.exists(_null_key(user_id, question, conversation_id)):
            return True, {"answer": "", "sources": []}
    except Exception as e:
        logger.warning(f"Redis 读取失败：{e}")
    return False, None


async def set_cached_answer(
    user_id: int,
    question: str,
    answer: str,
    sources: list | None,
    conversation_id: int | None = None,
    reasoning: str | None = None,
    thinking: bool = False,
) -> None:
    client = await get_redis()
    if client is None:
        return
    try:
        if not answer:
            await client.set(
                _null_key(user_id, question, conversation_id), "1", ex=settings.cache_null_ttl_seconds
            )
            return
        payload = json.dumps(
            {
                "answer": answer,
                "sources": sources or [],
                "reasoning": reasoning or "",
                "thinking": thinking,
            },
            ensure_ascii=False,
        )
        base = settings.cache_ttl_seconds
        jitter = random.randint(-int(base * 0.2), int(base * 0.2))
        ttl = max(60, base + jitter)
        await client.set(_qa_key(user_id, question, conversation_id), payload, ex=ttl)
    except Exception as e:
        logger.warning(f"Redis 写入失败：{e}")


async def acquire_lock(
    user_id: int, question: str, conversation_id: int | None = None, expire: int = 15
) -> str | None:
    """成功返回 token 字符串（释放时校验）；失败返回 None。Redis 不可用返回伪 token。"""
    client = await get_redis()
    if client is None:
        return "no-redis"
    token = uuid.uuid4().hex
    try:
        ok = await client.set(_lock_key(user_id, question, conversation_id), token, nx=True, ex=expire)
        return token if ok else None
    except Exception as e:
        logger.warning(f"Redis 加锁失败：{e}")
        return "no-redis"


async def release_lock(
    user_id: int, question: str, token: str | None, conversation_id: int | None = None
) -> None:
    if not token:
        return
    if token == "no-redis":
        return
    client = await get_redis()
    if client is None:
        return
    try:
        script = (
            "if redis.call('GET', KEYS[1]) == ARGV[1] then "
            "return redis.call('DEL', KEYS[1]) else return 0 end"
        )
        await client.eval(script, 1, _lock_key(user_id, question, conversation_id), token)
    except Exception as e:
        logger.warning(f"Redis 释放锁失败：{e}")


# ============ 会话摘要专用锁 ============
# 与问答互斥锁（_lock_key，按 question hash）分离：摘要生成慢、按 conversation 维度防重复。
# 场景：同一会话连续多轮都达阈值，多次 BackgroundTasks 触发，靠此锁保证只有一个 worker 真正生成。

def _summary_lock_key(conv_id: int) -> str:
    return f"summary_lock:conv_{conv_id}"


async def acquire_summary_lock(conv_id: int, expire: int = 60) -> str | None:
    """摘要生成专用互斥锁。成功返回 token，已被占返回 None，Redis 不可用返回 'no-redis'。

    expire=60s：摘要生成含一次 LLM 调用（数秒级），给足余量避免生成中被误判超时释放。
    """
    client = await get_redis()
    if client is None:
        return "no-redis"
    token = uuid.uuid4().hex
    try:
        ok = await client.set(_summary_lock_key(conv_id), token, nx=True, ex=expire)
        return token if ok else None
    except Exception as e:
        logger.warning(f"Redis 摘要锁加锁失败：{e}")
        return "no-redis"


async def release_summary_lock(conv_id: int, token: str | None) -> None:
    if not token or token == "no-redis":
        return
    client = await get_redis()
    if client is None:
        return
    try:
        script = (
            "if redis.call('GET', KEYS[1]) == ARGV[1] then "
            "return redis.call('DEL', KEYS[1]) else return 0 end"
        )
        await client.eval(script, 1, _summary_lock_key(conv_id), token)
    except Exception as e:
        logger.warning(f"Redis 摘要锁释放失败：{e}")


# ============ 对话历史缓存 ============
# 热点会话每次 ask 都 SELECT 历史较浪费；缓存按 conversation 维度，
# 写消息后失效（新消息落地 → 缓存作废）。decode_responses=True 故值为 JSON 字符串。

def _history_key(conv_id: int) -> str:
    return f"history:conv_{conv_id}"


async def get_history_cache(conv_id: int) -> list[dict] | None:
    """读取会话历史缓存。命中返回列表（正序，最旧在前）；未命中/Redis不可用返回 None。"""
    client = await get_redis()
    if client is None:
        return None
    try:
        raw = await client.get(_history_key(conv_id))
        if raw is None:
            return None
        return json.loads(raw)
    except Exception as e:
        logger.warning(f"Redis 历史缓存读取失败：{e}")
        return None


async def set_history_cache(conv_id: int, history: list[dict]) -> None:
    """写入会话历史缓存，带 TTL 抖动（防雪崩）。"""
    client = await get_redis()
    if client is None:
        return
    try:
        payload = json.dumps(history, ensure_ascii=False)
        base = settings.history_cache_ttl_seconds
        jitter = random.randint(-int(base * 0.2), int(base * 0.2))
        ttl = max(60, base + jitter)
        await client.set(_history_key(conv_id), payload, ex=ttl)
    except Exception as e:
        logger.warning(f"Redis 历史缓存写入失败：{e}")


async def invalidate_history_cache(conv_id: int) -> None:
    """失效会话历史缓存（写消息/删消息后调用，避免脏读）。"""
    client = await get_redis()
    if client is None:
        return
    try:
        await client.delete(_history_key(conv_id))
    except Exception as e:
        logger.warning(f"Redis 历史缓存失效失败：{e}")
