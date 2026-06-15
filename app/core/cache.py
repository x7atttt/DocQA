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


def _qa_key(user_id: int, question: str) -> str:
    h = hashlib.sha1(_normalize(question).encode("utf-8")).hexdigest()[:16]
    return f"qa:user_{user_id}:{h}"


def _null_key(user_id: int, question: str) -> str:
    return "null:" + _qa_key(user_id, question)


def _lock_key(user_id: int, question: str) -> str:
    return "lock:" + _qa_key(user_id, question)


async def get_cached_answer(user_id: int, question: str) -> tuple[bool, dict | None]:
    client = await get_redis()
    if client is None:
        return False, None
    try:
        raw = await client.get(_qa_key(user_id, question))
        if raw is not None:
            try:
                return True, json.loads(raw)
            except json.JSONDecodeError:
                return True, {"answer": raw, "sources": []}
        if await client.exists(_null_key(user_id, question)):
            return True, {"answer": "", "sources": []}
    except Exception as e:
        logger.warning(f"Redis 读取失败：{e}")
    return False, None


async def set_cached_answer(
    user_id: int,
    question: str,
    answer: str,
    sources: list | None,
    reasoning: str | None = None,
    thinking: bool = False,
) -> None:
    client = await get_redis()
    if client is None:
        return
    try:
        if not answer:
            await client.set(
                _null_key(user_id, question), "1", ex=settings.cache_null_ttl_seconds
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
        await client.set(_qa_key(user_id, question), payload, ex=ttl)
    except Exception as e:
        logger.warning(f"Redis 写入失败：{e}")


async def acquire_lock(user_id: int, question: str, expire: int = 15) -> str | None:
    """成功返回 token 字符串（释放时校验）；失败返回 None。Redis 不可用返回伪 token。"""
    client = await get_redis()
    if client is None:
        return "no-redis"
    token = uuid.uuid4().hex
    try:
        ok = await client.set(_lock_key(user_id, question), token, nx=True, ex=expire)
        return token if ok else None
    except Exception as e:
        logger.warning(f"Redis 加锁失败：{e}")
        return "no-redis"


async def release_lock(user_id: int, question: str, token: str | None) -> None:
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
        await client.eval(script, 1, _lock_key(user_id, question), token)
    except Exception as e:
        logger.warning(f"Redis 释放锁失败：{e}")
