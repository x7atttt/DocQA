import os
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///./data/test.db"
os.environ["JWT_SECRET_KEY"] = "test-secret-key"
os.environ["CHROMA_PERSIST_DIR"] = "./data/chroma_test"


@pytest.fixture(autouse=True)
def _reset_cache_state():
    """每个测试前重置 cache 模块全局状态（同步部分）。"""
    from app.core import cache as cache_module

    cache_module._redis = None
    cache_module._available = None
    yield
    cache_module._redis = None
    cache_module._available = None


@pytest_asyncio.fixture(autouse=True)
async def _flush_redis_after():
    """每个测试后清空 Redis，避免跨用例污染。"""
    yield
    try:
        import redis.asyncio as aioredis

        r = aioredis.from_url("redis://localhost:6379/0", decode_responses=True)
        async with r:
            await r.flushdb()
    except Exception:
        pass


@pytest_asyncio.fixture(scope="function")
async def client() -> AsyncIterator[AsyncClient]:
    import shutil

    from app.core.database import Base, engine, init_db
    from app.main import app

    os.makedirs("data", exist_ok=True)
    await init_db()

    from app.agent.graph import build_graph

    if not getattr(app.state, "graph", None):
        app.state.graph = build_graph()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

    if os.path.exists("data/chroma_test"):
        shutil.rmtree("data/chroma_test", ignore_errors=True)
