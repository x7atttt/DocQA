import asyncio
import os
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///./data/test.db"
os.environ["JWT_SECRET_KEY"] = "test-secret-key"


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="function")
async def client() -> AsyncIterator[AsyncClient]:
    from app.core.database import Base, engine, init_db
    from app.main import app

    await init_db()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
