from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import text

from app.config import get_settings


class Base(DeclarativeBase):
    pass


settings = get_settings()

engine = create_async_engine(settings.database_url, echo=False, future=True)
async_session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def init_db() -> None:
    import app.models  # noqa: F401

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_documents_user_id ON documents(user_id)"))
        await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_messages_created_at ON messages(created_at)"))
        # 轻量迁移：为旧库补 reasoning 列（SQLite 支持 IF NOT EXISTS 语义通过 try 兜底）
        await _ensure_column(conn, "messages", "reasoning", "TEXT")


async def _ensure_column(conn, table: str, column: str, ddl_type: str) -> None:
    """SQLite 没有 ADD COLUMN IF NOT EXISTS，用 PRAGMA 探测列是否存在后决定是否加。"""
    result = await conn.execute(text(f"PRAGMA table_info({table})"))
    existing = {row[1] for row in result.fetchall()}
    if column not in existing:
        await conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {ddl_type}"))


async def get_db():
    async with async_session_factory() as session:
        yield session
