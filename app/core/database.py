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


async def get_db():
    async with async_session_factory() as session:
        yield session
