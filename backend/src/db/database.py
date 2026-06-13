from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from src.core import get_settings


def build_engine(db_url: str | None = None):
    if db_url is None:
        db_url = str(get_settings().ASYNC_DB_URL)
    return create_async_engine(db_url, echo=True)


engine = build_engine()


SessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async with SessionLocal() as session:
        yield session
