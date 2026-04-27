from typing import Callable

from sqlalchemy.ext.asyncio import AsyncSession

from src.repositories import UserRepository

from .database import SessionLocal


class DBManager:
    def __init__(self, session_factory: Callable[[], AsyncSession] = SessionLocal):
        self.session_factory = session_factory
        self.session: AsyncSession | None = None
        self.users: UserRepository | None = None

    async def __aenter__(self) -> "DBManager":
        self.session = self.session_factory()
        self.users = UserRepository(self.session)
        return self

    async def __aexit__(self, *args) -> None:
        if self.session:
            await self.session.rollback()
            await self.session.close()

    async def commit(self) -> None:
        if self.session:
            await self.session.commit()
