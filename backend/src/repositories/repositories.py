from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from src.models import User


class UserRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_user_name(self, name: str) -> Optional[User]:
        return await self.session.scalar(select(User).where(User.name == name))

    async def get_user_id(self, user_id: int) -> Optional[User]:
        return await self.session.get(User, user_id)

    async def create_user(self, name: str, password_hash: str) -> User:
        user = User(name=name, password_hash=password_hash)
        self.session.add(user)
        await self.session.flush()
        return user
