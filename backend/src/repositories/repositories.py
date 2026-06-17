import json
from datetime import datetime
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models import MagtFeature, RefreshToken, User


class UserRepository:
    """Логика работы с данными пользователя в базе данных"""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_user_name(self, name: str) -> Optional[User]:
        return await self.session.scalar(select(User).where(User.name == name))

    async def get_user_email(self, email: str) -> Optional[User]:
        return await self.session.scalar(select(User).where(User.email == email))

    async def get_user_id(self, user_id: int) -> Optional[User]:
        return await self.session.get(User, user_id)

    async def create_user(self, name: str, email: str, password_hash: str) -> User:
        user = User(name=name, email=email, password_hash=password_hash)
        self.session.add(user)
        await self.session.flush()
        return user


class AuthRepository:
    """Логика работы с JWT пользователя в базе данных"""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create_refresh_token(
        self, user_id: int, token_hash: str, expires_at: datetime
    ) -> RefreshToken:
        token = RefreshToken(
            user_id=user_id, token_hash=token_hash, expires_at=expires_at
        )
        self.session.add(token)
        await self.session.flush()
        return token

    async def get_refresh_token(self, token_hash: str) -> Optional[RefreshToken]:
        return await self.session.scalar(
            select(RefreshToken).where(RefreshToken.token_hash == token_hash)
        )

    async def delete_refresh_token(self, token_obj: RefreshToken) -> None:
        await self.session.delete(token_obj)


class GeoRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_row(self, year: int, class_id: int) -> bool:
        flag = await self.session.scalar(
            select(MagtFeature).where(
                MagtFeature.year == year,
                MagtFeature.class_id == class_id,
            )
        )
        if flag:
            return True
        else:
            return False

    async def add_geo_data(
        self,
        year: int,
        class_id: int,
        color: str,
        magt_min: float,
        magt_max: str,
        label: str,
        geometry: object,
    ) -> MagtFeature:
        geo_data = MagtFeature(
            year=year,
            class_id=class_id,
            color=color,
            magt_min=magt_min,
            magt_max=magt_max,
            label=label,
            geometry=json.dumps(geometry),
        )
        self.session.add(geo_data)
        await self.session.flush()
        return geo_data
