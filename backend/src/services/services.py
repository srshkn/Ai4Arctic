from datetime import datetime, timedelta, timezone

from src.core import TokenHelper, get_security, get_settings
from src.core.exceptions import (
    EmailAlreadyExistsError,
    InvalidCredentialsError,
    RefreshTokenExpiredError,
    RefreshTokenNotFoundError,
    UserAlreadyExistsError,
    UserNotFoundError,
)
from src.db import DBManager
from src.schemas import TokenPair

settings = get_settings()
security = get_security()
tokens = TokenHelper()


class UserService:
    """Класс функционал пользователя"""

    def __init__(self, db: DBManager):
        self.db = db

    async def register(self, name: str, email: str, password: str):
        existing_name = await self.db.users.get_user_name(name)
        existing_email = await self.db.users.get_user_email(email)
        if existing_name:
            raise UserAlreadyExistsError
        elif existing_email:
            raise EmailAlreadyExistsError
        user = await self.db.users.create_user(
            name=name,
            email=email,
            password_hash=security.hash_password(password=password),
        )
        await self.db.session.commit()
        return user


class AuthService:
    """Класс функционал JWT"""

    def __init__(self, db: DBManager):
        self.db = db

    async def login(self, name: str, password: str):
        user = await self._get_user_or_raise(name, password)
        pair = await self._issue_tokens(user.id)
        return pair.access_token, pair.refresh_token

    async def refresh(self, raw_refresh_token: str):
        stored = await self._get_valid_refresh(raw_refresh_token)
        user = await self._get_user_for_token(stored.user_id)
        await self.db.auth.delete_refresh_token(stored)
        pair = await self._issue_tokens(user.id)
        return pair

    async def _get_valid_refresh(self, raw_refresh_token: str):
        token_hash = tokens.hash_session_token(raw_refresh_token)
        stored = await self.db.auth.get_refresh_token(token_hash)
        if not stored or stored.revoked:
            raise RefreshTokenNotFoundError
        now = datetime.now(timezone.utc)
        if stored.expires_at <= now:
            await self.db.auth.delete_refresh_token(stored)
            raise RefreshTokenExpiredError
        return stored

    async def _get_user_for_token(self, user_id: int):
        user = await self.db.users.get_user_id(user_id)
        if not user:
            raise UserNotFoundError
        return user

    async def _get_user_or_raise(self, name: str, password: str):
        user = await self.db.users.get_user_name(name)
        if not user or not security.verify_password(password, user.password_hash):
            raise InvalidCredentialsError
        return user

    def _refresh_expiry(self) -> datetime:
        return datetime.now(timezone.utc) + timedelta(
            minutes=settings.REFRESH_TOKEN_EXPIRES_MINUTES
        )

    async def _issue_tokens(self, user_id: int) -> TokenPair:
        access_token = tokens.create_access_token(user_id)
        refresh_token = tokens.create_refresh_token()
        refresh_hash = tokens.hash_session_token(refresh_token)
        expires_at = self._refresh_expiry()
        await self.db.auth.create_refresh_token(
            user_id=user_id, token_hash=refresh_hash, expires_at=expires_at
        )
        await self.db.session.commit()
        return TokenPair(access_token=access_token, refresh_token=refresh_token)
