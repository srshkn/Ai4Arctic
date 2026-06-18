import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from uuid import UUID

import jwt

from .settings import get_settings

settings = get_settings()


class TokenHelper:
    def create_access_token(self, user_id: UUID) -> str:
        """Формирует JWT access c типом access и истечением из настроек."""
        return self._create_token(
            user_id, "access", settings.ACCESS_TOKEN_EXPIRES_MINUTES
        )

    def hash_session_token(self, token: str) -> str:
        """Хэширует сессионный токен через SHA-256."""
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    def create_refresh_token(self) -> str:
        """Создает долгоживущий случайный refresh токен (не JWT)."""
        return secrets.token_urlsafe(48)

    def decode_token(self, token: str, expected_type: str) -> dict:
        """Декодирует JWT и проверяет совпадение типа."""
        payload = jwt.decode(
            token, settings.PUBLIC_KEY, algorithms=[settings.JWT_ALGORITHM]
        )
        if payload.get("type") != expected_type:
            raise jwt.InvalidTokenError("Invalid token type")
        return payload

    def _create_token(
        self, user_id: UUID, token_type: str, expires_minutes: int
    ) -> str:
        """Собирает JWT с указанным типом и временем жизни."""
        now = datetime.now(timezone.utc)
        payload = {
            "sub": str(user_id),
            "type": token_type,
            "iat": int(now.timestamp()),
            "exp": int((now + timedelta(minutes=expires_minutes)).timestamp()),
            "iss": settings.PROJECT_NAME,
        }
        return jwt.encode(
            payload, settings.PRIVATE_KEY, algorithm=settings.JWT_ALGORITHM
        )
