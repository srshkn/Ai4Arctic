from typing import Annotated
from uuid import UUID

import jwt
from fastapi import Depends, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.core import TokenHelper, get_settings
from src.core.security import oauth2_scheme
from src.db import DBManager, SessionLocal, get_session
from src.models import User

settings = get_settings()
tokens = TokenHelper()

SessionDep = Annotated[AsyncSession, Depends(get_session)]


async def get_db_manager():
    async with DBManager(SessionLocal) as manager:
        yield manager


DBManagerDep = Annotated[DBManager, Depends(get_db_manager)]


async def get_current_user(
    token: Annotated[str | None, Depends(oauth2_scheme)],
    request: Request,
    session: SessionDep,
) -> User:
    raw_token = _extract_access_token(token, request)
    try:
        payload = tokens.decode_token(raw_token, expected_type="access")
    except jwt.ExpiredSignatureError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Token expired") from None
    except jwt.InvalidTokenError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid token") from None
    user_id = UUID(payload["sub"])
    user = await session.get(User, user_id)
    if not user:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "User not found")
    return user


def _extract_access_token(token: str | None, request: Request) -> str:
    if token and token != "undefined":
        return token
    cookie_token = request.cookies.get(settings.ACCESS_COOKIE_NAME)
    if cookie_token:
        return cookie_token
    raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing access token")
