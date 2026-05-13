from fastapi import APIRouter, HTTPException, Response, status

from src.api import DBManagerDep
from src.core import get_settings
from src.core.exceptions import (
    AppError,
    InvalidCredentialsError,
    RefreshTokenExpiredError,
    RefreshTokenNotFoundError,
    UserAlreadyExistsError,
    UserNotFoundError,
)
from src.schemas import LoginRequest, RefreshRequest, TokenPair, UserCreate, UserOut
from src.services import AuthService, UserService

from ..tags import Tags

settings = get_settings()


router = APIRouter(prefix="/auth", tags=[Tags.AUTH])


def _set_token_cookies(
    response: Response, access_token: str, refresh_token: str
) -> None:
    response.set_cookie(
        key=settings.ACCESS_COOKIE_NAME,
        value=access_token,
        httponly=False,
        secure=settings.SESSION_COOKIE_SECURE,
        samesite="lax",
        max_age=settings.ACCESS_TOKEN_EXPIRES_MINUTES * 60,
        domain=settings.SESSION_COOKIE_DOMAIN,
        path="/",
    )
    response.set_cookie(
        key=settings.REFRESH_COOKIE_NAME,
        value=refresh_token,
        httponly=False,
        secure=settings.SESSION_COOKIE_SECURE,
        samesite="lax",
        max_age=settings.REFRESH_TOKEN_EXPIRES_MINUTES * 60,
        domain=settings.SESSION_COOKIE_DOMAIN,
        path="/",
    )


# Регистрация
@router.post(
    "/register",
    status_code=status.HTTP_201_CREATED,
    response_model=UserOut,
    summary="Регистрация пользователя",
    description="Тут пользователь регистрируется.",
    name="auth_register",
)
async def register(data: UserCreate, db: DBManagerDep) -> UserOut:
    service = UserService(db)
    try:
        return await service.register(data.name, data.email, data.password)
    except UserAlreadyExistsError as err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="") from err
    except AppError as err:
        detail = str(err) or ""
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=detail
        ) from err


# Аутентификация
@router.post(
    "/login",
    status_code=status.HTTP_201_CREATED,
    response_model=TokenPair,
    summary="Аутентификация пользователя",
    description="Тут пользователь входит в сервис, будучи зарегистрированным.",
    name="auth_login",
)
async def login(data: LoginRequest, response: Response, db: DBManagerDep) -> TokenPair:
    jwt_service = AuthService(db)
    try:
        access_token, refresh_token = await jwt_service.login(data.name, data.password)
    except InvalidCredentialsError as err:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail=str(err)) from err
    except AppError as err:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(err)) from err
    _set_token_cookies(response, access_token, refresh_token)
    return TokenPair(
        access_token=access_token,
        refresh_token=refresh_token,
    )


@router.post("/token/refresh", summary="Обновление access/refresh токенов")
async def refresh_tokens(
    data: RefreshRequest, response: Response, db: DBManagerDep
) -> TokenPair:
    jwt_service = AuthService(db)
    try:
        pair = await jwt_service.refresh(data.refresh_token)
    except RefreshTokenExpiredError as err:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail=str(err)) from err
    except RefreshTokenNotFoundError as err:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail=str(err)) from err
    except UserNotFoundError as err:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail=str(err)) from err
    except AppError as err:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(err)) from err
    _set_token_cookies(response, pair.access_token, pair.refresh_token)
    return pair
