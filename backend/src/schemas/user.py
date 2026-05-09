from uuid import UUID

from pydantic import Field

from .config import APIModel


class UserCreate(APIModel):
    name: str = Field(min_length=5, max_length=20)
    password: str = Field(min_length=5, max_length=75)


class UserOut(APIModel):
    id: UUID
    name: str = Field(min_length=5, max_length=20)


class LoginRequest(APIModel):
    name: str = Field(min_length=5, max_length=20)
    password: str = Field(min_length=5, max_length=75)


class TokenPair(APIModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class RefreshRequest(APIModel):
    refresh_token: str
