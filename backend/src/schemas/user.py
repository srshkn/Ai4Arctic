from uuid import UUID

from pydantic import Field

from .config import APIModel


class UserCreate(APIModel):
    name: str = Field(min_length=5, max_length=20)
    password: str = Field(min_length=5, max_length=75)


class UserOut(APIModel):
    id: UUID
    name: str
