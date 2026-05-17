from uuid import UUID

from pydantic import EmailStr, Field, model_validator

from .config import APIModel


class UserCreate(APIModel):
    name: str = Field(min_length=5, max_length=20)
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    confirm_password: str = Field(min_length=8, max_length=128)

    @model_validator(mode="after")
    def check_passwords_match(self) -> "UserCreate":
        if self.password != self.confirm_password:
            raise ValueError("Пароли не совпадают")
        return self


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
