from functools import lru_cache
from pathlib import Path

from pydantic import Field, PostgresDsn, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # App
    PROJECT_NAME: str = Field(default=...)
    PROJECT_DESCRIPTION: str = Field(default=...)
    PROJECT_VERSION: str = Field(default=...)

    # PostgreSQL
    POSTGRES_SERVER: str = Field(default=...)
    POSTGRES_USER: str = Field(default=...)
    POSTGRES_PASSWORD: str = Field(default=...)
    POSTGRES_PORT: int = Field(default=...)
    POSTGRES_DB: str = Field(default=...)

    # JWT
    JWT_PRIVATE_KEY_PATH: str = Field(default=...)
    JWT_PUBLIC_KEY_PATH: str = Field(default=...)
    JWT_ALGORITHM: str = Field(default=...)

    ACCESS_TOKEN_EXPIRES_MINUTES: int = Field(default=...)
    REFRESH_TOKEN_EXPIRES_MINUTES: int = Field(default=...)

    ACCESS_COOKIE_NAME: str = Field(default=...)
    REFRESH_COOKIE_NAME: str = Field(default=...)

    SESSION_COOKIE_SECURE: bool = Field(default=...)
    SESSION_COOKIE_DOMAIN: str | None = Field(default=None)

    JWT_ISSUER: str = Field(default=...)
    JWT_AUDIENCE: str = Field(default=...)

    PRIVATE_KEY: str = Field(default="")
    PUBLIC_KEY: str = Field(default="")

    # Frontend
    FRONTEND_URL: str = Field(default=...)

    # Testcontainer
    TESTCONTAINER: bool = Field(default=...)

    @computed_field
    @property
    def ASYNC_DB_URL(self) -> PostgresDsn:
        return PostgresDsn(
            f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_SERVER}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    @computed_field
    @property
    def SYNC_DB_URL(self) -> PostgresDsn:
        return PostgresDsn(
            f"postgresql+psycopg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_SERVER}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    model_config = SettingsConfigDict(
        env_file=".env", case_sensitive=True, extra="ignore"
    )

    def load_keys(self) -> None:
        self.PRIVATE_KEY = Path(self.JWT_PRIVATE_KEY_PATH).read_text()
        self.PUBLIC_KEY = Path(self.JWT_PUBLIC_KEY_PATH).read_text()


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.load_keys()
    return settings
