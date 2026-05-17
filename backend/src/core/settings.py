from functools import lru_cache
from pathlib import Path

from pydantic import PostgresDsn, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # App
    PROJECT_NAME: str
    PROJECT_DESCRIPTION: str
    PROJECT_VERSION: str

    # PostgreSQL
    POSTGRES_SERVER: str
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_PORT: int
    POSTGRES_DB: str

    # JWT
    JWT_PRIVATE_KEY_PATH: str
    JWT_PUBLIC_KEY_PATH: str
    JWT_ALGORITHM: str

    ACCESS_TOKEN_EXPIRES_MINUTES: int
    REFRESH_TOKEN_EXPIRES_MINUTES: int

    ACCESS_COOKIE_NAME: str
    REFRESH_COOKIE_NAME: str

    SESSION_COOKIE_SECURE: bool
    SESSION_COOKIE_DOMAIN: str | None = None
    JWT_ISSUER: str
    JWT_AUDIENCE: str

    PRIVATE_KEY: str | None = None
    PUBLIC_KEY: str | None = None

    # Frontend
    FRONTEND_URL: str

    @computed_field
    @property
    def DB_URL(self) -> PostgresDsn:
        return (
            f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
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
