from functools import lru_cache

from pydantic import Field, PostgresDsn, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # PostgreSQL
    POSTGRES_SERVER: str = Field(default=...)
    POSTGRES_USER: str = Field(default=...)
    POSTGRES_PASSWORD: str = Field(default=...)
    POSTGRES_PORT: int = Field(default=...)
    POSTGRES_DB: str = Field(default=...)

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


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    return settings
