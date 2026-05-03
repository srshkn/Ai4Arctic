import os
from typing import AsyncGenerator

import pytest
import pytest_asyncio
from alembic import command
from alembic.config import Config
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from src.db import Base, get_session
from src.main import app

TEST_DATABASE_URL = (
    f"postgresql+asyncpg://{os.environ['POSTGRES_USER']}:"
    f"{os.environ['POSTGRES_PASSWORD']}@"
    f"{os.environ['POSTGRES_SERVER']}:"
    f"{os.environ['POSTGRES_PORT']}/"
    f"{os.environ['POSTGRES_DB']}"
)


@pytest.fixture(scope="session")
def apply_migrations():
    """Фикстура для применения миграций Alembic к тестовой БД."""
    # Указываем путь к alembic.ini (убедись, что путь корректный относительно запуска pytest)
    alembic_cfg = Config("alembic.ini")

    # Переопределяем URL БД в конфиге Alembic, чтобы он смотрел в тестовую базу, а не в основную
    alembic_cfg.set_main_option("sqlalchemy.url", TEST_DATABASE_URL)

    # Накатываем миграции до актуального состояния
    command.upgrade(alembic_cfg, "head")

    yield

    # После завершения всех тестов откатываем БД в ноль
    command.downgrade(alembic_cfg, "base")


@pytest_asyncio.fixture(scope="session")
async def engine():
    """Асинхронный движок зависит от фикстуры миграций."""
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)

    yield engine

    await engine.dispose()


@pytest_asyncio.fixture(autouse=True)
async def clean_db(engine):
    async with engine.begin() as conn:
        for table in reversed(Base.metadata.sorted_tables):
            await conn.execute(table.delete())


@pytest_asyncio.fixture
async def db_session(engine):
    session_factory = async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    async with session_factory() as session:
        yield session
        await session.rollback()


@pytest_asyncio.fixture(scope="function")
async def async_client(db_session: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    """
    Асинхронный HTTP-клиент, в котором реальная база данных
    подменена на тестовую.
    """

    # Функция для переопределения зависимости FastAPI
    def override_get_async_session():
        yield db_session

    app.dependency_overrides[get_session] = override_get_async_session

    # Используем ASGITransport для обхода необходимости поднимать реальный сервер
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        yield client

    # Очищаем переопределения после теста
    app.dependency_overrides.clear()


@pytest.fixture
def valid_user_data() -> dict:
    """Фикстура с валидным payload для регистрации."""
    return {"name": "test_user", "password": "secure_password_123"}


@pytest.fixture
def another_user_data() -> dict:
    return {"name": "another_user", "password": "super_secure_456"}
