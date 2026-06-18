from typing import AsyncGenerator

import pytest
from alembic import command
from alembic.config import Config
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from testcontainers.postgres import PostgresContainer

from src.core import get_settings
from src.db import Base, get_session
from src.main import app

settings = get_settings()


@pytest.fixture
def fastapi_app():
    return app


@pytest.fixture(scope="session")
def postgres_container():

    if settings.TESTCONTAINER:
        container = PostgresContainer(
            "postgres:17",
            username=settings.POSTGRES_USER,
            password=settings.POSTGRES_PASSWORD,
            dbname=settings.POSTGRES_DB,
        )
        container.with_bind_ports(5432, settings.POSTGRES_PORT)

        container.start()

        yield container

        container.stop()
    else:
        yield None


@pytest.fixture(scope="session")
def test_db_url(postgres_container):
    if settings.TESTCONTAINER:
        sync_url = postgres_container.get_connection_url()

        async_url = sync_url.replace(
            "postgresql+psycopg2://",
            "postgresql+asyncpg://",
        )
    else:
        async_url = str(settings.ASYNC_DB_URL)

    return async_url


@pytest.fixture(scope="session")
async def engine(apply_migrations, test_db_url):
    """Асинхронный движок зависит от фикстуры миграций."""
    engine = create_async_engine(test_db_url, echo=False)

    yield engine

    await engine.dispose()


@pytest.fixture(scope="session", autouse=True)
def apply_migrations(postgres_container):
    """Фикстура для применения миграций Alembic к тестовой БД."""
    # Указываем путь к alembic.ini
    # (убедись, что путь корректный относительно запуска pytest)
    alembic_cfg = Config("alembic.ini")
    if settings.TESTCONTAINER:
        sync_url = postgres_container.get_connection_url()

        sync_url = sync_url.replace(
            "postgresql+psycopg2://",
            "postgresql+psycopg://",
        )
    else:
        sync_url = str(settings.SYNC_DB_URL)
    # Переопределяем URL БД в конфиге Alembic,
    # чтобы он смотрел в тестовую базу, а не в основную
    alembic_cfg.set_main_option("sqlalchemy.url", sync_url)

    # Накатываем миграции до актуального состояния
    command.downgrade(alembic_cfg, "base")
    command.upgrade(alembic_cfg, "head")
    yield

    # После завершения всех тестов откатываем БД в ноль
    command.downgrade(alembic_cfg, "base")


@pytest.fixture(autouse=True)
async def clean_db(engine, apply_migrations):
    async with engine.begin() as conn:
        for table in reversed(Base.metadata.sorted_tables):
            await conn.execute(table.delete())


@pytest.fixture
async def db_session(engine):
    session_factory = async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    async with session_factory() as session:
        yield session
        await session.rollback()


@pytest.fixture(scope="function")
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
    return {
        "name": "test_user",
        "email": "user@mail.com",
        "password": "secure_password_123",
        "confirm_password": "secure_password_123",
    }


@pytest.fixture
def another_user_data() -> dict:
    return {
        "name": "another_user",
        "email": "auser@mail.com",
        "password": "super_secure_456",
        "confirm_password": "super_secure_456",
    }
