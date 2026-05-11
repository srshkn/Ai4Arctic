from contextlib import asynccontextmanager

from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware

from src.api import Tags, api_v1_router, tags_metadata
from src.core import get_settings

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(
    title=settings.PROJECT_NAME,
    description=settings.PROJECT_DESCRIPTION,
    version=settings.PROJECT_VERSION,
    openapi_tags=tags_metadata,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(api_v1_router)


@app.get(
    "/",
    tags=[Tags.META],
    status_code=status.HTTP_200_OK,
    summary="Информация о сервисе",
    description="Возвращает базовую информацию о сервисе: статус и доступность API.",
)
async def main():
    return {"status": "Everything is OK, Bro!"}


@app.get(
    "/health",
    tags=[Tags.META],
    status_code=status.HTTP_200_OK,
    summary="Проверка состояния сервиса",
    description="Проверяет, что сервис работает и отвечает на запросы. Используется системами мониторинга.",
)
async def health():
    return {"status": "ok"}
