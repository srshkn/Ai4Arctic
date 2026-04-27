from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.routers import user_auth
from src.core import get_settings
from src.db import create_db_and_tables

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await create_db_and_tables()
    yield


app = FastAPI(
    title=settings.PROJECT_NAME,
    description=settings.PROJECT_DESCRIPTION,
    version=settings.PROJECT_VERSION,
    lifespan=lifespan,
)

app.include_router(user_auth)


@app.get("/")
async def main():
    return {"status": "Everything is OK, Bro!"}
