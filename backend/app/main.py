from contextlib import asynccontextmanager

from app.core import get_settings
from app.database import create_db_and_tables
from fastapi import FastAPI

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


@app.get("/")
async def main():
    return {"status": "Everything is OK, Bro!"}
