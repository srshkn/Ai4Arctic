from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from src.db import DBManager, SessionLocal, get_session

SessionDep = Annotated[AsyncSession, Depends(get_session)]


async def get_db_manager():
    async with DBManager(SessionLocal) as manager:
        yield manager


DBManagerDep = Annotated[DBManager, Depends(get_db_manager)]
