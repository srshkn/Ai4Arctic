from fastapi import APIRouter

from .constants import API_V1_PREFIX
from .routers import user_auth

all_router = APIRouter()
all_router.include_router(user_auth, prefix=API_V1_PREFIX)
