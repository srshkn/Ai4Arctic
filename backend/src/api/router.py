from fastapi import APIRouter

from .constants import API_V1_PREFIX
from .routers import user_auth

api_v1_router = APIRouter()
api_v1_router.include_router(user_auth, prefix=API_V1_PREFIX)
