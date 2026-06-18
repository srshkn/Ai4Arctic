from fastapi import APIRouter

from .constants import API_V1_PREFIX
from .routers import geo_magt, user_auth

all_router = APIRouter()
all_router.include_router(user_auth, prefix=API_V1_PREFIX)
all_router.include_router(geo_magt, prefix=API_V1_PREFIX)
