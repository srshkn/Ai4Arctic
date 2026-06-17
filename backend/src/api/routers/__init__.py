from .auth import router as user_auth
from .geo import router as geo_magt

__all__ = ["user_auth", "geo_magt"]
