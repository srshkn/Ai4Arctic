from functools import lru_cache

from fastapi.security import OAuth2PasswordBearer
from passlib.context import CryptContext

oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="/v1/auth/login",
    auto_error=False,
)


class Security:
    def __init__(self):
        self.pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")

    def hash_password(self, password: str) -> str:
        return self.pwd_context.hash(password)

    def verify_password(self, password: str, password_hash: str) -> bool:
        return self.pwd_context.verify(password, password_hash)


@lru_cache
def get_security() -> Security:
    return Security()
