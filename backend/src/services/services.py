from src.core import get_security
from src.core.exceptions import UserAlreadyExistsError
from src.db import DBManager

security = get_security()


class UserService:
    def __init__(self, db: DBManager):
        self.db = db

    async def register(self, name: str, password: str):
        existing = await self.db.users.get_user_name(name)
        if existing:
            raise UserAlreadyExistsError
        user = await self.db.users.create_user(
            name=name, password_hash=security.hash_password(password=password)
        )
        await self.db.session.commit()
        return user
