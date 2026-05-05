from fastapi import APIRouter, HTTPException, status

from src.api import DBManagerDep
from src.core.exceptions import AppError, UserAlreadyExistsError
from src.schemas import UserCreate, UserOut
from src.services import UserService

from ..tags import Tags

router = APIRouter(prefix="/auth", tags=[Tags.AUTH])


# Регистрация
@router.post(
    "/register",
    status_code=status.HTTP_201_CREATED,
    response_model=UserOut,
    summary="Регистрация пользователя",
    description="Тут пользователь регистрируется.",
)
async def register(data: UserCreate, db: DBManagerDep) -> UserOut:
    service = UserService(db)
    try:
        return await service.register(data.name, data.password)
    except UserAlreadyExistsError as err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="") from err
    except AppError as err:
        detail = str(err) or ""
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=detail
        ) from err
