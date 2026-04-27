class AppError(Exception):
    """Базовая ошибка приложения."""


class UserAlreadyExistsError(AppError):
    """Пользователь уже существует."""
