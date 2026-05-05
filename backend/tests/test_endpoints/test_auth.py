from fastapi import status
from httpx import AsyncClient


async def test_register_user_success(
    async_client: AsyncClient, app, valid_user_data: dict
):
    """
    Тест успешной регистрации пользователя (HTTP 201).
    """
    url = app.url_path_for("auth_register")

    response = await async_client.post(url, json=valid_user_data)

    assert response.status_code == status.HTTP_201_CREATED
    data = response.json()

    # Проверяем, что вернулись правильные данные (согласно UserOut)
    assert data["name"] == valid_user_data["name"]
    assert "id" in data
    # Убеждаемся, что хеш пароля или сам пароль не утекает в ответ
    assert "password" not in data


async def test_register_user_already_exists(
    async_client: AsyncClient, app, valid_user_data: dict
):
    """
    Тест попытки регистрации с уже существующим именем пользователя (HTTP 400).
    """
    url = app.url_path_for("auth_register")
    # 1. Успешно регистрируем пользователя
    first_response = await async_client.post(url, json=valid_user_data)
    assert first_response.status_code == status.HTTP_201_CREATED

    # 2. Пытаемся зарегистрировать точно такого же пользователя еще раз
    second_response = await async_client.post(url, json=valid_user_data)

    assert second_response.status_code == status.HTTP_400_BAD_REQUEST
    # В твоем коде стоит detail="", поэтому проверяем пустую строку.
    # (Рекомендуется возвращать осмысленный текст, например: "User already exists")
    assert second_response.json()["detail"] == ""


async def test_register_user_validation_error(async_client: AsyncClient, app):
    """
    Тест обработки невалидных данных с помощью Pydantic (HTTP 422).
    """
    url = app.url_path_for("auth_register")
    # Отправляем payload без обязательного поля password
    invalid_data = {"name": "test_user"}

    response = await async_client.post(url, json=invalid_data)

    # FastAPI перехватывает ошибку валидации Pydantic и отдает 422 до вызова тела функции
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    errors = response.json()["detail"]
    assert errors[0]["loc"] == ["body", "password"]
    assert errors[0]["msg"] == "Field required"
