# BACKEND

## Структура
```
backend/
│
├── certs/                     #ключи JWT
│
├── migrations/                # миграции (Alembic)
│   │
│   ├── versions/
│   │
│   ├── env.py
│   └── script.py.mako
│
├── src/
│   │
│   ├── api/
│   │   │
│   │   ├── routers/           # endpoints
│   │   │   ├── __init__..py
│   │   │   └── auth.py
│   │   │
│   │   ├── __init__.py
│   │   ├── constants.py
│   │   ├── dependencies.py
│   │   ├── router.py
│   │   └── tags.py
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── exceptions.py
│   │   ├── security.py
│   │   └── settings.py
│   │
│   ├── db/                    # подключение к БД, сессии, миграции
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── database.py
│   │   └── db_manager.py
│   │
│   ├── models/                # ORM модели (SQLAlchemy)
│   │   ├── __init__.py
│   │   └── models.py
│   │
│   ├── repositories/
│   │   ├── __init__.py
│   │   └── repositories.py
│   │
│   ├── schemas/                # Pydantic схемы
│   │   ├── __init__.py
│   │   ├── config.py
│   │   └── users.py
│   │
│   ├── services/               # бизнес-логика
│   │   ├── __init__.py
│   │   └── services.py
│   │
│   ├── __init__.py
│   └── main.py                 # точка входа FastAPI
│
├── tests/                      # тесты
│   │
│   ├── test_endpoints/
│   │   └── test_auth.py
│   │
│   ├── conftest.py
│   └── test_mapper_config.py   
│
├── .dockerignore
├── .env.example
├── .python-version
├── alembic.ini
├── Dockerfile
├── Dockerfile.dev
├── Makefile
├── pyproject.toml
└── uv.lock
```
