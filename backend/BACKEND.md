```
backend/
│
├── src/
│   │
│   ├── api/
│   │   │
│   │   ├── routers/           # endpoints
│   │   │
│   │   ├── __init__.py
│   │   └── dependencies.py
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   └── settings.py
│   │
│   ├── db/                    # подключение к БД, сессии, миграции
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── database.py
│   │   └── db_manager.py
│   │
│   ├── migrations/            # миграции (Alembic)
│   │   │
│   │   ├── versions/
│   │   │
│   │   ├── __init__.py
│   │   ├── env.py
│   │   └── script.py.mako
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
│   │   ├── schemas.py
│   │   └── users.py
│   │
│   ├── services/               # бизнес-логика
│   │
│   ├── gis/                    # работа с GeoPandas и геоданными
│   │
│   ├── __init__.py
│   └── main.py                 # точка входа FastAPI
│
├── tests/                      # тесты
│   └── conftest.py 
│
├── .dockerignore
├── .env.example
├── .python-version
├── alembic.ini
├── BACKEND.md
├── Dockerfile
├── pyproject.toml
└── uv.lock
```