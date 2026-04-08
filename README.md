# Ai4Arctic

Ai4Arctic — web-сервис мониторинга вечной мерзлоты.

## 🎯 Цель проекта

Автоматизировать анализ состояния мерзлоты по спутниковым данным.

## Быстрый старт

✍️

## Структура проекта
```
Ai4Arctic/
│
├── .github/workflows         # CI/CD
│   ├── ruff.yml
│   ├── test.yml
│   └── docker-ci.yml
│
├── backend/
│   ├── app/
│   │   ├── api/              # роуты (endpoints)
│   │   ├── core/             # конфиги, настройки, security
│   │   ├── models/           # ORM модели (SQLAlchemy)
│   │   ├── schemas/          # Pydantic схемы
│   │   ├── services/         # бизнес-логика
│   │   ├── db/               # подключение к БД, сессии, миграции
│   │   ├── gis/              # работа с GeoPandas и геоданными
│   │   ├── utils/            # вспомогательные функции
│   │   └── main.py           # точка входа FastAPI
│   │
│   ├── migrations/           # Alembic
│   ├── tests/                # тесты
│   ├── .python-version
│   ├── pyproject.toml
│   ├── uv.lock
│   ├── .dockerignore
│   └── Dockerfile
│
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── api/              # запросы к backend
│   │   ├── components/
│   │   ├── pages/
│   │   ├── features/         # (если используешь feature-based подход)
│   │   ├── hooks/
│   │   ├── utils/
│   │   └── main.(js|tsx)
│   │
│   ├── package.json
│   ├── .dockerignore
│   └── Dockerfile
│
├── database/
│   ├── init.sql             # начальная инициализация
│   └── seeds/               # тестовые данные
│
├── notebooks/               # иследование
│
├── docker-compose.yml
├── .env.example
├── .gitignore
└── README.md
```