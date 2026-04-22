# Ai4Arctic

Ai4Arctic — web-сервис мониторинга вечной мерзлоты.

## 🎯 Цель проекта

Автоматизировать анализ состояния мерзлоты по спутниковым данным.

## Быстрый старт

### 1
```bash
cd backend
```

### 2
```bash
cp .env.example .env
```

### 3
```bash
docker compose --env-file ./backend/.env up --build
```

✍️

## Структура проекта
```
Ai4Arctic/
│
├── .github/workflows        # CI/CD
│   └── backend.yml
│
├── backend/
│
├── frontend/
│
├── database/
│   ├── init.sql             # начальная инициализация
│   └── seeds/               # тестовые данные
│
├── notebooks/               # исследование
│
├── .gitignore
├── docker-compose.yml
└── README.md
```