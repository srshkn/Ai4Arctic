# Ai4Arctic Frontend — Интерактивная карта вечной мерзлоты

React 18 + Vite + TypeScript + MapLibre GL JS + Tailwind CSS приложение для визуализации данных об изменении территорий вечной мерзлоты на карте России.

## 📋 Содержание

- [Технологии](#технологии)
- [Быстрый старт](#быстрый-старт)
- [Разработка](#разработка)
- [Сборка и развёртывание](#сборка-и-развёртывание)
- [Структура проекта](#структура-проекта)
- [API](#api)
- [Аутентификация](#аутентификация)
- [Конфигурация](#конфигурация)

## 🛠 Технологии

| Компонент | Технология |
|-----------|------------|
| Framework | React 18 + Vite |
| Язык | TypeScript |
| Карта | MapLibre GL JS |
| Стилизация | Tailwind CSS |
| Состояние | Zustand |
| Формы | React Hook Form + Zod |
| HTTP | Axios |
| Роутинг | React Router v6 |
| Сборка | Docker + Nginx |

## 🚀 Быстрый старт

### Локальная разработка

```bash
# 1. Перейти в директорию frontend
cd frontend

# 2. Установить зависимости
npm install

# 3. Создать файл .env на основе .env.example
cp .env.example .env

# 4. Запустить dev-сервер
npm run dev
```

Приложение будет доступно по адресу: `http://localhost:5173`

### Вход в систему

Используйте учётные данные администратора:
- **Email:** `admin@ai4arctic.com`
- **Пароль:** `admin1`

## 🔧 Разработка

### Запуск с бекендом через Docker Compose

```bash
# В одной терминале (из корня проекта)
cd ..
docker compose up --build

# В другом терминале (frontend)
npm run dev
```

### Структура проекта

```
frontend/
├── src/
│   ├── api/                    # HTTP-клиент и API функции
│   │   ├── client.ts           # Axios instance с interceptor
│   │   ├── auth.ts             # Функции авторизации
│   │   └── permafrost.ts       # Функции получения данных
│   ├── components/             # React компоненты
│   │   ├── ControlsPanel.tsx   # Панель управления (zoom, год, легенда)
│   │   ├── MapContainer.tsx    # Обёртка над MapLibre картой
│   │   ├── PermafrostLayer.tsx # Слой вечной мерзлоты
│   │   └── ProtectedRoute.tsx  # HOC для защищённых маршрутов
│   ├── pages/                  # Страницы приложения
│   │   ├── LoginPage.tsx       # Авторизация
│   │   ├── RegisterPage.tsx    # Регистрация
│   │   └── MapPage.tsx         # Главная страница с картой
│   ├── store/                  # Zustand хранилища
│   │   ├── authStore.ts        # Состояние авторизации
│   │   └── mapStore.ts         # Состояние карты (год, zoom)
│   ├── types/                  # TypeScript типы
│   │   ├── index.ts            # Основные типы
│   │   └── maplibre.d.ts       # Типы для MapLibre
│   ├── styles/
│   │   └── globals.css         # Глобальные стили (Tailwind)
│   ├── App.tsx                 # Корневой компонент с роутингом
│   ├── main.tsx                # Точка входа
│   └── vite-env.d.ts           # Типы для Vite env
├── public/
│   └── data/                   # Моковые GeoJSON данные
│       └── permafrost_2024.geojson
├── Dockerfile                  # Многоступенчатый Dockerfile
├── nginx.conf                  # Конфигурация Nginx
├── docker-compose.yml          # Docker Compose для фронтенда
├── package.json
├── tsconfig.json
├── vite.config.ts
├── tailwind.config.js
└── .env.example
```

## 📦 Сборка и развёртывание

### Docker Build

```bash
# Собрать образ
docker build -t ai4arctic-frontend:latest .

# Запустить контейнер
docker run -p 3000:80 ai4arctic-frontend:latest
```

### Docker Compose (Frontend + Backend)

```yaml
# docker-compose.yml (в корне проекта)
services:
  frontend:
    build: ./frontend
    ports:
      - "3000:80"
    depends_on:
      - backend
  
  backend:
    build: ./backend
    ports:
      - "8000:8000"

networks:
  app:
    driver: bridge
```

```bash
docker compose up --build
```

## 🔌 API

### Endpoints

| Метод | Endpoint | Описание |
|-------|----------|----------|
| POST | `/api/auth/register` | Регистрация нового пользователя |
| POST | `/api/auth/login` | Авторизация |
| GET | `/api/permafrost/{year}` | Получить GeoJSON за год |

### Примеры запросов

```bash
# Регистрация
curl -X POST http://localhost:8000/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "securepass"}'

# Авторизация
curl -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "admin@ai4arctic.com", "password": "admin1"}'

# Данные вечной мерзлоты
curl http://localhost:8000/api/permafrost/2024
```

## 🔐 Аутентификация

- JWT-токен сохраняется в `localStorage`
- Автоматически добавляется к каждому запросу через Axios interceptor
- Защищённые маршруты проверяют токен и перенаправляют на логин
- Кнопка «Выход» сбрасывает сессию и очищает localStorage

## ⚙️ Конфигурация

### Переменные окружения

| Переменная | Описание | По умолчанию |
|------------|----------|--------------|
| `VITE_API_BASE_URL` | URL бекенда | `http://localhost:8000/api` |
| `VITE_MAP_TILES_URL` | URL тайлов карты | OpenStreetMap |
| `VITE_MAP_MIN_ZOOM` | Минимальный зум | `3` |
| `VITE_MAP_MAX_ZOOM` | Максимальный зум | `12` |
| `VITE_YEAR_MIN` | Минимальный год | `1990` |
| `VITE_YEAR_MAX` | Максимальный год | `2025` |
| `VITE_YEAR_DEFAULT` | Год по умолчанию | `2024` |

## 🎨 Цвета легенды мерзлоты

| Тип | Цвет |
|-----|------|
| Continuous (Сплошная) | `#1a5276` — Тёмно-синий |
| Discontinuous (Прерывистая) | `#2e86c1` — Синий |
| Isolated (Островная) | `#85c1e9` — Светло-синий |
| Relict (Реликтовая) | `#aed6f1` — Бледно-голубой |

## 📝 Скрипты

```bash
npm run dev          # Запуск dev-сервера
npm run build        # Сборка для production
npm run preview      # Предпросмотр сборки
npm run lint         # Проверка кода