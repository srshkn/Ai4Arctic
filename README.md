# Ai4Arctic

Web-сервис мониторинга вечной мерзлоты.

## Содержание
- [Цель проекта](#цель-проекта)
- [Предварительные требования](#предварительные-требования)
    - [1. Docker](#1-docker)
    - [2. Make](#2-make)
- [Быстрый локальный старт](#быстрый-локальный-старт)
- [Доступ после старта](#доступ-после-старта)
- [Структура проекта](#структура-проекта)

Дополнительно:
- [Документация по Backend части](./docs/BACKEND.md)

## Цель проекта

Автоматизировать анализ состояния мерзлоты по спутниковым данным.

## Предварительные требования

Для максимально удобного и быстрого развертывания проекта локально, убедитесь, что установлены такие инструменты, как:

### 1. Docker
Используется для поднятия изолированного окружения. Для локальной разработки удобнее всего использовать **Docker Desktop**.

- **macOS:** [Скачать Docker Desktop для Mac](https://docs.docker.com/desktop/setup/install/mac-install/)
- **Windows:** [Скачать Docker Desktop для Windows](https://docs.docker.com/desktop/install/windows-install/)
- **Linux:** [Инструкции по установке для Linux](https://docs.docker.com/desktop/install/linux-install/) 

### 2. Make (опционально)
В проекте используется **Makefile**, который выступает единой точкой входа для всех базовых команд.

- **macOS:** Установлен по умолчанию в составе утилит разработчика. Если команда `make` не найдена, выполните в терминале:
```bash
xcode-select --install
```
- **Windows:** Утилита `make` можно установить, выполнил команду в **PowerShell** от имени администратора:
```powershell
winget install GnuWin32.Make
```
- **Linux (Ubuntu/Debian):** Устанавливается через терминал командой:
```bash
sudo apt update && sudo apt install make
```

## Быстрый локальный старт

**1. Клонируем проект**

```bash
git clone https://github.com/srshkn/Ai4Arctic.git
```

**2. Перемещаемся в директорию проекта**
```bash
cd Ai4Arctic
```

**3. Запуск проекта**

На этом этапе опционально можно запустить проект через Makefile или сразу через Docker (если Make не установлен).

- **Make**
```bash
make up
```

- **Docker**
```bash
cp backend/.env.example backend/.env
docker compose --env-file ./backend/.env up --build
```

## Доступ после старта
**API**
```
http://localhost:8080
```

**Swagger UI (OpenAPI)**
```
http://localhost:8080/docs
```

**ReDoc**
```
http://localhost:8080/redoc
```

**pgAdmin**
```
http://localhost:5050
```

## Структура проекта
```
Ai4Arctic/
│
├── .github/workflows        # CI/CD
│   └── backend-ci.yml
│
├── backend/
│
├── frontend/
│
├── docs/
│
├── infra/                   # инфраструктура
│   ├── composes/
│   └── nginx/
│
├── notebooks/               # исследование
│
├── .gitignore
├── compose.yml
├── Makefile
└── README.md
```
