# Ai4Arctic/Makefile

# Устанавливает цель по умолчанию
.DEFAULT_GOAL := help

.PHONY: help env jwt-keys dev local prod down clean rev

# Docker
COMPOSE=docker compose
BASE=-f compose.yml

# .ENV
ENV_FILE=backend/.env
ENV_EXAMPLE=backend/.env.example

# JWT
PRIVATE_KEY=backend/certs/private.pem
PUBLIC_KEY=backend/certs/public.pem

# Показывает список доступных команд
help:
	@echo "Ai4Arctic project"
	@echo "Usage:"
	@echo "  make env - copy environment variables from .env.example."
	@echo "  make jwt-keys - generate JWT encryption keys (public key & private key)."
	@echo "  make dev - raise the project in development mode."
	@echo "  make local - launch the project locally."
	@echo "  make prod - bring the project into production."
	@echo "  make down - delete project container."
	@echo "  make clean - delete the project container along with volumes and cache."

# Создаёт .env файл из шаблона, если его нет
env:
	@if [ ! -f $(ENV_FILE) ]; then \
		echo "Creating .env from .env.example"; \
		cp $(ENV_EXAMPLE) $(ENV_FILE); \
	fi

# Генерирует JWT RSA ключи, если они отсутствуют
jwt-keys:
	@if [ -f $(PRIVATE_KEY) ] && [ -f $(PUBLIC_KEY) ]; then \
		echo "JWT keys already exist. Skipping generation."; \
	else \
		echo "Generating JWT RS256 keys..."; \
		mkdir -p backend/certs; \
		openssl genrsa -out $(PRIVATE_KEY) 2048; \
		openssl rsa -in $(PRIVATE_KEY) -pubout -out $(PUBLIC_KEY); \
		echo "JWT keys generated in backend/certs"; \
	fi

rev:
	$(COMPOSE) --env-file ./$(ENV_FILE) \
		$(BASE) \
		-f infra/compose/dev.yml \
		up

# Запускает проект в dev-режиме
dev: env jwt-keys
	$(COMPOSE) --env-file ./$(ENV_FILE) \
		$(BASE) \
		-f infra/compose/dev.yml \
		up --build

# Запускает проект локально
local: env jwt-keys
	$(COMPOSE) --env-file ./$(ENV_FILE) \
		$(BASE) \
		-f infra/compose/local.yml \
		up --build

# Запускает проект в production-режиме
#prod: env jwt-keys
#	$(COMPOSE) --env-file ./$(ENV_FILE) \
		$(BASE) \
		-f infra/compose/prod.yml \
		up --build -d

# Останавливает контейнеры проекта
down:
	$(COMPOSE) down

# Полностью очищает контейнеры, volume и docker-кэш
clean:
	$(COMPOSE) down -v
	docker builder prune -af
