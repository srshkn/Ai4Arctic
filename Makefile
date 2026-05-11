# Ai4Arctic/Makefile

.PHONY: help env dev local prod down clean

COMPOSE=docker compose
BASE=-f compose.yml
ENV_FILE=backend/.env
ENV_EXAMPLE=backend/.env.example

help:
	@echo "Usage:"
	@echo "  make env"
	@echo "  make dev"
	@echo "  make local"
	@echo "  make prod"
	@echo "  make down"
	@echo "  make clean"

env:
	@if [ ! -f $(ENV_FILE) ]; then \
		echo "Creating .env from .env.example"; \
		cp $(ENV_EXAMPLE) $(ENV_FILE); \
	fi

dev: env
	$(COMPOSE) --env-file ./$(ENV_FILE) \
		$(BASE) \
		-f infra/compose/dev.yml \
		up --build

local: env
	$(COMPOSE) --env-file ./$(ENV_FILE) \
		$(BASE) \
		-f infra/compose/local.yml \
		up --build

prod: env
	$(COMPOSE) --env-file ./$(ENV_FILE) \
		$(BASE) \
		-f infra/compose/prod.yml \
		up --build -d

down:
	$(COMPOSE) down

clean:
	$(COMPOSE) down -v
	docker builder prune -af
