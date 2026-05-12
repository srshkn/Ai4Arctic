# Ai4Arctic/Makefile

.PHONY: help env jwt-keys dev local prod down clean

# Docker
COMPOSE=docker compose
BASE=-f compose.yml

# .ENV
ENV_FILE=backend/.env
ENV_EXAMPLE=backend/.env.example

# JWT
PRIVATE_KEY=backend/certs/private.pem
PUBLIC_KEY=backend/certs/public.pem

help:
	@echo "Usage:"
	@echo "  make env"
	@echo "  make jwt-keys"
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

dev: env jwt-keys
	$(COMPOSE) --env-file ./$(ENV_FILE) \
		$(BASE) \
		-f infra/compose/dev.yml \
		up --build

local: env jwt-keys
	$(COMPOSE) --env-file ./$(ENV_FILE) \
		$(BASE) \
		-f infra/compose/local.yml \
		up --build

#prod: env jwt-keys
#	$(COMPOSE) --env-file ./$(ENV_FILE) \
		$(BASE) \
		-f infra/compose/prod.yml \
		up --build -d

down:
	$(COMPOSE) down

clean:
	$(COMPOSE) down -v
	docker builder prune -af
