# Ai4Arctic/Makefile

.PHONY: help env up down clean

ENV_FILE=backend/.env
ENV_EXAMPLE=backend/.env.example

help:
	@echo "Usage:"
	@echo "  make env"
	@echo "  make up"
	@echo "  make down"
	@echo "  make clean"

env:
	@if [ ! -f $(ENV_FILE) ]; then \
		echo "Creating .env from .env.example"; \
		cp $(ENV_EXAMPLE) $(ENV_FILE); \
	fi

up: env
	docker compose --env-file ./backend/.env up --build

down:
	docker compose down

clean:
	docker compose down -v
	docker builder prune -af
