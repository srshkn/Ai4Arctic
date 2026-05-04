# Ai4Arctic/backend/Makefile

.PHONY: migrate run help

help:
	@echo "Usage:"
	@echo "  make migrate
	@echo "  make run

# Команда для миграций
migrate:
	alembic upgrade head

# Команда для запуска приложения
run: migrate
	uvicorn src.main:app --host 0.0.0.0 --port 8080