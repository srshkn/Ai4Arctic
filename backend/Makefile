# Ai4Arctic/backend/Makefile

# Устанавливает цель по умолчанию
.DEFAULT_GOAL := help

.PHONY: revision head run help

# Показывает список доступных команд
help:
	@echo "Backend of the Ai4Arctic project"
	@echo "Usage:"
	@echo "  make revision m='msg' - создать alembic migration."
	@echo "  make head - update migrations."
	@echo "  make run - run the backend."

# Создаёт новую Alembic migration
revision:
	@if [ -z "$(m)" ]; then \
		echo 'Usage: make revision m="migration message"'; \
		exit 1; \
	fi
	alembic revision --autogenerate -m "$(m)"

# Применяет все миграции до последней версии
head:
	alembic upgrade head

# Запускает backend-приложение
run: head
	uvicorn src.main:app --host 0.0.0.0 --port 8080