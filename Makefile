# Ai4Arctic/Makefile

.PHONY: up down clean help

help:
	@echo "Usage:"
	@echo "  make up"
	@echo "  make down"
	@echo "  make clean"

up:
	docker compose --env-file ./backend/.env up --build

down:
	docker compose down

clean:
	docker compose down -v
