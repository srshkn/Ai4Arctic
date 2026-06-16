# Ai4Arctic/shared/Makefile

# Устанавливает цель по умолчанию
.DEFAULT_GOAL := help

.PHONY: help proto-gen

# Path
PROTO_SCRIPT := ./scripts/generate_proto.sh
CONTRACT_DIR := ./contract
BACKEND_GRPC_DIR := ../backend/src/grpc_server/contract
MODEL_GRPC_DIR := ../model/grpc_client/contract

# Показывает список доступных команд
help:
	@echo "Ai4Arctic project"
	@echo "Usage:"
	@echo "  make proto-gen - ..."

proto-gen:
	@bash $(PROTO_SCRIPT)

	@mkdir -p $(BACKEND_GRPC_DIR)
	@cp $(CONTRACT_DIR)/*.py $(BACKEND_GRPC_DIR)/
	@cp $(CONTRACT_DIR)/*.py $(MODEL_GRPC_DIR)/

	@echo "Copied generated proto files to backend & model!"
