# Ai4Arctic/Makefile

# Устанавливает цель по умолчанию
#.DEFAULT_GOAL := help

PYTHON := python

PROTO_SCRIPT := ./scripts/generate_proto.sh

CONTRACT_DIR := ./contract
BACKEND_GRPC_DIR := ../backend/src/grpc_server/contract
MODEL_GRPC_DIR := ../model/grpc_client/contract

proto-gen:
	@bash $(PROTO_SCRIPT)

	@mkdir -p $(BACKEND_GRPC_DIR)
	@cp $(CONTRACT_DIR)/*.py $(BACKEND_GRPC_DIR)/
	@cp $(CONTRACT_DIR)/*.py $(MODEL_GRPC_DIR)/

	@echo "Copied generated proto files to backend & model!"
