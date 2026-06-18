#!/usr/bin/env bash
set -e

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

PROTO_DIR="$ROOT_DIR/proto"
OUT_DIR="$ROOT_DIR/contract"

mkdir -p "$OUT_DIR"

uv run python -m grpc_tools.protoc \
  -I "$PROTO_DIR" \
  --python_out="$OUT_DIR" \
  --grpc_python_out="$OUT_DIR" \
  "$PROTO_DIR"/*.proto

echo "Proto generated into $OUT_DIR"
