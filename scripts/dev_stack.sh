#!/usr/bin/env bash
# Local WSL development stack runner.
#
# Starts services in the stable order:
#   Redis / OpenSearch -> DOCX MCP / XLSX MCP -> main API -> Feishu WS worker
#
# Usage:
#   scripts/dev_stack.sh start
#   scripts/dev_stack.sh stop
#   scripts/dev_stack.sh status
#   scripts/dev_stack.sh logs api

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
RUNTIME_DIR="$PROJECT_DIR/.runtime/dev-stack"
LOG_DIR="$PROJECT_DIR/logs/dev-stack"
PYTHON_BIN="${PYTHON_BIN:-/root/.venvs/unified_api_service/bin/python}"

APP_HOST="${APP_HOST:-0.0.0.0}"
APP_PORT="${APP_PORT:-8000}"
DOCX_MCP_PORT="${DOCX_MCP_PORT:-9100}"
XLSX_MCP_PORT="${XLSX_MCP_PORT:-9101}"

START_REDIS="${DEV_REDIS_ENABLED:-true}"
START_OPENSEARCH="${DEV_OPENSEARCH_ENABLED:-true}"
START_DOCX_MCP="${DEV_DOCX_MCP_ENABLED:-true}"
START_XLSX_MCP="${DEV_XLSX_MCP_ENABLED:-true}"
START_API="${DEV_API_ENABLED:-true}"
START_FEISHU_WS="${DEV_FEISHU_WS_ENABLED:-true}"

mkdir -p "$RUNTIME_DIR" "$LOG_DIR"
cd "$PROJECT_DIR"

load_env() {
  if [[ -f .env ]]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
  fi

  export DOCUMENT_MCP_ENABLED="${DOCUMENT_MCP_ENABLED:-true}"
  export DOCX_MCP_BASE_URL="${DOCX_MCP_BASE_URL:-http://localhost:${DOCX_MCP_PORT}}"
  export XLSX_MCP_BASE_URL="${XLSX_MCP_BASE_URL:-http://localhost:${XLSX_MCP_PORT}}"
  export MCP_ALLOWED_DIR="${MCP_ALLOWED_DIR:-$PROJECT_DIR/data/uploads,$PROJECT_DIR/data/personal_uploads}"
  export LOG_LEVEL="${LOG_LEVEL:-INFO}"
}

pid_file() {
  echo "$RUNTIME_DIR/$1.pid"
}

is_running() {
  local pid="$1"
  [[ -n "$pid" ]] && kill -0 "$pid" >/dev/null 2>&1
}

read_pid() {
  local file
  file="$(pid_file "$1")"
  [[ -f "$file" ]] && cat "$file" || true
}

port_ready() {
  local port="$1"
  "$PYTHON_BIN" - "$port" <<'PY'
import socket
import sys

port = int(sys.argv[1])
with socket.socket() as sock:
    sock.settimeout(1)
    sys.exit(0 if sock.connect_ex(("127.0.0.1", port)) == 0 else 1)
PY
}

wait_for_port() {
  local name="$1"
  local port="$2"
  local timeout_seconds="${3:-30}"
  local start
  start="$(date +%s)"
  until port_ready "$port"; do
    if (( "$(date +%s)" - start >= timeout_seconds )); then
      echo "ERROR: $name did not become ready on port $port within ${timeout_seconds}s"
      return 1
    fi
    sleep 1
  done
}

wait_for_http() {
  local name="$1"
  local url="$2"
  local timeout_seconds="${3:-60}"
  local start
  start="$(date +%s)"
  until curl -fsS "$url" >/dev/null 2>&1; do
    if (( "$(date +%s)" - start >= timeout_seconds )); then
      echo "ERROR: $name did not become ready at $url within ${timeout_seconds}s"
      return 1
    fi
    sleep 2
  done
}

start_process() {
  local name="$1"
  local logfile="$2"
  shift 2

  local existing_pid
  existing_pid="$(read_pid "$name")"
  if is_running "$existing_pid"; then
    echo "$name already running (PID $existing_pid)"
    return 0
  fi

  echo "Starting $name..."
  nohup "$@" >"$logfile" 2>&1 &
  local pid=$!
  echo "$pid" >"$(pid_file "$name")"
  echo "$name started (PID $pid, log $logfile)"
}

stop_process() {
  local name="$1"
  local pid
  pid="$(read_pid "$name")"
  if ! is_running "$pid"; then
    rm -f "$(pid_file "$name")"
    echo "$name not running"
    return 0
  fi

  echo "Stopping $name (PID $pid)..."
  kill "$pid" >/dev/null 2>&1 || true
  for _ in $(seq 1 10); do
    if ! is_running "$pid"; then
      rm -f "$(pid_file "$name")"
      echo "$name stopped"
      return 0
    fi
    sleep 1
  done
  kill -9 "$pid" >/dev/null 2>&1 || true
  rm -f "$(pid_file "$name")"
  echo "$name force stopped"
}

start_docker_service() {
  local service="$1"
  echo "Starting docker compose service: $service"
  docker compose up -d "$service"
}

start_stack() {
  load_env
  mkdir -p data/uploads data/personal_uploads data/chroma logs

  if [[ "$START_REDIS" == "true" ]]; then
    if command -v redis-cli >/dev/null 2>&1 && redis-cli ping >/dev/null 2>&1; then
      echo "Redis already available"
    elif command -v systemctl >/dev/null 2>&1 && systemctl list-unit-files redis-server.service >/dev/null 2>&1; then
      echo "Starting redis-server via systemd..."
      systemctl start redis-server || start_docker_service redis
    else
      start_docker_service redis
    fi
  fi

  if [[ "$START_OPENSEARCH" == "true" ]]; then
    start_docker_service opensearch
    wait_for_http opensearch "http://localhost:9201/_cluster/health" 120
  fi

  if [[ "$START_DOCX_MCP" == "true" ]]; then
    start_process \
      docx-mcp \
      "$LOG_DIR/docx-mcp.log" \
      "$PYTHON_BIN" -m uvicorn services.docx_mcp.main:app --host 127.0.0.1 --port "$DOCX_MCP_PORT"
    wait_for_port docx-mcp "$DOCX_MCP_PORT" 30
  fi

  if [[ "$START_XLSX_MCP" == "true" ]]; then
    start_process \
      xlsx-mcp \
      "$LOG_DIR/xlsx-mcp.log" \
      "$PYTHON_BIN" -m uvicorn services.xlsx_mcp.main:app --host 127.0.0.1 --port "$XLSX_MCP_PORT"
    wait_for_port xlsx-mcp "$XLSX_MCP_PORT" 30
  fi

  if [[ "$START_API" == "true" ]]; then
    start_process \
      api \
      "$LOG_DIR/api.log" \
      "$PYTHON_BIN" -m uvicorn app.main:app --host "$APP_HOST" --port "$APP_PORT"
    wait_for_port api "$APP_PORT" 60
  fi

  if [[ "$START_FEISHU_WS" == "true" ]]; then
    start_process \
      feishu-ws \
      "$LOG_DIR/feishu-ws.log" \
      "$PYTHON_BIN" -m app.channels.feishu_ws_worker
  fi

  echo
  echo "Development stack started."
  status_stack
}

stop_stack() {
  stop_process feishu-ws
  stop_process api
  stop_process xlsx-mcp
  stop_process docx-mcp
  echo "Docker services are left running. Stop them with:"
  echo "  docker compose stop opensearch redis"
}

status_one() {
  local name="$1"
  local pid
  pid="$(read_pid "$name")"
  if is_running "$pid"; then
    echo "$name: running (PID $pid)"
  else
    echo "$name: stopped"
  fi
}

status_stack() {
  status_one docx-mcp
  status_one xlsx-mcp
  status_one api
  status_one feishu-ws
  if command -v redis-cli >/dev/null 2>&1 && redis-cli ping >/dev/null 2>&1; then
    echo "redis: available"
  else
    echo "redis: unavailable"
  fi
  if command -v curl >/dev/null 2>&1 && curl -fsS "http://localhost:9201" >/dev/null 2>&1; then
    echo "opensearch: available"
  else
    echo "opensearch: unavailable or disabled"
  fi
}

show_logs() {
  local name="${1:-api}"
  local file="$LOG_DIR/$name.log"
  if [[ ! -f "$file" ]]; then
    echo "No log file for $name: $file"
    exit 1
  fi
  tail -f "$file"
}

case "${1:-start}" in
  start)
    start_stack
    ;;
  stop)
    stop_stack
    ;;
  restart)
    stop_stack
    start_stack
    ;;
  status)
    load_env
    status_stack
    ;;
  logs)
    show_logs "${2:-api}"
    ;;
  *)
    echo "Usage: $0 {start|stop|restart|status|logs [api|docx-mcp|xlsx-mcp|feishu-ws]}"
    exit 1
    ;;
esac
