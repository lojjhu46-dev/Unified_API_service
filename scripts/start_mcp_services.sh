#!/bin/bash
# 启动 MCP 服务脚本（本地开发用）

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# 设置环境变量
export MCP_ALLOWED_DIR="${MCP_ALLOWED_DIR:-./data/uploads,./data/personal_uploads}"
export LOG_LEVEL="${LOG_LEVEL:-INFO}"

echo "启动 DOCX MCP 服务 (端口 9100)..."
/root/.venvs/unified_api_service/bin/python -m uvicorn services.docx_mcp.main:app --host 127.0.0.1 --port 9100 &
DOCX_PID=$!

echo "启动 XLSX MCP 服务 (端口 9101)..."
/root/.venvs/unified_api_service/bin/python -m uvicorn services.xlsx_mcp.main:app --host 127.0.0.1 --port 9101 &
XLSX_PID=$!

echo "MCP 服务已启动："
echo "  - DOCX MCP: http://localhost:9100 (PID: $DOCX_PID)"
echo "  - XLSX MCP: http://localhost:9101 (PID: $XLSX_PID)"
echo ""
echo "按 Ctrl+C 停止所有服务"

# 等待子进程
trap "kill $DOCX_PID $XLSX_PID 2>/dev/null; exit" INT TERM
wait
