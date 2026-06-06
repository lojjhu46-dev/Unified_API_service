FROM python:3.11-slim

WORKDIR /app

# 安装依赖
COPY services/docker/xlsx_mcp/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制共享模块和服务代码
COPY services/mcp_common /app/services/mcp_common
COPY services/xlsx_mcp /app/services/xlsx_mcp

# 环境变量
ENV MCP_ALLOWED_DIR=/data/uploads,/data/personal_uploads
ENV LOG_LEVEL=INFO

EXPOSE 9101

CMD ["uvicorn", "services.xlsx_mcp.main:app", "--host", "0.0.0.0", "--port", "9101"]
