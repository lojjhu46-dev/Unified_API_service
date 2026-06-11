# Unified API Service

统一Agentic RAG API服务，整合文档检索、会话记忆、联网搜索和工具调用能力。

## 技术栈

* FastAPI
* Pydantic v2
* AsyncOpenAI
* DeepSeek API
* Chroma
* Redis
* Serper.dev

## 快速开始

推荐在 WSL 的 Linux 原生虚拟环境中运行和测试，不要激活 Windows venv 映射路径：

```bash
# 不推荐：这是 Windows Python，不是真正的 WSL/Linux Python
source /mnt/d/C/python/new_python/.venv/Scripts/activate
```

推荐做法：

```bash
# 创建 WSL/Linux venv，建议放在 WSL ext4 文件系统，避免 /mnt/d 大量小文件写入过慢
python3 -m venv /root/.venvs/unified_api_service

# 进入项目
cd /mnt/d/LLM/Unified_API_service

# 安装依赖
/root/.venvs/unified_api_service/bin/python -m pip install -r requirements.txt

# 复制配置文件
cp .env.example .env

# 启动服务
/root/.venvs/unified_api_service/bin/python -m app.main
```

验证当前解释器应显示 Linux：

```bash
/root/.venvs/unified_api_service/bin/python -c "import sys, platform; print(sys.executable); print(platform.system()); print(sys.platform)"
```

期望输出类似：

```text
/root/.venvs/unified_api_service/bin/python
Linux
linux
```

运行测试：

```bash
cd /mnt/d/LLM/Unified_API_service
/root/.venvs/unified_api_service/bin/python -m pytest
```

本地开发可用一键脚本按稳定顺序启动 Redis/OpenSearch、DOCX MCP、XLSX MCP、主应用和飞书长连接 worker：

```bash
cd /mnt/d/LLM/Unified_API_service
scripts/dev_stack.sh start
scripts/dev_stack.sh status
scripts/dev_stack.sh stop
```

## API文档

启动服务后访问 http://localhost:8000/docs

## LLM 配置

正常文档问答、RAG 和联网综合回答必须使用真实 LLM，例如：

```env
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=your_deepseek_api_key_here
```

`LLM_PROVIDER=mock` 仅用于本地调试和单元测试；文档问答会拒绝 mock，避免向用户返回 `[MOCK回答]`。

## API 认证

生产环境必须配置 `API_KEY`。配置后，`/ask` 调用方需要在请求头中传入可信身份：

```bash
curl -X POST http://localhost:8000/ask \
  -H "Authorization: Bearer your_api_key_here" \
  -H "X-User-Id: ou_xxx" \
  -H "X-Channel: api" \
  -H "Content-Type: application/json" \
  -d '{"user_id":"legacy_field","question":"你好"}'
```

`user_id` 和 `channel` 请求体字段仅为兼容保留，不作为权限身份；服务会使用 `X-User-Id` 和 `X-Channel` 注入可信身份。

`/documents/upload` 同样需要 API Key 和 `X-User-Id`。上传到个人知识库时，owner 来自 `X-User-Id`，表单中的 `owner_open_id` 仅为本地兼容保留，生产认证模式下不会作为权限身份。

```bash
curl -X POST http://localhost:8000/documents/upload \
  -H "Authorization: Bearer your_api_key_here" \
  -H "X-User-Id: ou_xxx" \
  -F "knowledge_base_type=personal" \
  -F "file=@example.txt"
```

生产环境如需浏览器跨域访问，请配置可信前端域名；默认不开放任意 CORS 来源：

```env
CORS_ALLOWED_ORIGINS=https://your-frontend.example.com
CORS_ALLOW_CREDENTIALS=false
```

飞书文件确认 pending 状态和公开入口限流优先使用 Redis。生产环境请保证 `REDIS_URL` 可用；如果 Redis 不可用，服务会回退到本机内存，适合本地开发但不适合多实例生产：

```env
REDIS_URL=redis://localhost:6379/0
REDIS_SOCKET_TIMEOUT=5
REDIS_CONNECT_TIMEOUT=5
REDIS_RETRY_COOLDOWN_SECONDS=10
RATE_LIMIT_PER_MINUTE=60
RATE_LIMIT_PER_HOUR=1000
```

本机直接运行服务时可使用 `redis://localhost:6379/0`；如果服务也运行在 Docker Compose 网络内，请使用 `redis://redis:6379/0`。`docker-compose.yml` 已包含 Redis 服务，生产环境建议监控 `/health` 中的 Redis-backed components，避免长期处于 memory fallback。

会话记忆同样优先使用 Redis。`MEMORY_MAX_MESSAGES` 控制每个 session 最多保存多少条消息；每轮请求会按用途读取不同窗口，默认最近答案复用读取 20 条、追问改写读取 12 条、最终回答 prompt 使用 8 条。飞书同一用户在同一聊天窗口会生成稳定 `session_id`，Redis 正常时服务重启后仍可读取最近上下文。

```env
MEMORY_REUSE_WINDOW_MESSAGES=20
MEMORY_REWRITE_WINDOW_MESSAGES=12
MEMORY_PROMPT_WINDOW_MESSAGES=8
MEMORY_MAX_MESSAGES=20
MEMORY_SESSION_TTL_SECONDS=604800
```

## 文档 MCP 服务

DOCX 和 XLSX 的结构读取、编辑与另存为通过独立 MCP HTTP 服务执行；TXT/PDF 仍使用主应用内置后端。启用 MCP 后端时，主应用需要能访问 MCP 服务地址，MCP 服务也必须能访问同一份上传文件。

主应用配置示例：

```env
DOCUMENT_MCP_ENABLED=true
DOCUMENT_MCP_TIMEOUT_SECONDS=30
DOCX_MCP_BASE_URL=http://localhost:9100
XLSX_MCP_BASE_URL=http://localhost:9101
MCP_API_KEY=your_secure_mcp_api_key_here
```

本地 WSL 开发启动：

```bash
cd /mnt/d/LLM/Unified_API_service
export MCP_API_KEY=your_secure_mcp_api_key_here
./scripts/start_mcp_services.sh
```

`scripts/start_mcp_services.sh` 默认只绑定 `127.0.0.1`，适合本机开发。生产或容器部署时不要裸露无认证的 MCP 端口；配置 `MCP_API_KEY` 后，主应用的 HTTP adapter 会自动携带 `X-MCP-API-Key`，MCP 服务会对写入和读取接口返回 401/403。

Docker Compose 启动：

```bash
export MCP_API_KEY=your_secure_mcp_api_key_here
docker-compose up docx-mcp xlsx-mcp
```

如果主应用传给 MCP 的文件路径和容器内挂载路径不同，配置 `MCP_PATH_MAP` 做前缀映射。常见 Docker Compose 示例：

```env
MCP_ALLOWED_DIR=/data/uploads,/data/personal_uploads
MCP_PATH_MAP=./data/uploads=/data/uploads,./data/personal_uploads=/data/personal_uploads
```

WSL/Windows 路径示例：

```env
MCP_PATH_MAP=/mnt/d/LLM/Unified_API_service/data/uploads=/data/uploads,/mnt/d/LLM/Unified_API_service/data/personal_uploads=/data/personal_uploads
```

健康检查地址：

```text
http://localhost:9100/health
http://localhost:9101/health
```

### 文档编辑权限与流程

可编辑文件必须先保存到当前用户的个人知识库，并且状态为 `ready`。系统不会允许直接编辑任意本地路径、企业知识库文件、其他用户个人知识库文件或仍在 processing 的文件。

查看当前用户个人知识库中可用文件：

```bash
curl -X GET http://localhost:8000/documents/personal \
  -H "Authorization: Bearer your_api_key_here" \
  -H "X-User-Id: ou_xxx"
```

返回内容只包含 `document_id`、原始文件名和保存时间，不暴露内部 `stored_path`。编辑时建议使用 `document_id`：

```bash
curl -X POST http://localhost:8000/ask \
  -H "Authorization: Bearer your_api_key_here" \
  -H "X-User-Id: ou_xxx" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "编辑document_id: 47d6f22b-4ae，删除实验目的的内容",
    "document_id": "47d6f22b-4ae",
    "document_action": "plan"
  }'
```

`plan` 只生成结构化编辑方案，不会修改文件。确认无误后，把返回的 `document_plan` 原样提交，并设置 `document_confirmed=true`：

```bash
curl -X POST http://localhost:8000/ask \
  -H "Authorization: Bearer your_api_key_here" \
  -H "X-User-Id: ou_xxx" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "执行编辑",
    "document_id": "47d6f22b-4ae",
    "document_plan": {
      "intent": "edit",
      "file_type": "docx",
      "backend_required": "docx_mcp",
      "operations": [
        {
          "action": "clear_table_cell",
          "target": {"table_index": 0, "row": 1, "col": 0},
          "description": "清空实验目的单元格"
        }
      ]
    },
    "document_confirmed": true
  }'
```

所有编辑都在副本上执行，原文件不覆盖。DOCX 目前支持普通段落编辑，以及表格单元格文本读取、替换和清空：

```text
POST /docx/read_table_cell
POST /docx/replace_table_cell
POST /docx/clear_table_cell
```

表格坐标使用 `read_structure()` 暴露的 0-based `table_index`、`row`、`col`。当前只处理单元格纯文本，不处理复杂样式、图片、嵌套表格和合并单元格范围语义。

DOCX 编辑规划会做确定性相对表格定位后处理。比如用户说“在 `教师评语及成绩：` 的上一个表格填充内容”时，系统会先在 `read_structure()` 的表格文本中定位包含该锚点的表格，再把目标强制改为它的前一个表格，避免完全依赖 LLM 猜测 `table_index`。如果找不到锚点，或锚点所在表格前面没有表格，系统会返回需要补充信息的澄清问题，不会继续乱填。

### DOCX 编辑失败排查

如果飞书端提示“文档编辑执行失败：未知错误”，优先查看主应用和 DOCX MCP 日志：

```bash
tail -n 100 logs/app.log
tail -n 100 logs/dev-stack/docx-mcp.log
tail -n 100 logs/dev-stack/feishu-ws.log
```

常见原因是 DOCX MCP 写入文件时返回 `Permission denied`。这通常表示目标 `.docx` 正被 Word、WPS、LibreOffice、预览器或其他进程打开并锁定，或者 WSL 对该 Windows 挂载路径没有写权限。处理方式：

1. 关闭正在打开该 `.docx` 的 Word/WPS/LibreOffice/预览窗口。
2. 确认文件位于当前用户个人知识库 `ready` 记录对应路径下。
3. 确认 MCP 服务和主应用访问的是同一份文件路径；容器部署时检查 `MCP_PATH_MAP`。
4. 重新发起确认执行。

系统生成的编辑副本会登记为当前用户个人知识库文件。首次编辑源文档会生成副本；后续编辑该副本时会在副本上原地续编，不再层层生成新副本。因此，续编副本时更容易受到“文件已打开/被锁定”的影响。

## 飞书事件订阅

当前支持两种飞书事件接收方式，生产环境建议只启用其中一种主通道，避免同一事件双投递；代码层仍会通过 Redis 事件去重兜底。

HTTP 回调方式：在飞书开放平台选择“将事件发送至开发者服务器”，请求地址配置为：

```text
https://your-domain.example.com/channels/feishu/events
```

长连接备用方式：在飞书开放平台选择“使用长连接接收事件”，然后单独启动 worker：

```bash
cd /mnt/d/LLM/Unified_API_service
/root/.venvs/unified_api_service/bin/python -m app.channels.feishu_ws_worker
```

长连接 worker 使用 `FEISHU_APP_ID` 和 `FEISHU_APP_SECRET` 建连；HTTP 回调仍使用 `FEISHU_VERIFICATION_TOKEN` 和 `FEISHU_ENCRYPT_KEY` 进行回调校验与解密。

飞书文本消息中包含 Docx、Sheets 或 Bitable 链接时，Agent 会默认用应用身份读取在线资源内容并参与本轮回答：

```env
FEISHU_LINK_READ_ENABLED=true
FEISHU_LINK_MAX_COUNT=3
FEISHU_RESOURCE_MAX_CHARS=12000
FEISHU_SHEET_SAMPLE_ROWS=30
FEISHU_BITABLE_SAMPLE_RECORDS=50
```

该能力只使用 `tenant_access_token` 读取资源，不使用用户登录态、不爬网页、不自动入库知识库。生产环境需要在飞书开放平台为应用开通 Docx、Sheets、Bitable 对应只读权限，并确保目标文档已授权给机器人/应用；否则会提示应用没有读取权限。

### 飞书通知与外部服务重试

飞书普通消息、回复消息、交互卡片和文件下载会对临时网络错误（如 timeout、连接中断）做短重试；飞书 OpenAPI 返回业务错误码时不会盲目重试，鉴权错误会清理 token 缓存。发送失败不会再静默吞掉，日志会记录 `chat_id`、`message_id`、`pending_id`、`dedupe_key` 和业务场景。

飞书在线资源读取和 DOCX/XLSX MCP HTTP 调用同样会对临时网络错误短重试；HTTP 错误、非 JSON 响应或 MCP `success=false` 会返回结构化失败并记录响应摘要。后台飞书任务和启动健康检查也会通过安全 wrapper 记录异常，避免 fire-and-forget 任务失败后没有日志。

## 项目结构

```
Unified_API_service/
  app/
    main.py              # FastAPI应用入口
    config.py            # 配置模块
    schemas.py           # 数据模型
    orchestrator.py      # 编排器
    observability/       # 可观测性模块
    llm/                 # LLM网关
    retrieval/           # 检索模块
    memory/              # 会话记忆
    tools/               # 工具模块
    channels/            # 渠道适配
  tests/                 # 测试
  requirements.txt       # 依赖
  .env.example           # 配置模板
```
