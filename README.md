# Unified API Service

统一Agentic RAG API服务，整合文档检索、会话记忆、联网搜索和工具调用能力。

## 技术栈

- FastAPI
- Pydantic v2
- AsyncOpenAI
- DeepSeek API
- Chroma
- Redis
- Serper.dev

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

## 开发阶段

- [x] 阶段1: 统一API骨架
- [ ] 阶段2: DeepSeek LLM Gateway
- [ ] 阶段3: 文档上传与本地RAG
- [ ] 阶段4: 会话记忆与追问改写
- [ ] 阶段5: 联网工具与Agentic RAG
- [ ] 阶段6: 飞书Channel Adapter
- [ ] 阶段7: 生产化增强
