# Unified API Service 分阶段具体计划书

## Summary

按“先骨架、再能力、后生产化”的顺序推进 `D:\LLM\Unified_API_service`。每个阶段都必须形成可运行、可测试、可验收的小闭环，避免一次性大合并导致系统不可控。

默认技术选择：`FastAPI + Pydantic v2 + AsyncOpenAI + DeepSeek API + Chroma + Redis + Serper.dev`。

## 阶段 1：统一 API 骨架

| 项目 | 计划 |
| --- | --- |
| 目标 | 建立统一服务的最小可运行骨架，先不接真实 RAG/工具，保证统一请求响应模型成立。 |
| 核心模块 | `API Runtime`、`Config`、`Schemas`、`Orchestrator`、`Mock LLM`、`Mock Retriever`、`Observability` |
| 主要任务 | 创建 FastAPI 应用、统一配置、统一 schema、全局异常处理、request_id、基础日志、mock 编排流程。 |
| API | `GET /health`、`POST /ask` |
| `/ask` 行为 | 接收统一 `AskRequest`，返回 `request_id`、`session_id`、`route`、`answer`、`sources`、`tool_trace`、`timing`。 |
| 验收标准 | 服务可启动；`/health` 返回 ok；`/ask` 无需外部 API Key 也能返回完整结构。 |
| 测试 | schema 校验、健康检查、mock ask、异常不暴露内部栈。 |

建议目录：

```text
Unified_API_service/
  app/
    main.py
    config.py
    schemas.py
    orchestrator.py
    observability/
    llm/
    retrieval/
    memory/
    tools/
    channels/
  tests/
  requirements.txt
  .env.example
  README.md
```

## 阶段 2：DeepSeek LLM Gateway

| 项目 | 计划 |
| --- | --- |
| 目标 | 接入真实 DeepSeek API，同时保留 mock LLM，方便测试和无 Key 开发。 |
| 核心模块 | `LLM Gateway`、`Prompt Templates`、`Config`、`Error Handling` |
| 主要任务 | 使用 `AsyncOpenAI` 调 DeepSeek；封装 `generate()`；支持 timeout、temperature、max_tokens、model 配置；无 Key 时返回明确错误。 |
| API 变化 | `/ask` 可通过配置选择 `mock` 或 `deepseek`。 |
| 关键配置 | `DEEPSEEK_API_KEY`、`DEEPSEEK_BASE_URL`、`DEEPSEEK_MODEL`、`LLM_TIMEOUT_SECONDS`、`LLM_PROVIDER` |
| 验收标准 | 配置 DeepSeek Key 后 `/ask` 能真实生成回答；未配置 Key 时测试仍可使用 mock。 |
| 测试 | mock LLM 单测、DeepSeek 配置缺失测试、LLM 超时测试、LLM Gateway 返回空内容兜底测试。 |

实现原则：

```text
Orchestrator 不直接调用 AsyncOpenAI
-> 只依赖 LLMGateway 接口
-> LLMGateway 内部决定 mock/deepseek
```

## 阶段 3：文档上传与本地 RAG

| 项目 | 计划 |
| --- | --- |
| 目标 | 接入本地知识库能力，实现上传文档后可检索问答。 |
| 核心模块 | `Retrieval Ingestion`、`Vector Store Adapter`、`Retriever`、`Source Formatter` |
| 主要任务 | 迁移 PDF/TXT 加载、文本切块、embedding、Chroma 入库、top_k 检索、来源去重。 |
| API | `POST /documents/upload`、`POST /ask` 的 `rag` 路由 |
| 上传限制 | 初版只支持 `.pdf`、`.txt`；限制文件大小；保存文件名加 UUID，避免覆盖。 |
| 检索行为 | 用户问题触发 `rag` 时，查询 Chroma，返回知识库来源。 |
| 验收标准 | 上传文档成功返回 `document_id` 和 `chunks`；针对文档内容提问能返回答案和来源。 |
| 测试 | 上传文件类型校验、切块数量、top_k 生效、无文档时友好提示、来源去重。 |

复用来源：

```text
Basic_RAG_QA_API/code/ingest.py
Contextualized_conversational_retrieval/code/rag_chain.py
High-performance_async_service/app/real_chroma.py
```

## 阶段 4：会话记忆与追问改写

| 项目 | 计划 |
| --- | --- |
| 目标 | 支持连续对话，让系统能理解“它/这个/那价格呢”等追问。 |
| 核心模块 | `Memory Store`、`Question Rewriter`、`Query Understanding`、`Session API` |
| 主要任务 | 接入 Redis session；实现历史读取/写入；实现 standalone question；支持 session 查看和删除。 |
| API | `GET /sessions/{session_id}`、`DELETE /sessions/{session_id}`、`POST /ask` 返回 `standalone_question` |
| 会话策略 | 初版使用窗口式历史；保留摘要压缩接口但不强制启用。 |
| Redis 策略 | 生产默认 Redis；测试使用 in-memory fake store。 |
| 验收标准 | 第一轮问主题，第二轮问“它有什么优点”，系统能改写并正确检索。 |
| 测试 | session 自动创建、指定 session 续聊、历史写入、追问改写、删除 session、Redis 不可用错误。 |

数据流：

```text
AskRequest
-> Memory 读取历史
-> Question Rewriter 生成 standalone_question
-> Retriever 用 standalone_question 检索
-> LLM 用 原问题 + 历史 + 文档 回答
-> Memory 写入本轮问答
```

## 阶段 5：联网工具与 Agentic RAG

| 项目 | 计划 |
| --- | --- |
| 目标 | 接入联网搜索、计算器和知识库工具，让 Orchestrator 能组合内部知识和外部信息。 |
| 核心模块 | `Tool Runtime`、`Tool Registry`、`Web Search`、`Calculator`、`Knowledge Search Tool`、`Route Decision` |
| 主要任务 | 迁移 Serper 搜索、安全计算器；定义统一工具返回结构；记录工具 trace；实现 direct/rag/web/tool/agentic_rag 路由。 |
| API 变化 | `/ask` 的 `need_web` 支持 `auto`、`always`、`never`；响应返回 `tool_trace`。 |
| 工具列表 | `web_search`、`calculator`、`knowledge_search` |
| 路由规则 | 企业文档问题走 `rag`；天气/新闻/价格走 `web`；计算走 `tool`；内外信息结合走 `agentic_rag`。 |
| 验收标准 | “今天北京天气适合穿羽绒服吗”会搜索天气；“100 华氏度是多少摄氏度”会调用计算器；“结合知识库和最新公告”会同时使用 RAG 和搜索。 |
| 测试 | 搜索 401/429/timeout 降级、计算器非法表达式、工具 trace 完整、`need_web=never` 不联网。 |

降级策略：

```text
web_search 失败
-> 如果有本地 RAG 结果，基于本地结果回答并说明无法获取最新信息
-> 如果无 RAG 结果，返回友好错误 + request_id
```

## 阶段 6：飞书 Channel Adapter

| 项目 | 计划 |
| --- | --- |
| 目标 | 让飞书机器人作为渠道接入统一 API，而不是单独实现一套 RAG/Agent。 |
| 核心模块 | `Feishu Adapter`、`Channel Adapter`、`Auth`、`Response Renderer` |
| 主要任务 | 处理飞书 challenge；校验 token/signature；解析文本消息；映射 user/session；调用 Orchestrator；回复飞书消息。 |
| API | `POST /channels/feishu/events` |
| session 映射 | 使用 `feishu:{open_id}:{chat_id}` 生成稳定 session_id。 |
| 回复格式 | 初版纯文本；包含答案和简化来源；失败时返回友好提示。 |
| 验收标准 | 飞书单聊可问答；群聊 @ 机器人可触发；同一用户同一会话能保留上下文。 |
| 测试 | challenge 校验、文本解析、非文本消息忽略、session 映射、飞书 API 失败降级。 |

不进入初版：

```text
复杂飞书卡片
主动群发
飞书文件自动入库
审批或多步骤业务动作
```

## 阶段 7：生产化增强

| 项目 | 计划 |
| --- | --- |
| 目标 | 补齐稳定性、安全、成本、部署和压测能力，使项目具备生产演进基础。 |
| 核心模块 | `Auth`、`Rate Limit`、`Cache`、`Observability`、`Benchmark`、`Deployment` |
| 主要任务 | API Key/JWT、限流、搜索缓存、答案缓存、结构化日志、token/cost 统计、Docker、Locust 压测。 |
| 安全要求 | 不记录密钥；不返回内部异常栈；上传文件做类型/大小检查；工具调用白名单。 |
| 可观测性 | 每次请求记录 `request_id`、`user_id`、`channel`、`route`、工具、耗时、错误类型、token 用量。 |
| 验收标准 | 高并发下接口不阻塞；外部依赖失败可降级；日志能按 request_id 追踪完整请求。 |
| 测试 | 限流测试、缓存测试、异常脱敏测试、并发测试、Docker 启动测试。 |

压测目标：

```text
先用 mock LLM/mock retrieval 建立基准
再用真实 DeepSeek/Chroma 做 smoke 测试
不要用真实 LLM 做大规模压测，避免成本失控
```

## Cross-Stage Rules

- 所有复杂逻辑必须有中文注释，尤其是 Orchestrator、追问改写、混合检索、工具降级、安全计算器。
- 所有阶段都必须保留 mock 能力，避免测试依赖真实 DeepSeek、Redis、Serper。
- 所有外部依赖都必须有 timeout 和友好错误。
- 所有响应都必须带 `request_id`，方便排查。
- 不允许把四个教学项目原样合并进统一项目，只迁移成熟实现和设计思路。
- 默认虚拟环境按项目记忆执行：Windows 使用 `D:\C\python\new_python\.venv`，WSL 使用 `source /mnt/d/C/python/new_python/.venv/Scripts/activate`。
