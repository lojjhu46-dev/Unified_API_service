# Unified API Service 逻辑模块细分方案

## Summary

将 `D:\LLM` 下四个基础项目沉淀为一个生产化统一 Agentic RAG API 服务，核心不是合并代码，而是抽象出稳定模块边界：统一入口、统一请求响应、统一编排、统一检索、统一会话、统一工具、统一模型网关、统一观测和降级。

复用来源默认如下：

| 来源项目 | 主要复用能力 |
| --- | --- |
| `Basic_RAG_QA_API` | 文档上传、PDF/TXT 解析、切块、Chroma 入库、基础 RAG |
| `Contextualized_conversational_retrieval` | Redis 会话、追问改写、BM25 + Vector 混合检索、全局汇总检索 |
| `Network-connected_AI_Agent` | Serper 联网搜索、安全计算器、工具调用轨迹、Agent 工具协议 |
| `High-performance_async_service` | FastAPI 异步骨架、AsyncOpenAI、全局异常、计时、并发检索、测试结构 |

## Logic Modules

| 模块 | 职责 | 初始实现策略 |
| --- | --- | --- |
| `API Runtime` | FastAPI 应用启动、路由注册、CORS、异常处理、健康检查 | 参考异步服务项目，提供 `/health`、`/ask`、文档、会话、飞书入口 |
| `Config & Secrets` | 环境变量、路径、模型、超时、API Key、功能开关 | 默认 DeepSeek API；禁止硬编码密钥；`.env.example` 只放占位名 |
| `Schemas` | 统一请求、响应、来源、工具轨迹、耗时、错误结构 | 使用 Pydantic v2；所有外部 API 都只依赖这里的模型 |
| `Auth & Access Control` | API Key/JWT、飞书签名、用户身份、知识库范围控制 | 初版用 API Key + `knowledge_scope`；后续扩展部门/用户权限 |
| `Channel Adapter` | 把不同入口转换成统一 `AskRequest` | 初版支持 `api` 和 `feishu`；飞书不直接调用 RAG/LLM/Tool |
| `Orchestrator` | 核心调度大脑，决定 direct/rag/web/tool/agentic_rag | 规则优先，必要时 LLM 辅助；统一处理超时、失败、降级 |
| `Query Understanding` | 会话上下文读取、追问改写、问题分类、检索 query 扩展 | 复用会话检索项目的 standalone question 逻辑 |
| `Memory` | session 创建、历史读写、窗口记忆、摘要压缩、清理 | 生产默认 Redis；本地测试可提供 in-memory mock |
| `Retrieval Ingestion` | 上传文件保存、类型校验、PDF/TXT 解析、切块、embedding 入库 | 复用基础 RAG 的 ingest 流程，增加文件大小/类型限制 |
| `Retriever` | Chroma 检索、BM25 检索、混合检索、全局汇总、来源去重 | 复用会话检索项目的 hybrid/global 思路 |
| `Vector Store Adapter` | 封装 Chroma 本地/HTTP 客户端，屏蔽 SDK 差异 | 同步 SDK 用 `asyncio.to_thread` 包装，避免阻塞事件循环 |
| `Tool Runtime` | 统一工具注册、参数校验、执行、trace、超时、错误包装 | 初版工具：`web_search`、`calculator`、`knowledge_search` |
| `LLM Gateway` | DeepSeek/OpenAI-compatible 异步调用、prompt、重试、mock | 默认 `AsyncOpenAI(base_url=https://api.deepseek.com)` |
| `Response Composer` | 汇总检索结果、工具结果、LLM 输出，生成最终响应 | 输出 answer、sources、tool_trace、route、timing、request_id |
| `Observability` | 日志、request_id、trace_id、耗时、错误类型、token/cost | 初版结构化日志；后续接 OpenTelemetry/Prometheus |
| `Rate Limit & Cost Control` | 用户限流、搜索缓存、答案缓存、token 预算、配额 | 初版预留接口和配置，生产阶段开启 |
| `Background Tasks` | 大文档异步入库、索引刷新、缓存清理、批处理任务 | 初版可先同步入库；接口设计预留 job_id |
| `Benchmark & Tests` | 单元测试、API 测试、mock LLM、mock retrieval、压测脚本 | 复用异步服务项目的测试风格和 Locust 思路 |

## Public APIs & Core Types

统一外部 API：

| 方法 | 路径 | 作用 |
| --- | --- | --- |
| `GET` | `/health` | 健康检查 |
| `POST` | `/ask` | 统一问答入口 |
| `POST` | `/documents/upload` | 上传文档并入库 |
| `GET` | `/sessions/{session_id}` | 查看会话状态和历史摘要 |
| `DELETE` | `/sessions/{session_id}` | 清理会话 |
| `POST` | `/channels/feishu/events` | 飞书事件回调入口 |

核心请求类型：

| 类型 | 关键字段 |
| --- | --- |
| `AskRequest` | `channel`、`user_id`、`session_id`、`question`、`knowledge_scope`、`need_web`、`top_k`、`return_trace` |
| `AgentResponse` | `request_id`、`session_id`、`route`、`answer`、`sources`、`tool_trace`、`timing`、`standalone_question` |
| `SourceItem` | `title`、`url`、`source_type`、`snippet`、`score`、`metadata` |
| `ToolTrace` | `tool_name`、`tool_input`、`status`、`output_preview`、`latency_ms`、`error_message` |
| `TimingInfo` | `rewrite_ms`、`retrieval_ms`、`tool_ms`、`llm_ms`、`total_ms` |
| `UploadResponse` | `document_id`、`filename`、`chunks`、`status`、`message` |

核心数据流：

```text
HTTP / Feishu
-> Channel Adapter
-> Auth & request_id
-> AskRequest
-> Orchestrator
-> Memory 读取历史
-> Query Understanding 追问改写
-> Route Decision
-> Retrieval / Tools / LLM Gateway 并行或串行执行
-> Response Composer
-> AgentResponse
-> Observability 记录日志和耗时
```

## Implementation Order

| 阶段 | 目标 | 完成标准 |
| --- | --- | --- |
| Stage 1 | 建立统一 API 骨架、配置、schema、mock LLM/mock Retriever | `/health` 和 `/ask` 返回完整统一结构 |
| Stage 2 | 接入 DeepSeek 异步 LLM Gateway 和全局异常/计时 | 无 API Key 时友好提示，异常不暴露栈和密钥 |
| Stage 3 | 接入文档上传、切块、Chroma 入库、基础检索 | 上传 PDF/TXT 后可基于知识库回答并返回来源 |
| Stage 4 | 接入 Redis 会话、追问改写、混合检索 | 第二轮“它/这个/那价格呢”能改写为 standalone question |
| Stage 5 | 接入 `web_search`、`calculator`、`knowledge_search` 工具运行时 | 工具调用有 trace，搜索失败可降级 |
| Stage 6 | 接入飞书 Channel Adapter | 支持 challenge、单聊文本、群聊 @、基于 open_id/chat_id 的 session |
| Stage 7 | 生产化增强 | API Key/JWT、限流、日志、缓存、压测、Docker 部署 |

## Test Plan

必须覆盖：

| 测试类型 | 场景 |
| --- | --- |
| Schema 测试 | `AskRequest` 字段校验、`need_web` 枚举、`top_k` 范围、空问题拒绝 |
| API 测试 | `/health`、`/ask`、`/documents/upload`、session 查询/删除 |
| Orchestrator 测试 | direct/rag/web/tool/agentic_rag 路由判断 |
| Memory 测试 | 新建 session、连续追问、历史窗口、Redis 不可用时的错误 |
| Retrieval 测试 | PDF/TXT 入库、top_k、生效、来源去重、全局汇总问题 |
| Tool 测试 | calculator 安全表达式、非法表达式、Serper 401/429/timeout 降级 |
| LLM Gateway 测试 | mock LLM、DeepSeek 配置缺失、超时、重试 |
| Observability 测试 | 响应包含 request_id、route、timing、tool_trace |
| Feishu 测试 | challenge 回调、文本消息解析、session 映射、失败友好回复 |
| 并发测试 | 多请求 `/ask` 不阻塞，Chroma 同步 SDK 不阻塞事件循环 |

## Assumptions

- 默认开发语言为 Python，服务框架为 FastAPI。
- 默认模型供应商为 DeepSeek，使用 OpenAI-compatible 异步客户端。
- 复杂业务逻辑代码需要中文注释，尤其是编排、检索、工具、安全计算器、降级逻辑。
- 统一项目不直接复制四个教学项目的整体结构，只迁移成熟思路和可复用模块。
- 生产默认使用 Redis 存储会话，本地测试允许 mock/in-memory 替代。
- 初版知识库使用 Chroma；后续可以替换或扩展为 Milvus、pgvector、Elasticsearch 混合检索。
- 初版联网搜索使用 Serper.dev；后续可新增 Tavily 或企业内部搜索。
- 飞书只作为 Channel Adapter，不允许飞书模块直接依赖 RAG、LLM 或工具实现。
