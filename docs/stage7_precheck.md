# 阶段七前置验收

阶段七生产化增强前，先用真实 API 验证阶段一到六核心链路可用。

## 前置条件

- 在 `.env.toml` 填写真实 `deepseek_api_key` 和 `serper_api_key`。
- 飞书配置和 `public_base_url` 暂缺时，飞书验收会自动标记为 `SKIPPED`。
- 启动服务前，将 `.env.toml` 中的真实配置同步到 `.env`；当前应用默认读取 `.env`。

```powershell
D:\C\python\new_python\.venv\Scripts\python.exe scripts\sync_env_from_toml.py
```

## 启动服务

```powershell
D:\C\python\new_python\.venv\Scripts\python.exe -m app.main
```

默认服务地址为 `http://127.0.0.1:8000`，可通过 `.env.toml` 的 `[app].base_url` 调整。

## 运行验收

```powershell
D:\C\python\new_python\.venv\Scripts\python.exe scripts\smoke_real_api.py
```

脚本会生成 Markdown 报告到 `reports/stage7_precheck_*.md`。

## 验收范围

- 基础服务：`/health`、`/`
- 直接问答：纯问候路由
- 工具调用：计算器真实路径
- DeepSeek：真实 LLM 回答，不能返回 mock 文本
- 文档上传与 RAG：上传临时 TXT 并检索唯一短语
- 会话记忆：同一 session 连续两轮对话
- Serper：`need_web=always` 真实联网搜索
- Agentic RAG：时效问题触发本地检索与联网组合
- 飞书：配置暂缺时跳过；配置齐备后再做真实回调验收

## 通过标准

- 必跑项 `health/root/direct/tool/upload_and_rag/session_memory` 必须 `PASS`。
- DeepSeek、Serper 配置存在时必须 `PASS`，配置缺失时允许 `SKIPPED`。
- 飞书相关配置暂缺时允许 `SKIPPED`。
