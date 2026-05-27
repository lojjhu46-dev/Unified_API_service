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

```bash
# 安装依赖
pip install -r requirements.txt

# 复制配置文件
cp .env.example .env

# 启动服务
python -m app.main
```

## API文档

启动服务后访问 http://localhost:8000/docs

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
