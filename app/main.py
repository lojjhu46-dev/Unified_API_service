"""FastAPI应用入口"""

import uuid
from datetime import datetime
from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.config import settings
from app.schemas import (
    AskRequest,
    AgentResponse,
    HealthResponse,
    UploadResponse,
    SessionCreateResponse,
    SessionHistoryResponse,
    ErrorResponse,
)
from app.llm.gateway import LLMGatewayError
from app.orchestrator import orchestrator
from app.channels.feishu import feishu_adapter
from app.retrieval.ingest import (
    ingest_file,
    sanitize_filename,
    save_uploaded_file,
    validate_file_extension,
)
from app.observability.logging import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="统一Agentic RAG API服务",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(LLMGatewayError)
async def llm_gateway_exception_handler(request: Request, exc: LLMGatewayError) -> JSONResponse:
    request_id = str(uuid.uuid4())[:12]
    logger.warning(
        f"LLM网关错误: {exc}",
        extra={"request_id": request_id},
    )
    return JSONResponse(
        status_code=503,
        content=ErrorResponse(
            detail=str(exc),
            request_id=request_id,
        ).model_dump(),
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    request_id = str(uuid.uuid4())[:12]
    logger.error(
        f"未处理异常: {type(exc).__name__}: {exc}",
        extra={"request_id": request_id},
        exc_info=True,
    )
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            detail="内部服务错误，请稍后重试",
            request_id=request_id,
        ).model_dump(),
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        version=settings.app_version,
        timestamp=datetime.now(),
    )


@app.post("/ask", response_model=AgentResponse)
async def ask(request: AskRequest):
    return await orchestrator.process(request)


@app.post("/documents/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """上传文档到知识库"""
    if not validate_file_extension(file.filename):
        raise HTTPException(status_code=400, detail="仅支持 PDF 和 TXT 文件")

    content = await file.read()
    if len(content) > 10 * 1024 * 1024:  # 10MB限制
        raise HTTPException(status_code=400, detail="文件大小不能超过10MB")

    try:
        safe_filename = sanitize_filename(file.filename)
        file_path = save_uploaded_file(content, file.filename)
        result = ingest_file(file_path, original_filename=safe_filename)
        orchestrator.retriever.refresh()
        return UploadResponse(
            document_id=result["document_id"],
            filename=result["filename"],
            chunks=result["chunks"],
            status="success",
            message=f"文档上传成功，共{result['chunks']}个切块",
        )
    except Exception as e:
        request_id = str(uuid.uuid4())[:12]
        logger.error(
            f"文档上传失败: {e}",
            extra={"request_id": request_id},
            exc_info=True,
        )
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                detail="文档处理失败，请稍后重试",
                request_id=request_id,
            ).model_dump(),
        )


@app.post("/sessions", response_model=SessionCreateResponse)
async def create_session():
    """创建新会话"""
    session_id = await orchestrator.memory.create_session()
    return SessionCreateResponse(session_id=session_id)


@app.get("/sessions/{session_id}/history", response_model=SessionHistoryResponse)
async def get_session_history(session_id: str):
    """获取会话历史"""
    history = await orchestrator.memory.get_history(session_id)
    return SessionHistoryResponse(
        session_id=session_id,
        messages=history,
    )


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """删除会话"""
    await orchestrator.memory.clear_session(session_id)
    return {"status": "deleted", "session_id": session_id}


@app.post("/channels/feishu/events")
async def feishu_events(request: Request):
    """飞书事件回调入口

    处理流程：
    1. challenge 验证（配置事件订阅时）
    2. token 校验
    3. 解析消息事件
    4. 调用 Orchestrator 处理
    5. 回复飞书消息
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON"})

    # 1. challenge 验证
    challenge_response = feishu_adapter.verify_challenge(body)
    if challenge_response:
        return challenge_response

    # 2. token 校验
    if not feishu_adapter.verify_token(body):
        return JSONResponse(status_code=403, content={"error": "Invalid token"})

    # 3. 解析消息事件
    event_data = feishu_adapter.parse_event(body)
    if not event_data:
        return {"code": 0}

    open_id = event_data.get("open_id")
    chat_id = event_data.get("chat_id")
    message_id = event_data.get("message_id")
    text = event_data.get("text")

    if not text:
        return {"code": 0}

    # 4. 生成 session_id 并调用 Orchestrator
    session_id = feishu_adapter.generate_session_id(open_id, chat_id)

    ask_request = AskRequest(
        channel="feishu",
        user_id=open_id,
        session_id=session_id,
        question=text,
    )

    try:
        response = await orchestrator.process(ask_request)
        answer = response.answer
    except Exception as e:
        logger.error(f"飞书消息处理失败: {e}", exc_info=True)
        answer = "抱歉，处理您的问题时出现错误，请稍后重试。"

    # 5. 回复飞书消息
    await feishu_adapter.reply_message(message_id, answer)

    return {"code": 0}


@app.get("/")
async def root():
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.app_host, port=settings.app_port)
