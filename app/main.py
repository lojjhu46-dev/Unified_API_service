"""FastAPI应用入口"""

import uuid
from datetime import datetime
from fastapi import BackgroundTasks, FastAPI, Request, UploadFile, File, HTTPException
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


async def process_feishu_message(event_data: dict) -> None:
    """后台处理飞书消息并回复"""
    open_id = event_data.get("open_id")
    chat_id = event_data.get("chat_id")
    message_id = event_data.get("message_id")
    text = event_data.get("text")
    dedupe_key = event_data.get("dedupe_key")

    logger.info(
        "Feishu message processing started",
        extra={
            "chat_type": event_data.get("chat_type"),
            "message_id": message_id,
            "dedupe_key": dedupe_key,
        },
    )

    try:
        session_id = feishu_adapter.generate_session_id(open_id, chat_id)
        ask_request = AskRequest(
            channel="feishu",
            user_id=open_id,
            session_id=session_id,
            question=text,
        )
        response = await orchestrator.process(ask_request)
        answer = response.answer
    except Exception as e:
        logger.error(f"飞书消息处理失败: {e}", exc_info=True)
        answer = "抱歉，处理您的问题时出现错误，请稍后重试。"

    try:
        reply_ok = await feishu_adapter.reply_message(message_id, answer)
        logger.info(
            "Feishu reply finished",
            extra={
                "message_id": message_id,
                "dedupe_key": dedupe_key,
                "reply_ok": reply_ok,
            },
        )
    except Exception as e:
        logger.error(f"飞书消息回复失败: {e}", exc_info=True)


@app.post("/channels/feishu/events")
async def feishu_events(request: Request, background_tasks: BackgroundTasks):
    """飞书事件回调入口

    处理流程：
    1. token 校验
    2. challenge 验证（配置事件订阅时）
    3. 解析消息事件
    4. 幂等登记
    5. 后台调用 Orchestrator 并回复飞书消息
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON"})

    encrypted_payload = "encrypt" in body
    logger.info(
        "Feishu callback received",
        extra={
            "encrypted": encrypted_payload,
            "body_keys": sorted(body.keys()),
        },
    )

    body = feishu_adapter.decrypt_event_body(body)
    if body is None:
        logger.warning("Feishu callback rejected: decrypt failed")
        return JSONResponse(status_code=400, content={"error": "Invalid encrypted payload"})

    # 1. token 校验。challenge 也必须通过 token 校验后再返回。
    if not feishu_adapter.verify_token(body):
        logger.warning(
            "Feishu callback rejected: invalid token",
            extra={
                "encrypted": encrypted_payload,
                "event_type": body.get("header", {}).get("event_type"),
                "type": body.get("type"),
            },
        )
        return JSONResponse(status_code=403, content={"error": "Invalid token"})

    # 2. challenge 验证
    challenge_response = feishu_adapter.verify_challenge(body)
    if challenge_response:
        logger.info("Feishu challenge responded", extra={"encrypted": encrypted_payload})
        return challenge_response

    # 3. 解析消息事件
    event_data = feishu_adapter.parse_event(body)
    if not event_data:
        logger.info(
            "Feishu callback acknowledged without processing",
            extra={
                "encrypted": encrypted_payload,
                "event_type": body.get("header", {}).get("event_type"),
                "type": body.get("type"),
            },
        )
        return {"code": 0}

    # 4. 幂等登记，重复事件直接确认，避免飞书重试造成重复回复。
    if not feishu_adapter.mark_event_seen(event_data.get("dedupe_key")):
        logger.info(
            "Feishu duplicate event acknowledged",
            extra={
                "dedupe_key": event_data.get("dedupe_key"),
                "message_id": event_data.get("message_id"),
            },
        )
        return {"code": 0}

    # 5. 后台处理，确保回调入口快速返回。
    logger.info(
        "Feishu message event accepted",
        extra={
            "event_id": event_data.get("event_id"),
            "message_id": event_data.get("message_id"),
            "chat_type": event_data.get("chat_type"),
            "dedupe_key": event_data.get("dedupe_key"),
        },
    )
    background_tasks.add_task(process_feishu_message, event_data)
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
