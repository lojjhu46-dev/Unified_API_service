"""FastAPI应用入口"""

import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from fastapi import BackgroundTasks, Depends, FastAPI, Request, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.config import parse_csv_setting, settings
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
from app.channels.pending_store import get_pending_file_store, pending_feishu_files
from app.retrieval.ingest import (
    ingest_file,
    sanitize_filename,
    save_uploaded_file,
    validate_file_extension,
)
from app.observability.logging import get_logger
from app.redis_client import log_redis_startup_health
from app.security.auth import AuthContext, get_api_auth_context
from app.security import rate_limit as rate_limit_module
from app.security.rate_limit import enforce_rate_limit, identity_from_auth, identity_from_feishu_event

logger = get_logger(__name__)

pending_file_store = get_pending_file_store()
SUPPORTED_FILE_TEXT = "仅支持 PDF、DOCX、TXT 和 XLSX 文件。"

@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(log_redis_startup_health())
    yield


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan,
    description="统一Agentic RAG API服务",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=parse_csv_setting(settings.cors_allowed_origins),
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=parse_csv_setting(settings.cors_allowed_methods),
    allow_headers=parse_csv_setting(settings.cors_allowed_headers),
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
    feishu_dedupe_health = await feishu_adapter.get_event_dedupe_health()
    memory_health = await orchestrator.memory.health()
    pending_health = await pending_file_store.health()
    rate_limit_health = await rate_limit_module.rate_limiter.health()
    token_cache_health = await feishu_adapter.get_tenant_token_cache_health()
    return HealthResponse(
        status="ok",
        version=settings.app_version,
        timestamp=datetime.now(),
        components={
            "memory_store": memory_health,
            "feishu_pending_store": pending_health,
            "feishu_event_dedupe": feishu_dedupe_health,
            "rate_limiter": rate_limit_health,
            "feishu_token_cache": token_cache_health,
        },
    )


@app.post("/ask", response_model=AgentResponse)
async def ask(
    http_request: Request,
    request: AskRequest,
    auth_context: AuthContext | None = Depends(get_api_auth_context),
):
    await enforce_rate_limit("ask", identity_from_auth(http_request, auth_context))
    trusted_request = request
    if auth_context is not None:
        trusted_request = request.model_copy(
            update={
                "user_id": auth_context.user_id,
                "channel": auth_context.channel,
            }
        )
    return await orchestrator.process(trusted_request)


@app.post("/documents/upload", response_model=UploadResponse)
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    knowledge_base_type: str = Form("enterprise"),
    owner_open_id: str | None = Form(None),
    auth_context: AuthContext | None = Depends(get_api_auth_context),
):
    """上传文档到知识库"""
    await enforce_rate_limit("documents_upload", identity_from_auth(request, auth_context))
    if not validate_file_extension(file.filename):
        raise HTTPException(status_code=400, detail=SUPPORTED_FILE_TEXT)

    knowledge_base_type = (knowledge_base_type or "enterprise").strip().lower()
    owner_open_id = owner_open_id.strip() if owner_open_id else None
    if knowledge_base_type not in {"enterprise", "personal"}:
        raise HTTPException(status_code=400, detail="knowledge_base_type 仅支持 enterprise 或 personal")
    if auth_context is not None and knowledge_base_type == "personal":
        owner_open_id = auth_context.user_id
    if knowledge_base_type == "personal" and not owner_open_id:
        raise HTTPException(status_code=400, detail="上传到个人知识库时必须提供 owner_open_id")

    content = await file.read()
    if len(content) > 10 * 1024 * 1024:  # 10MB限制
        raise HTTPException(status_code=400, detail="文件大小不能超过10MB")

    try:
        safe_filename = sanitize_filename(file.filename)
        file_path = save_uploaded_file(content, file.filename)
        result = ingest_file(
            file_path,
            original_filename=safe_filename,
            knowledge_base_type=knowledge_base_type,
            owner_open_id=owner_open_id if knowledge_base_type == "personal" else None,
            channel="api",
        )
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

    heartbeat_task = None
    if settings.feishu_heartbeat_enabled:
        heartbeat_task = asyncio.create_task(
            send_feishu_processing_heartbeat(chat_id, dedupe_key)
        )

    try:
        try:
            session_id = feishu_adapter.generate_session_id(open_id, chat_id)
            knowledge_scope = decide_feishu_knowledge_scope(text)
            ask_request = AskRequest(
                channel="feishu",
                user_id=open_id,
                session_id=session_id,
                question=text,
                knowledge_scope=knowledge_scope,
            )
            response = await orchestrator.process(ask_request)
            answer = response.answer
        except Exception as e:
            logger.error(f"飞书消息处理失败: {e}", exc_info=True)
            answer = "抱歉，处理您的问题时出现错误，请稍后重试。"
    finally:
        if heartbeat_task:
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass

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


def decide_feishu_knowledge_scope(text: str) -> list[str]:
    """飞书默认同时查企业知识库和当前用户个人知识库。"""
    question = text or ""
    enterprise_only_words = ["只查企业", "仅查企业", "只看企业", "企业知识库"]
    personal_only_words = ["只查个人", "仅查个人", "只看个人", "个人知识库", "我的知识库", "我上传", "我上传的文件"]
    combine_words = ["结合", "同时", "全部", "企业和个人", "个人和企业", "企业个人", "混合检索"]
    wants_combine = any(word in question for word in combine_words)
    if wants_combine:
        return ["enterprise", "personal"]

    wants_enterprise_only = any(word in question for word in enterprise_only_words)
    wants_personal = any(word in question for word in personal_only_words)
    if wants_personal and not wants_enterprise_only:
        return ["personal"]
    if wants_enterprise_only and not wants_personal:
        return ["enterprise"]
    return ["enterprise", "personal"]


async def process_feishu_file_event(event_data: dict) -> None:
    """处理飞书文件消息：先发确认卡片，不直接入库。"""
    raw_filename = event_data.get("file_name")
    if not validate_file_extension(raw_filename):
        await feishu_adapter.reply_message(event_data.get("message_id"), SUPPORTED_FILE_TEXT)
        return

    filename = sanitize_filename(raw_filename)
    pending_id = str(uuid.uuid4())[:12]
    pending = {
        **event_data,
        "file_name": filename,
        "status": "pending",
        "created_at": time.time(),
        "expires_at": time.time() + settings.feishu_pending_file_ttl_seconds,
    }
    await pending_file_store.create(pending_id, pending, settings.feishu_pending_file_ttl_seconds)
    card = build_personal_file_confirm_card(filename, pending_id)
    await feishu_adapter.send_interactive_card(event_data.get("chat_id"), card)


def build_personal_file_confirm_card(filename: str, pending_id: str) -> dict:
    return {
        "config": {"wide_screen_mode": True},
        "header": {
            "template": "blue",
            "title": {"tag": "plain_text", "content": "保存到知识库"},
        },
        "elements": [
            {"tag": "div", "text": {"tag": "lark_md", "content": f"请选择将 `{filename}` 保存到哪个知识库。"}},
            {
                "tag": "action",
                "actions": [
                    {
                        "tag": "button",
                        "text": {"tag": "plain_text", "content": "保存到个人知识库"},
                        "type": "primary",
                        "value": {"action": "confirm_save_personal_file", "pending_id": pending_id},
                    },
                    {
                        "tag": "button",
                        "text": {"tag": "plain_text", "content": "保存到企业知识库"},
                        "type": "default",
                        "value": {"action": "confirm_save_enterprise_file", "pending_id": pending_id},
                    },
                    {
                        "tag": "button",
                        "text": {"tag": "plain_text", "content": "取消"},
                        "type": "default",
                        "value": {"action": "cancel_save_personal_file", "pending_id": pending_id},
                    },
                ],
            },
        ],
    }


async def process_feishu_card_action(action_data: dict) -> None:
    action = action_data.get("action")
    pending_id = action_data.get("pending_id")
    pending = await pending_file_store.get(pending_id)
    if not pending:
        chat_id = action_data.get("chat_id")
        if chat_id:
            await feishu_adapter.send_message(chat_id, "文件确认已过期，请重新发送文件。")
        return
    if pending.get("expires_at", 0) < time.time():
        await pending_file_store.delete(pending_id)
        chat_id = action_data.get("chat_id") or pending.get("chat_id")
        if chat_id:
            await feishu_adapter.send_message(chat_id, "文件确认已过期，请重新发送文件。")
        return
    if not action_data.get("open_id") or action_data.get("open_id") != pending.get("open_id"):
        await feishu_adapter.send_message(pending.get("chat_id"), "只有上传文件的用户可以确认保存。")
        return

    if action == "cancel_save_personal_file":
        await pending_file_store.delete(pending_id)
        await feishu_adapter.send_message(pending.get("chat_id"), f"已取消保存 `{pending.get('file_name')}`。")
        return

    if action not in {"confirm_save_personal_file", "confirm_save_enterprise_file"}:
        return

    pending = await pending_file_store.consume(pending_id)
    if not pending:
        chat_id = action_data.get("chat_id")
        if chat_id:
            await feishu_adapter.send_message(chat_id, "文件确认已过期，请重新发送文件。")
        return
    if action == "confirm_save_enterprise_file":
        await save_feishu_file_to_enterprise_knowledge(pending)
    else:
        await save_feishu_file_to_personal_knowledge(pending)


async def save_feishu_file_to_personal_knowledge(pending: dict) -> None:
    filename = pending.get("file_name")
    chat_id = pending.get("chat_id")
    try:
        content = await feishu_adapter.download_message_resource(
            pending.get("message_id"),
            pending.get("file_key"),
        )
        if not content:
            await feishu_adapter.send_message(chat_id, "文件下载失败，请稍后重试。")
            return
        if len(content) > 10 * 1024 * 1024:
            await feishu_adapter.send_message(chat_id, "文件大小不能超过10MB。")
            return

        file_path = save_uploaded_file(content, filename, upload_dir=settings.personal_upload_dir)
        result = ingest_file(
            file_path,
            original_filename=filename,
            knowledge_base_type="personal",
            owner_open_id=pending.get("open_id"),
            chat_id=chat_id,
            channel="feishu",
        )
        orchestrator.retriever.refresh()
        await feishu_adapter.send_message(
            chat_id,
            f"已保存到个人知识库：{result['filename']}，共 {result['chunks']} 个切块。",
        )
    except Exception as e:
        logger.error(f"飞书文件保存到个人知识库失败: {e}", exc_info=True)
        await feishu_adapter.send_message(chat_id, "文件保存失败，请稍后重试。")


async def save_feishu_file_to_enterprise_knowledge(pending: dict) -> None:
    filename = pending.get("file_name")
    chat_id = pending.get("chat_id")
    try:
        content = await feishu_adapter.download_message_resource(
            pending.get("message_id"),
            pending.get("file_key"),
        )
        if not content:
            await feishu_adapter.send_message(chat_id, "文件下载失败，请稍后重试。")
            return
        if len(content) > 10 * 1024 * 1024:
            await feishu_adapter.send_message(chat_id, "文件大小不能超过10MB。")
            return

        file_path = save_uploaded_file(content, filename, upload_dir=settings.upload_dir)
        result = ingest_file(
            file_path,
            original_filename=filename,
            knowledge_base_type="enterprise",
            owner_open_id=None,
            chat_id=chat_id,
            channel="feishu",
        )
        orchestrator.retriever.refresh()
        await feishu_adapter.send_message(
            chat_id,
            f"已保存到企业知识库：{result['filename']}，共 {result['chunks']} 个切块。",
        )
    except Exception as e:
        logger.error(f"飞书文件保存到企业知识库失败: {e}", exc_info=True)
        await feishu_adapter.send_message(chat_id, "文件保存失败，请稍后重试。")


async def send_feishu_processing_heartbeat(chat_id: str, dedupe_key: str | None = None) -> None:
    """长耗时飞书问答处理心跳，使用普通新消息提示用户仍在处理。"""
    messages = [
        "正在处理，请稍候...",
        "还在检索和整理资料，请稍候...",
        "问题稍复杂，仍在生成回答...",
    ]
    max_count = max(settings.feishu_heartbeat_max_count, 0)

    try:
        for index in range(max_count):
            delay = (
                settings.feishu_heartbeat_initial_delay_seconds
                if index == 0
                else settings.feishu_heartbeat_interval_seconds
            )
            await asyncio.sleep(delay)
            text = messages[index % len(messages)]
            await send_feishu_status_message(chat_id, text, dedupe_key, f"heartbeat_{index + 1}")

        if max_count > 0:
            await send_feishu_status_message(
                chat_id,
                "处理时间较长，我会继续尝试完成回答。",
                dedupe_key,
                "max_count",
            )
    except asyncio.CancelledError:
        raise
    except Exception as e:
        logger.warning(
            f"飞书处理心跳发送失败: {e}",
            extra={"chat_id": chat_id, "dedupe_key": dedupe_key},
            exc_info=True,
        )


async def send_feishu_status_message(
    chat_id: str,
    text: str,
    dedupe_key: str | None,
    status_type: str,
) -> bool:
    """发送飞书处理状态普通消息，失败只记录日志。"""
    try:
        ok = await feishu_adapter.send_message(chat_id, text)
        logger.info(
            "Feishu status message sent",
            extra={
                "chat_id": chat_id,
                "dedupe_key": dedupe_key,
                "status_type": status_type,
                "send_ok": ok,
            },
        )
        return ok
    except Exception as e:
        logger.warning(
            f"飞书状态消息发送失败: {e}",
            extra={
                "chat_id": chat_id,
                "dedupe_key": dedupe_key,
                "status_type": status_type,
            },
            exc_info=True,
        )
        return False


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

    card_action = parse_feishu_card_action(body)
    if card_action:
        await enforce_rate_limit(
            "feishu_events",
            identity_from_feishu_event(request, None, card_action),
        )
        background_tasks.add_task(process_feishu_card_action, card_action)
        return {
            "code": 0,
            "toast": {
                "type": "info",
                "content": "已收到操作，正在处理。",
            },
        }

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

    await enforce_rate_limit(
        "feishu_events",
        identity_from_feishu_event(request, event_data),
    )

    # 4. 幂等登记，重复事件直接确认，避免飞书重试造成重复回复。
    is_first_seen = await feishu_adapter.mark_event_seen(event_data.get("dedupe_key"))
    dedupe_health = await feishu_adapter.get_event_dedupe_health()
    dedupe_log_fields = {
        "event_id": event_data.get("event_id"),
        "message_id": event_data.get("message_id"),
        "dedupe_key": event_data.get("dedupe_key"),
        "dedupe_key_source": event_data.get("dedupe_key_source"),
        "dedupe_backend": dedupe_health.get("backend"),
        "dedupe_degraded": dedupe_health.get("degraded"),
        "dedupe_ttl_seconds": dedupe_health.get("ttl_seconds"),
    }
    if not is_first_seen:
        logger.info(
            "Feishu duplicate event acknowledged",
            extra={
                **dedupe_log_fields,
                "dedupe_result": "duplicate",
                "feishu_retry_likely": event_data.get("dedupe_key_source") == "message_id",
            },
        )
        return {"code": 0}

    if event_data.get("event_kind") == "file":
        logger.info(
            "Feishu file event accepted",
            extra={
                **dedupe_log_fields,
                "dedupe_result": "accepted",
                "chat_type": event_data.get("chat_type"),
            },
        )
        background_tasks.add_task(process_feishu_file_event, event_data)
        return {"code": 0}

    # 5. 后台处理，确保回调入口快速返回。
    logger.info(
        "Feishu message event accepted",
        extra={
            **dedupe_log_fields,
            "dedupe_result": "accepted",
            "chat_type": event_data.get("chat_type"),
        },
    )
    background_tasks.add_task(process_feishu_message, event_data)
    return {"code": 0}


def parse_feishu_card_action(body: dict) -> dict | None:
    header = body.get("header", {})
    if header.get("event_type") != "card.action.trigger":
        return None
    event = body.get("event", {})
    action = event.get("action", {})
    value = action.get("value") or {}
    operator = event.get("operator", {})
    open_id = (
        operator.get("operator_id", {}).get("open_id")
        or operator.get("user_id", {}).get("open_id")
        or operator.get("open_id")
        or event.get("user_id", {}).get("open_id")
    )
    return {
        "action": value.get("action"),
        "pending_id": value.get("pending_id"),
        "open_id": open_id,
        "chat_id": event.get("context", {}).get("open_chat_id") or event.get("context", {}).get("chat_id"),
    }


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
