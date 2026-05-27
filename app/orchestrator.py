"""编排器"""

import uuid
import time
from app.schemas import (
    AskRequest,
    AgentResponse,
    SourceItem,
    ToolTrace,
    TimingInfo,
)
from app.llm.gateway import llm_gateway
from app.llm.prompts import build_direct_prompt, build_rag_prompt
from app.retrieval.retriever import retriever
from app.memory.store import memory_store
from app.memory.rewrite import _history_to_text, rewrite_question
from app.observability.logging import get_logger

logger = get_logger(__name__)


class Orchestrator:
    """编排器"""

    def __init__(self):
        self.llm = llm_gateway
        self.retriever = retriever
        self.memory = memory_store

    async def process(self, request: AskRequest) -> AgentResponse:
        """处理请求"""
        request_id = str(uuid.uuid4())[:12]
        session_id = request.session_id or await self.memory.create_session()
        start_time = time.perf_counter()

        logger.info(
            f"处理请求",
            extra={
                "request_id": request_id,
                "user_id": request.user_id,
                "channel": request.channel,
                "session_id": session_id,
            },
        )

        try:
            # 读取历史
            history = await self.memory.get_history(session_id)

            # 追问改写
            standalone_question = request.question
            rewrite_ms = 0
            if history:
                start = time.perf_counter()
                standalone_question = await rewrite_question(request.question, history)
                rewrite_ms = (time.perf_counter() - start) * 1000

            # 决定路由
            route = self._decide_route(request)

            # 执行对应路由
            if route == "direct":
                result = await self._handle_direct(request)
            elif route == "rag":
                result = await self._handle_rag(request, standalone_question, history)
            else:
                result = await self._handle_direct(request)

            # 写入历史
            await self.memory.append_turn(session_id, request.question, result["answer"])

            total_ms = (time.perf_counter() - start_time) * 1000

            return AgentResponse(
                request_id=request_id,
                session_id=session_id,
                route=route,
                answer=result.get("answer", ""),
                sources=result.get("sources", []),
                tool_trace=result.get("tool_trace", []),
                timing=TimingInfo(
                    rewrite_ms=rewrite_ms,
                    retrieval_ms=result.get("retrieval_ms", 0),
                    tool_ms=result.get("tool_ms", 0),
                    llm_ms=result.get("llm_ms", 0),
                    total_ms=total_ms,
                ),
                standalone_question=standalone_question if history else None,
            )
        except Exception as e:
            logger.error(
                f"处理请求失败: {e}",
                extra={"request_id": request_id},
                exc_info=True,
            )
            raise

    def _decide_route(self, request: AskRequest) -> str:
        """决定路由"""
        question = request.question.strip().lower()

        direct_phrases = {
            "你好",
            "您好",
            "hello",
            "hi",
            "谢谢",
            "thanks",
            "thank you",
        }
        if question in direct_phrases:
            return "direct"

        return "rag"

    async def _handle_direct(self, request: AskRequest) -> dict:
        """处理直接问答"""
        system_prompt, prompt = build_direct_prompt(request.question)
        start = time.perf_counter()
        answer = await self.llm.generate(prompt, system_prompt=system_prompt)
        llm_ms = (time.perf_counter() - start) * 1000

        return {
            "answer": answer,
            "sources": [],
            "tool_trace": [],
            "llm_ms": llm_ms,
        }

    async def _handle_rag(
        self,
        request: AskRequest,
        standalone_question: str,
        history: list[dict],
    ) -> dict:
        """处理RAG问答"""
        start = time.perf_counter()
        sources = await self.retriever.search(
            standalone_question,
            request.top_k,
            request.knowledge_scope,
        )
        retrieval_ms = (time.perf_counter() - start) * 1000

        context = "\n\n".join([s.snippet for s in sources])
        enriched_question = (
            f"最近对话历史：\n{_history_to_text(history)}\n\n"
            f"用户原始问题：\n{request.question}\n\n"
            f"独立检索问题：\n{standalone_question}"
        )
        system_prompt, prompt = build_rag_prompt(enriched_question, context)

        start = time.perf_counter()
        answer = await self.llm.generate(prompt, system_prompt=system_prompt)
        llm_ms = (time.perf_counter() - start) * 1000

        return {
            "answer": answer,
            "sources": sources,
            "tool_trace": [],
            "retrieval_ms": retrieval_ms,
            "llm_ms": llm_ms,
        }


orchestrator = Orchestrator()
