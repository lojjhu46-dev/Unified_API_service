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
from app.tools.registry import tool_registry
from app.tools.search import format_search_results
from app.observability.logging import get_logger

logger = get_logger(__name__)


class Orchestrator:
    """编排器"""

    def __init__(self):
        self.llm = llm_gateway
        self.retriever = retriever
        self.memory = memory_store
        self.tools = tool_registry

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
            elif route == "web":
                result = await self._handle_web(request, standalone_question)
            elif route == "tool":
                result = await self._handle_tool(request)
            elif route == "agentic_rag":
                result = await self._handle_agentic_rag(request, standalone_question, history)
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

        # 纯寒暄/感谢走 direct
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

        # 需要计算
        if any(word in question for word in ["计算", "多少", "等于", "换算", "加", "减", "乘", "除"]):
            return "tool"

        # 需要联网搜索
        if request.need_web == "always":
            return "web"
        if request.need_web == "auto":
            if any(word in question for word in ["最新", "今天", "新闻", "天气", "价格", "股价", "汇率"]):
                return "web"

        # 默认走 RAG
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

    async def _handle_web(self, request: AskRequest, standalone_question: str) -> dict:
        """处理联网搜索"""
        tool_trace = []
        sources = []

        # 执行搜索
        trace = await self.tools.execute("web_search", {"query": standalone_question})
        tool_trace.append(trace)

        if trace.status == "success":
            # 解析搜索结果
            search_result = await self.tools._run_web_search({"query": standalone_question})
            if search_result.get("success"):
                results = search_result.get("results", [])
                for r in results:
                    sources.append(SourceItem(
                        title=r.get("title", ""),
                        url=r.get("url", ""),
                        source_type="web_search",
                        snippet=r.get("snippet", ""),
                        score=1.0,
                    ))

                # 生成回答
                search_text = format_search_results(results)
                prompt = f"基于以下搜索结果回答问题：\n\n{search_text}\n\n问题：{request.question}"
                system_prompt = "你是一个有用的中文助手。请基于搜索结果回答问题，并注明来源。"

                start = time.perf_counter()
                answer = await self.llm.generate(prompt, system_prompt=system_prompt)
                llm_ms = (time.perf_counter() - start) * 1000

                return {
                    "answer": answer,
                    "sources": sources,
                    "tool_trace": tool_trace,
                    "tool_ms": trace.latency_ms,
                    "llm_ms": llm_ms,
                }

        # 搜索失败，降级为直接回答
        return await self._handle_direct(request)

    async def _handle_tool(self, request: AskRequest) -> dict:
        """处理工具调用"""
        tool_trace = []

        # 判断是否为计算问题
        if any(word in request.question for word in ["计算", "多少", "等于", "换算"]):
            # 提取表达式（简化处理）
            expression = request.question
            trace = await self.tools.execute("calculator", {"expression": expression})
            tool_trace.append(trace)

            if trace.status == "success":
                result = await self.tools._run_calculator({"expression": expression})
                if result.get("success"):
                    answer = f"计算结果：{result.get('result')}"
                    return {
                        "answer": answer,
                        "sources": [],
                        "tool_trace": tool_trace,
                        "tool_ms": trace.latency_ms,
                        "llm_ms": 0,
                    }

        # 降级为直接回答
        return await self._handle_direct(request)

    async def _handle_agentic_rag(
        self,
        request: AskRequest,
        standalone_question: str,
        history: list[dict],
    ) -> dict:
        """处理Agentic RAG（结合RAG和搜索）"""
        tool_trace = []
        sources = []

        # 并行执行 RAG 和搜索
        import asyncio

        rag_task = self.retriever.search(
            standalone_question,
            request.top_k,
            request.knowledge_scope,
        )
        search_task = self.tools.execute("web_search", {"query": standalone_question})

        rag_results, search_trace = await asyncio.gather(rag_task, search_task)

        # 处理 RAG 结果
        for r in rag_results:
            sources.append(r)

        # 处理搜索结果
        tool_trace.append(search_trace)
        if search_trace.status == "success":
            search_result = await self.tools._run_web_search({"query": standalone_question})
            if search_result.get("success"):
                for r in search_result.get("results", [])[:3]:
                    sources.append(SourceItem(
                        title=r.get("title", ""),
                        url=r.get("url", ""),
                        source_type="web_search",
                        snippet=r.get("snippet", ""),
                        score=1.0,
                    ))

        # 生成综合回答
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
            "tool_trace": tool_trace,
            "tool_ms": search_trace.latency_ms,
            "llm_ms": llm_ms,
        }


orchestrator = Orchestrator()
