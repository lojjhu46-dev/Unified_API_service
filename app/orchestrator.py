"""编排器"""

import asyncio
import uuid
import time
from app.schemas import (
    AskRequest,
    AgentResponse,
    SourceItem,
    TimingInfo,
)
from app.llm.gateway import llm_gateway
from app.llm.prompts import build_direct_prompt, build_rag_prompt
from app.retrieval.retriever import retriever
from app.memory.store import memory_store
from app.memory.rewrite import _history_to_text, rewrite_question
from app.tools.registry import tool_registry
from app.tools.calculator import extract_math_expression
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
                result = await self._handle_web(request, standalone_question, history)
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

        # 显式联网优先级最高，避免被“多少”等词误判为计算。
        if request.need_web == "always":
            return "web"
        if request.need_web == "auto":
            if any(word in question for word in ["最新", "今天", "新闻", "天气", "价格", "股价", "汇率"]):
                return "agentic_rag"

        # 需要计算。要求能提取出真实数学表达式，避免“今天气温多少”误入计算器。
        if self._is_calculation_question(question):
            return "tool"

        # 默认走 RAG
        return "rag"

    def _is_calculation_question(self, question: str) -> bool:
        """判断是否为可计算问题"""
        expression = extract_math_expression(question)
        if not expression:
            return False
        has_digit = any(ch.isdigit() for ch in expression)
        has_operator = any(op in expression for op in ["+", "-", "*", "/", "%", "^", ">", "<", "="])
        has_math_word = any(word in question for word in ["计算", "换算", "加", "减", "乘", "除", "等于"])
        return has_digit and (has_operator or has_math_word)

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

    def _web_sources_from_results(self, results: list[dict], limit: int | None = None) -> list[SourceItem]:
        """将联网搜索结果转换为统一来源结构"""
        selected = results[:limit] if limit else results
        return [
            SourceItem(
                title=r.get("title", ""),
                url=r.get("url", ""),
                source_type="web_search",
                snippet=r.get("snippet", ""),
                score=1.0,
            )
            for r in selected
        ]

    def _build_enriched_question(
        self,
        request: AskRequest,
        standalone_question: str,
        history: list[dict],
    ) -> str:
        """构造带历史与独立问题的最终提问上下文"""
        return (
            f"最近对话历史：\n{_history_to_text(history)}\n\n"
            f"用户原始问题：\n{request.question}\n\n"
            f"独立检索问题：\n{standalone_question}"
        )

    async def _search_local_sources(
        self,
        request: AskRequest,
        standalone_question: str,
    ) -> tuple[list[SourceItem], float]:
        """检索本地知识库并返回耗时"""
        start = time.perf_counter()
        sources = await self.retriever.search(
            standalone_question,
            request.top_k,
            request.knowledge_scope,
        )
        retrieval_ms = (time.perf_counter() - start) * 1000
        return sources, retrieval_ms

    async def _handle_web(
        self,
        request: AskRequest,
        standalone_question: str,
        history: list[dict],
    ) -> dict:
        """处理联网搜索"""
        tool_trace = []

        # 执行搜索
        execution = await self.tools.execute_with_result("web_search", {"query": standalone_question})
        tool_trace.append(execution.trace)
        search_result = execution.result

        if execution.trace.status == "success" and search_result.get("success"):
            results = search_result.get("results", [])
            sources = self._web_sources_from_results(results)

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
                "tool_ms": execution.trace.latency_ms,
                "llm_ms": llm_ms,
            }

        # 搜索失败时只降级到本地 RAG，不直接让 LLM 编造时效信息。
        local_sources, retrieval_ms = await self._search_local_sources(request, standalone_question)
        if local_sources:
            context = "\n\n".join([s.snippet for s in local_sources])
            enriched_question = (
                f"{self._build_enriched_question(request, standalone_question, history)}\n\n"
                "联网搜索失败，请明确说明未能获取最新联网信息，并仅基于本地知识库回答。"
            )
            system_prompt, prompt = build_rag_prompt(enriched_question, context)

            start = time.perf_counter()
            answer = await self.llm.generate(prompt, system_prompt=system_prompt)
            llm_ms = (time.perf_counter() - start) * 1000

            return {
                "answer": answer,
                "sources": local_sources,
                "tool_trace": tool_trace,
                "retrieval_ms": retrieval_ms,
                "tool_ms": execution.trace.latency_ms,
                "llm_ms": llm_ms,
            }

        return {
            "answer": "暂时无法获取联网搜索结果，也没有找到可用的本地知识库内容，请稍后重试。",
            "sources": [],
            "tool_trace": tool_trace,
            "retrieval_ms": retrieval_ms,
            "tool_ms": execution.trace.latency_ms,
            "llm_ms": 0,
        }

    async def _handle_tool(self, request: AskRequest) -> dict:
        """处理工具调用"""
        tool_trace = []

        # 判断是否为计算问题
        expression = extract_math_expression(request.question)
        if expression:
            execution = await self.tools.execute_with_result("calculator", {"expression": expression})
            tool_trace.append(execution.trace)

            if execution.trace.status == "success" and execution.result.get("success"):
                answer = f"计算结果：{execution.result.get('result')}"
                return {
                    "answer": answer,
                    "sources": [],
                    "tool_trace": tool_trace,
                    "tool_ms": execution.trace.latency_ms,
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
        rag_task = self._search_local_sources(request, standalone_question)
        search_task = self.tools.execute_with_result("web_search", {"query": standalone_question})

        (rag_results, retrieval_ms), search_execution = await asyncio.gather(rag_task, search_task)

        # 处理 RAG 结果
        for r in rag_results:
            sources.append(r)

        # 处理搜索结果
        tool_trace.append(search_execution.trace)
        if search_execution.trace.status == "success" and search_execution.result.get("success"):
            sources.extend(self._web_sources_from_results(search_execution.result.get("results", []), limit=3))

        # 生成综合回答
        local_context = "\n\n".join([s.snippet for s in rag_results])
        web_context = "\n\n".join([
            s.snippet for s in sources if s.source_type == "web_search"
        ])
        context = f"本地知识库片段：\n{local_context or '无'}\n\n联网搜索片段：\n{web_context or '无'}"
        enriched_question = (
            f"{self._build_enriched_question(request, standalone_question, history)}\n\n"
            "请综合本地知识库片段和联网搜索片段回答；如果联网搜索失败或为空，需要明确说明。"
        )
        system_prompt, prompt = build_rag_prompt(enriched_question, context)

        start = time.perf_counter()
        answer = await self.llm.generate(prompt, system_prompt=system_prompt)
        llm_ms = (time.perf_counter() - start) * 1000

        return {
            "answer": answer,
            "sources": sources,
            "tool_trace": tool_trace,
            "retrieval_ms": retrieval_ms,
            "tool_ms": search_execution.trace.latency_ms,
            "llm_ms": llm_ms,
        }


orchestrator = Orchestrator()
