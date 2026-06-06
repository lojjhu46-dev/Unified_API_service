"""编排器"""

import asyncio
import json
import re
import uuid
import time
import unicodedata
from difflib import SequenceMatcher
from app.schemas import (
    AskRequest,
    AgentResponse,
    SourceItem,
    TimingInfo,
)
from app.llm.gateway import llm_gateway
from app.llm.prompts import build_direct_prompt, build_rag_prompt, build_web_search_prompt
from app.retrieval.retriever import retriever
from app.memory.store import memory_store
from app.memory.rewrite import _history_to_text, rewrite_question
from app.tools.registry import tool_registry
from app.tools.calculator import extract_math_expression
from app.tools.search import format_search_results
from app.config import settings
from app.observability.logging import get_logger

logger = get_logger(__name__)

_NON_REUSABLE_RECENT_ANSWER_MARKERS = (
    "[MOCK回答]",
    "模拟检索结果",
    "Mock检索",
    "mock检索",
    "使用Mock检索器",
    "无法从文档中找到依据",
    "没有找到依据",
    "文档片段中没有找到",
    "未包含关于",
    "检索失败",
    "加载Embedding模型失败",
    "获取向量存储失败",
)


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
            # 决定路由
            route = self._decide_route(request)
            standalone_question = request.question
            rewrite_ms = 0
            reuse_history = []
            rewrite_history = []
            prompt_history = []

            if route == "rag":
                reuse_history = await self.memory.get_history(
                    session_id,
                    self._memory_reuse_window_messages(),
                )
                reuse_decision = self._evaluate_recent_answer_reuse(request.question, reuse_history)
                if reuse_decision["hit"]:
                    result = {
                        "answer": reuse_decision["answer"],
                        "sources": [],
                        "tool_trace": [],
                        "retrieval_ms": 0,
                        "tool_ms": 0,
                        "llm_ms": 0,
                    }
                    logger.info(
                        "Recent answer reuse hit",
                        extra={
                            "request_id": request_id,
                            "session_id": session_id,
                            "route": route,
                            "recent_answer_reuse_hit": True,
                            "recent_answer_reuse_similarity": reuse_decision["similarity"],
                            "recent_answer_reuse_turn_offset": reuse_decision["turn_offset"],
                            "recent_answer_reuse_candidate_count": reuse_decision["candidate_count"],
                            "recent_answer_reuse_threshold": reuse_decision["threshold"],
                            "recent_answer_reuse_min_chars": reuse_decision["min_chars"],
                            "memory_reuse_history_count": len(reuse_history),
                            "memory_rewrite_history_count": 0,
                            "memory_prompt_history_count": 0,
                            "memory_max_messages": settings.memory_max_messages,
                        },
                    )
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
                            rewrite_ms=0,
                            retrieval_ms=0,
                            tool_ms=0,
                            llm_ms=0,
                            total_ms=total_ms,
                        ),
                        standalone_question=request.question,
                    )
                logger.info(
                    "Recent answer reuse miss",
                    extra={
                        "request_id": request_id,
                        "session_id": session_id,
                        "route": route,
                        "recent_answer_reuse_hit": False,
                        "recent_answer_reuse_miss_reason": reuse_decision["miss_reason"],
                        "recent_answer_reuse_best_similarity": reuse_decision["best_similarity"],
                        "recent_answer_reuse_turn_offset": reuse_decision["best_turn_offset"],
                        "recent_answer_reuse_candidate_count": reuse_decision["candidate_count"],
                        "recent_answer_reuse_threshold": reuse_decision["threshold"],
                        "recent_answer_reuse_min_chars": reuse_decision["min_chars"],
                        "memory_reuse_history_count": len(reuse_history),
                        "memory_rewrite_history_count": 0,
                        "memory_prompt_history_count": 0,
                        "memory_max_messages": settings.memory_max_messages,
                    },
                )

            if route in {"rag", "web", "agentic_rag"}:
                rewrite_history = await self.memory.get_history(
                    session_id,
                    self._memory_rewrite_window_messages(),
                )
                prompt_history = await self.memory.get_history(
                    session_id,
                    self._memory_prompt_window_messages(),
                )
                logger.info(
                    "Memory context windows loaded",
                    extra={
                        "request_id": request_id,
                        "session_id": session_id,
                        "route": route,
                        "memory_reuse_history_count": len(reuse_history),
                        "memory_rewrite_history_count": len(rewrite_history),
                        "memory_prompt_history_count": len(prompt_history),
                        "memory_max_messages": settings.memory_max_messages,
                    },
                )

            if route != "rag" and rewrite_history:
                logger.info(
                    "Recent answer reuse skipped",
                    extra={
                        "request_id": request_id,
                        "session_id": session_id,
                        "route": route,
                        "recent_answer_reuse_hit": False,
                        "recent_answer_reuse_miss_reason": "route_not_rag",
                        "memory_rewrite_history_count": len(rewrite_history),
                        "memory_prompt_history_count": len(prompt_history),
                        "memory_max_messages": settings.memory_max_messages,
                    },
                )

            # 追问改写
            if rewrite_history and route in {"rag", "web", "agentic_rag"}:
                start = time.perf_counter()
                standalone_question = await rewrite_question(request.question, rewrite_history)
                rewrite_ms = (time.perf_counter() - start) * 1000

            # 执行对应路由
            if route == "direct":
                result = await self._handle_direct(request)
            elif route == "rag":
                result = await self._handle_rag(request, standalone_question, prompt_history)
            elif route == "web":
                result = await self._handle_web(request, standalone_question, prompt_history)
            elif route == "tool":
                result = await self._handle_tool(request)
            elif route == "agentic_rag":
                result = await self._handle_agentic_rag(request, standalone_question, prompt_history)
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
                standalone_question=standalone_question if rewrite_history else None,
            )
        except Exception as e:
            logger.error(
                f"处理请求失败: {e}",
                extra={"request_id": request_id},
                exc_info=True,
            )
            raise

    def _normalize_recent_answer_text(self, text: str) -> str:
        """归一化最近答案复用的比较文本。"""
        normalized = unicodedata.normalize("NFKC", text or "").casefold()
        return "".join(
            char
            for char in normalized
            if unicodedata.category(char)[0] not in {"P", "Z", "S"}
        )

    def _memory_window(self, value: int | None) -> int:
        """Return a positive history window, falling back to the legacy setting."""
        fallback = max(int(settings.memory_window_messages), 1)
        try:
            return max(int(value if value is not None else fallback), 1)
        except (TypeError, ValueError):
            return fallback

    def _memory_reuse_window_messages(self) -> int:
        return self._memory_window(getattr(settings, "memory_reuse_window_messages", None))

    def _memory_rewrite_window_messages(self) -> int:
        return self._memory_window(getattr(settings, "memory_rewrite_window_messages", None))

    def _memory_prompt_window_messages(self) -> int:
        return self._memory_window(getattr(settings, "memory_prompt_window_messages", None))

    def _iter_recent_turns(self, history: list[dict]) -> list[dict]:
        """从消息历史中重建问答轮次。"""
        turns = []
        current_question = None
        for item in history:
            role = item.get("role")
            content = (item.get("content") or "").strip()
            if not content:
                continue
            if role == "user":
                current_question = content
                continue
            if role == "assistant" and current_question is not None:
                turns.append(
                    {
                        "question": current_question,
                        "answer": content,
                    }
                )
                current_question = None
        return turns

    def _is_reusable_recent_answer(self, answer: str) -> bool:
        """过滤非正常检索产生的历史答案，避免复用错误或Mock答案。"""
        if not (answer or "").strip():
            return False
        return not any(marker in answer for marker in _NON_REUSABLE_RECENT_ANSWER_MARKERS)

    def _evaluate_recent_answer_reuse(self, question: str, history: list[dict]) -> dict:
        """评估是否可以直接复用最近答案，并返回命中或未命中原因。"""
        min_chars = max(int(settings.recent_answer_reuse_min_chars), 1)
        threshold = float(settings.recent_answer_reuse_similarity_threshold)
        base_result = {
            "hit": False,
            "answer": None,
            "question": None,
            "similarity": 0.0,
            "turn_offset": None,
            "miss_reason": None,
            "best_similarity": 0.0,
            "best_turn_offset": None,
            "candidate_count": 0,
            "threshold": threshold,
            "min_chars": min_chars,
        }

        if not history:
            return {**base_result, "miss_reason": "no_history"}

        normalized_question = self._normalize_recent_answer_text(question)
        if not normalized_question:
            return {**base_result, "miss_reason": "empty_question"}

        turns = self._iter_recent_turns(history)
        if not turns:
            return {**base_result, "miss_reason": "no_complete_turns"}

        best_similarity = 0.0
        best_turn_offset = None
        candidate_count = 0
        non_reusable_answer_count = 0
        short_question_blocked = False
        for turn_offset, turn in enumerate(reversed(turns)):
            candidate_question = turn.get("question", "")
            candidate_answer = turn.get("answer", "")
            if not candidate_answer:
                continue
            if not self._is_reusable_recent_answer(candidate_answer):
                non_reusable_answer_count += 1
                continue

            normalized_candidate = self._normalize_recent_answer_text(candidate_question)
            if not normalized_candidate:
                continue

            candidate_count += 1
            if min(len(normalized_question), len(normalized_candidate)) < min_chars:
                similarity = 1.0 if normalized_question == normalized_candidate else 0.0
                if similarity == 0.0:
                    short_question_blocked = True
            else:
                similarity = SequenceMatcher(
                    None,
                    normalized_question,
                    normalized_candidate,
                ).ratio()

            if similarity > best_similarity:
                best_similarity = similarity
                best_turn_offset = turn_offset

            if similarity > threshold:
                return {
                    **base_result,
                    "hit": True,
                    "answer": candidate_answer,
                    "question": candidate_question,
                    "similarity": similarity,
                    "turn_offset": turn_offset,
                    "miss_reason": None,
                    "best_similarity": similarity,
                    "best_turn_offset": turn_offset,
                    "candidate_count": candidate_count,
                }

        if candidate_count == 0:
            miss_reason = "non_reusable_answer" if non_reusable_answer_count else "no_usable_answer"
        elif short_question_blocked and best_similarity == 0.0:
            miss_reason = "short_question_not_exact"
        else:
            miss_reason = "similarity_below_threshold"

        return {
            **base_result,
            "miss_reason": miss_reason,
            "best_similarity": best_similarity,
            "best_turn_offset": best_turn_offset,
            "candidate_count": candidate_count,
        }

    def _find_recent_answer_reuse(self, question: str, history: list[dict]) -> dict | None:
        """查找是否可以直接复用最近答案。"""
        decision = self._evaluate_recent_answer_reuse(question, history)
        if not decision["hit"]:
            return None
        return {
            "answer": decision["answer"],
            "question": decision["question"],
            "similarity": decision["similarity"],
            "turn_offset": decision["turn_offset"],
        }

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

        # 文档操作优先于联网和计算（避免 .txt/.docx 被误判为 URL 域名）
        if self._is_document_request(request):
            return "tool"

        # 显式联网优先级最高，避免被"多少"等词误判为计算。
        if request.need_web == "always":
            return "web"
        if request.need_web == "auto":
            web_words = [
                "联网",
                "搜索",
                "网上",
                "查一下",
                "最新",
                "今天",
                "新闻",
                "资讯",
                "天气",
                "气温",
                "价格",
                "股价",
                "汇率",
                "论坛",
                "社区",
                "帖子",
                "技术方案",
                "解决方案",
                "官方文档",
                "报错",
                "故障",
                "site:",
                "网站",
                "github",
                "stackoverflow",
                "reddit",
                "v2ex",
            ]
            if any(word in question for word in web_words) or re.search(
                r"(?:https?://|site:)?[a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)+",
                question,
            ):
                return "agentic_rag"

        # 需要计算。要求能提取出真实数学表达式，避免"今天气温多少"误入计算器。
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
        has_math_word = bool(
            re.search(r"(计算|换算|等于|加上|减去|乘以|除以)", question)
            or re.search(r"(?<!删)[加减乘除](?![一-龥A-Za-z])", question)
        )
        return has_digit and (has_operator or has_math_word)

    # ---- 文档请求识别 ----

    _DOC_EXTENSIONS = (".docx", ".xlsx", ".txt", ".pdf")
    _DOC_EDIT_INTENT_WORDS = frozenset({
        "修改", "编辑", "替换", "删除", "改写", "添加", "插入", "清空", "移除", "更新", "改动",
        "edit", "replace", "delete", "modify", "remove", "clear",
    })
    _DOC_REVIEW_INTENT_WORDS = frozenset({
        "审阅", "审阅文档", "提取", "提取结构", "总结", "总结文档", "查看", "查看结构",
        "review", "extract", "summarize",
    })
    _DOC_LIST_VERBS = frozenset({
        "列出", "查看", "看看", "有哪些", "有什么", "显示", "展示",
        "list", "show", "view",
    })
    _DOC_LIST_OBJECTS = frozenset({
        "文件", "文档", "个人知识库文件", "知识库文件", "保存的文件", "已保存",
        "files", "documents",
    })

    def _is_list_intent(self, question: str) -> bool:
        """判断是否为列出文件意图：必须同时命中动词和对象词"""
        q = question.lower()
        has_verb = any(v in q for v in self._DOC_LIST_VERBS)
        has_obj = any(o in q for o in self._DOC_LIST_OBJECTS)
        return has_verb and has_obj

    def _is_document_request(self, request: AskRequest) -> bool:
        """判断是否为文档操作请求"""
        # 结构化字段优先
        if request.document_file_path:
            return True
        if request.document_plan:
            return True
        if request.document_action != "auto":
            return True

        question = request.question

        # 自然语言识别：列出个人知识库文件
        if self._is_list_intent(question):
            return True

        # 自然语言识别：文件路径 + 意图词
        has_doc_path = any(ext in question.lower() for ext in self._DOC_EXTENSIONS)
        has_intent = any(
            word in question
            for word in self._DOC_EDIT_INTENT_WORDS | self._DOC_REVIEW_INTENT_WORDS
        )
        if has_doc_path:
            return has_intent

        # 用户明确说“文档/文件/内容”并带编辑意图时，仍走文档工具，
        # 由工具层提示补充 file_path，避免编号文本被误判成计算。
        has_document_subject = any(word in question for word in ["文档", "文件", "正文"])
        return has_document_subject and self._has_edit_intent(question)

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
        subquestions = await self._prepare_rag_subquestions(standalone_question)
        sources = await self.retriever.search(
            standalone_question,
            request.top_k,
            request.knowledge_scope,
            owner_open_id=self._owner_open_id_for_request(request),
            subquestions=subquestions,
        )
        retrieval_ms = (time.perf_counter() - start) * 1000

        context = self._build_sources_context(sources)
        enriched_question = (
            f"最近对话历史：\n{_history_to_text(history)}\n\n"
            f"用户原始问题：\n{request.question}\n\n"
            f"独立检索问题：\n{standalone_question}"
        )
        system_prompt, prompt = build_rag_prompt(
            enriched_question,
            context,
            is_global=self._is_global_question(request.question, standalone_question),
        )

        start = time.perf_counter()
        answer = await self.llm.generate(prompt, system_prompt=system_prompt, allow_mock=False)
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

    def _extract_search_domains(self, question: str) -> list[str]:
        """从自然语言里的 URL、site: 语法中提取站点限制。"""
        text = question or ""
        candidates = []
        candidates.extend(re.findall(r"site:([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})", text, flags=re.IGNORECASE))
        candidates.extend(re.findall(r"https?://([^/\s，。]+)", text, flags=re.IGNORECASE))
        candidates.extend(re.findall(r"\b([a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)+)\b", text))

        domains = []
        for candidate in candidates:
            domain = candidate.lower().strip(" .，,。")
            if domain.startswith("www."):
                domain = domain[4:]
            if "." in domain and domain not in domains:
                domains.append(domain)
        return domains

    def _clean_site_syntax_from_query(self, question: str) -> str:
        query = re.sub(r"site:[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", "", question or "", flags=re.IGNORECASE)
        query = re.sub(r"https?://\S+", "", query, flags=re.IGNORECASE)
        query = re.sub(r"\b[a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)+\b", "", query)
        return " ".join(query.split()) or question

    def _classify_web_search_intent(self, question: str, domains: list[str] | None = None) -> str:
        text = (question or "").lower()
        if domains:
            return "site"
        if any(word in text for word in ["天气", "气温", "降雨", "下雨", "台风", "预报"]):
            return "weather"
        if any(word in text for word in ["新闻", "资讯", "最新", "快讯", "热点"]):
            return "news"
        if any(word in text for word in ["论坛", "社区", "帖子", "reddit", "v2ex", "知乎"]):
            return "forum"
        if any(word in text for word in ["技术方案", "解决方案", "官方文档", "报错", "bug", "github", "stackoverflow"]):
            return "tech"
        return "general"

    def _build_web_search_input(self, question: str) -> dict:
        domains = self._extract_search_domains(question)
        query = self._clean_site_syntax_from_query(question) if domains else question
        search_input = {"query": query}
        if domains:
            search_input["domains"] = domains
        return search_input

    def _is_web_first_question(self, question: str, domains: list[str] | None = None) -> bool:
        return self._classify_web_search_intent(question, domains) in {"weather", "news", "forum", "tech", "site"}

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

    def _build_sources_context(self, sources: list[SourceItem]) -> str:
        """构造给LLM使用的完整上下文，API展示摘要仍由 snippet 承担。"""
        parts = []
        total_chars = 0
        for index, source in enumerate(sources, start=1):
            body = source.content or source.snippet
            if not body:
                continue
            title = source.title or f"来源{index}"
            part = f"[{index}] {title}\n{body}"
            remaining = settings.rag_context_max_chars - total_chars
            if remaining <= 0:
                break
            if len(part) > remaining:
                part = part[:remaining]
            parts.append(part)
            total_chars += len(part)
        return "\n\n".join(parts)

    def _is_global_question(self, question: str, standalone_question: str) -> bool:
        text = f"{question}\n{standalone_question}"
        keywords = ["主要内容", "时代演变", "历史影响", "有哪些", "列出", "简介", "概括"]
        return any(keyword in text for keyword in keywords)

    def _owner_open_id_for_request(self, request: AskRequest) -> str | None:
        """Use the trusted request user as personal knowledge-base owner."""
        user_id = (request.user_id or "").strip()
        return user_id or None

    async def _prepare_rag_subquestions(self, question: str) -> list[str] | None:
        """准备可选 LLM 子问题；规则可拆时交给 Retriever 处理。"""
        rule_variants = self.retriever._build_query_variants(question)
        if len(rule_variants) > 1:
            logger.info(
                "RAG subquestion split uses rule variants",
                extra={
                    "split_strategy": "rule",
                    "subquestion_count": len(rule_variants),
                },
            )
            return None
        if not self._should_llm_split_subquestions(question):
            return None

        prompt = (
            "请把下面的知识库检索问题拆成最多4个可独立检索的小问题。"
            "只返回JSON数组，数组元素是字符串，不要输出解释。\n\n"
            f"问题：{question}"
        )
        try:
            raw = await self.llm.generate(
                prompt,
                system_prompt="你只负责把复杂检索问题拆成小问题。",
                max_tokens=300,
                temperature=0,
                allow_mock=False,
            )
            subquestions = self._parse_llm_subquestions(raw, question)
        except Exception as e:
            logger.info(
                "RAG LLM subquestion split skipped",
                extra={
                    "split_strategy": "llm",
                    "subquestion_count": 0,
                    "fallback_reason": "llm_split_failed",
                    "error": str(e),
                },
            )
            return None

        if not subquestions:
            return None
        logger.info(
            "RAG subquestion split uses LLM variants",
            extra={
                "split_strategy": "llm",
                "subquestion_count": len(subquestions),
            },
        )
        return subquestions

    def _should_llm_split_subquestions(self, question: str) -> bool:
        if not settings.rag_llm_subquestion_split_enabled or self.llm.provider == "mock":
            return False
        text = question or ""
        if len(text) < 24:
            return False
        intent_words = ["查找", "检索", "找出", "比较", "总结", "列出", "相关文本", "代码片段"]
        separators = ["和", "与", "以及", "并且", "同时"]
        return any(word in text for word in intent_words) and any(separator in text for separator in separators)

    def _parse_llm_subquestions(self, raw: str, original_question: str) -> list[str]:
        text = (raw or "").strip()
        if not text:
            return []
        match = re.search(r"\[[\s\S]*\]", text)
        if match:
            text = match.group(0)
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return []
        if not isinstance(payload, list):
            return []

        subquestions = []
        max_count = max(int(settings.rag_parallel_subquestion_max), 1)
        for item in payload:
            if not isinstance(item, str):
                continue
            candidate = item.strip()
            if not (2 <= len(candidate) <= 200):
                continue
            if candidate == original_question or candidate in subquestions:
                continue
            subquestions.append(candidate)
            if len(subquestions) >= max_count:
                break
        return subquestions

    async def _search_local_sources(
        self,
        request: AskRequest,
        standalone_question: str,
    ) -> tuple[list[SourceItem], float]:
        """检索本地知识库并返回耗时"""
        start = time.perf_counter()
        subquestions = await self._prepare_rag_subquestions(standalone_question)
        sources = await self.retriever.search(
            standalone_question,
            request.top_k,
            request.knowledge_scope,
            owner_open_id=self._owner_open_id_for_request(request),
            subquestions=subquestions,
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
        search_input = self._build_web_search_input(standalone_question)
        execution = await self.tools.execute_with_result("web_search", search_input)
        tool_trace.append(execution.trace)
        search_result = execution.result

        if execution.trace.status == "success" and search_result.get("success"):
            results = search_result.get("results", [])
            sources = self._web_sources_from_results(results)

            search_text = format_search_results(results)
            system_prompt, prompt = build_web_search_prompt(request.question, search_text)

            start = time.perf_counter()
            answer = await self.llm.generate(prompt, system_prompt=system_prompt, allow_mock=False)
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
            context = self._build_sources_context(local_sources)
            enriched_question = (
                f"{self._build_enriched_question(request, standalone_question, history)}\n\n"
                "联网搜索失败，请明确说明未能获取最新联网信息，并仅基于本地知识库回答。"
            )
            system_prompt, prompt = build_rag_prompt(
                enriched_question,
                context,
                is_global=self._is_global_question(request.question, standalone_question),
            )

            start = time.perf_counter()
            answer = await self.llm.generate(prompt, system_prompt=system_prompt, allow_mock=False)
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

        # ---- 文档操作优先 ----
        if self._is_document_request(request):
            return await self._handle_document_tool(request, tool_trace)

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

    async def _handle_document_tool(
        self,
        request: AskRequest,
        tool_trace: list,
    ) -> dict:
        """处理文档工具调用"""
        file_path = request.document_file_path or self._extract_file_path(request.question)
        file_type = request.document_file_type
        action = request.document_action

        # 列出个人知识库文件
        if action == "list" or (not file_path and not request.document_plan and self._is_list_intent(request.question)):
            execution = await self.tools.execute_with_result("document_list_personal_files", {
                "owner_user_id": request.user_id,
            })
            tool_trace.append(execution.trace)
            result = execution.result
            if result.get("success") and result.get("files"):
                files = result["files"]
                file_list = "\n".join(
                    f"  {i+1}. {f['original_filename']}（保存于 {f['created_at'][:19]}）"
                    for i, f in enumerate(files)
                )
                answer = f"您的个人知识库文件（共 {result['count']} 个）：\n{file_list}"
            elif result.get("success"):
                answer = "您的个人知识库暂无文件。"
            else:
                answer = f"查询失败：{result.get('error', '未知错误')}"
            return {
                "answer": answer,
                "sources": [],
                "tool_trace": tool_trace,
                "tool_ms": execution.trace.latency_ms,
                "llm_ms": 0,
            }

        # apply：执行已有 plan
        if action == "apply" or request.document_plan:
            if not request.document_plan:
                return {
                    "answer": "请先通过 plan 操作生成编辑方案，再提交执行。",
                    "sources": [],
                    "tool_trace": tool_trace,
                    "tool_ms": 0,
                    "llm_ms": 0,
                }
            # 所有 EDIT plan 都需要确认，不只是高风险
            plan_intent = request.document_plan.get("intent", "")
            if plan_intent == "edit" and not request.document_confirmed:
                return {
                    "answer": "编辑方案已收到，请确认后执行（设置 document_confirmed=true）。",
                    "sources": [],
                    "tool_trace": tool_trace,
                    "tool_ms": 0,
                    "llm_ms": 0,
                }
            execution = await self.tools.execute_with_result("document_apply_plan", {
                "plan": request.document_plan,
                "confirmed": request.document_confirmed,
                "owner_user_id": request.user_id,
            })
            tool_trace.append(execution.trace)
            result = execution.result
            if result.get("success"):
                output_file = result.get("result", {}).get("output_file", "")
                summary = result.get("result", {}).get("summary", "")
                answer = f"编辑执行成功。{summary}"
                if output_file:
                    answer += f"\n输出文件：{output_file}"
            elif result.get("requires_confirmation"):
                answer = "该操作风险较高，请确认后重新提交（设置 document_confirmed=true）。"
            else:
                answer = f"编辑执行失败：{result.get('error', '未知错误')}"
            return {
                "answer": answer,
                "sources": [],
                "tool_trace": tool_trace,
                "tool_ms": execution.trace.latency_ms,
                "llm_ms": 0,
            }

        if not file_path and action in {"auto", "plan", "extract", "review"}:
            # 提示用户查看个人知识库文件列表
            list_result = await self.tools.execute_with_result("document_list_personal_files", {
                "owner_user_id": request.user_id,
                "limit": 10,
            })
            if list_result.result.get("success") and list_result.result.get("files"):
                files = list_result.result["files"]
                file_list = "\n".join(
                    f"  - {f['original_filename']}（{f['created_at'][:10]}）" for f in files
                )
                answer = f"请指定要处理的文件路径。您个人知识库中的文件：\n{file_list}"
            else:
                answer = "请补充要处理的文档文件路径。可通过 document_file_path 传入已上传文件路径，或先上传文件到个人知识库。"
            return {
                "answer": answer,
                "sources": [],
                "tool_trace": tool_trace,
                "tool_ms": 0,
                "llm_ms": 0,
            }

        # extract：提取结构
        if action == "extract":
            execution = await self.tools.execute_with_result("document_extract", {
                "file_path": file_path,
                "file_type": file_type,
            })
            tool_trace.append(execution.trace)
            result = execution.result
            if result.get("success"):
                structure = result.get("structure", {})
                answer = f"文档结构提取成功：\n类型：{result.get('file_type')}\n结构：{json.dumps(structure, ensure_ascii=False, indent=2)}"
            else:
                answer = f"文档结构提取失败：{result.get('error', '未知错误')}"
            return {
                "answer": answer,
                "sources": [],
                "tool_trace": tool_trace,
                "tool_ms": execution.trace.latency_ms,
                "llm_ms": 0,
            }

        # plan：生成编辑方案
        if action == "plan" or (file_path and self._has_edit_intent(request.question)):
            execution = await self.tools.execute_with_result("document_plan", {
                "user_command": request.question,
                "file_path": file_path,
                "file_type": file_type,
                "owner_user_id": request.user_id,
            })
            tool_trace.append(execution.trace)
            result = execution.result
            if result.get("success"):
                plan = result.get("plan", {})
                plan_intent = plan.get("intent", "")
                unsupported_reason = plan.get("unsupported_reason")
                clarification = plan.get("clarification_question")
                if plan_intent == "unsupported":
                    answer = f"无法生成编辑方案：{unsupported_reason or '该操作不支持'}"
                elif clarification:
                    answer = f"需要补充信息：{clarification}"
                else:
                    answer = (
                        "已生成编辑方案，请确认后执行（设置 document_confirmed=true）：\n"
                        f"```json\n{json.dumps(plan, ensure_ascii=False, indent=2)}\n```"
                    )
            else:
                answer = f"编辑方案生成失败：{result.get('error', '未知错误')}"
            return {
                "answer": answer,
                "sources": [],
                "tool_trace": tool_trace,
                "tool_ms": execution.trace.latency_ms,
                "llm_ms": 0,
            }

        # 默认：review（审阅/总结）
        execution = await self.tools.execute_with_result("document_review", {
            "file_path": file_path,
            "file_type": file_type,
        })
        tool_trace.append(execution.trace)
        result = execution.result
        if result.get("success"):
            answer = f"文档审阅结果：\n{result.get('summary', '')}"
            structure = result.get("structure")
            if structure:
                answer += f"\n\n文档结构：{json.dumps(structure, ensure_ascii=False)}"
        else:
            answer = f"文档审阅失败：{result.get('error', '未知错误')}"
        return {
            "answer": answer,
            "sources": [],
            "tool_trace": tool_trace,
            "tool_ms": execution.trace.latency_ms,
            "llm_ms": 0,
        }

    def _has_edit_intent(self, question: str) -> bool:
        """判断问题是否包含编辑意图"""
        return any(word in question for word in self._DOC_EDIT_INTENT_WORDS)

    @staticmethod
    def _extract_file_path(question: str) -> str:
        """从自然语言中提取文件路径"""
        # Windows 路径：D:\docs\file.docx
        match = re.search(r'[A-Za-z]:\\[^\s，。？！]+', question)
        if match:
            return match.group(0)
        # Linux/WSL 路径：/tmp/file.txt
        match = re.search(r'/(?:tmp|home|root|mnt|opt|var)[^\s，。？！]+', question)
        if match:
            return match.group(0)
        # 引号包裹路径
        match = re.search(r'[""「]([^""」]+\.\w{3,4})[""」]', question)
        if match:
            return match.group(1)
        return ""

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
        search_input = self._build_web_search_input(standalone_question)
        search_task = self.tools.execute_with_result("web_search", search_input)

        (rag_results, retrieval_ms), search_execution = await asyncio.gather(rag_task, search_task)

        # 处理 RAG 结果
        for r in rag_results:
            sources.append(r)

        # 处理搜索结果
        tool_trace.append(search_execution.trace)
        if search_execution.trace.status == "success" and search_execution.result.get("success"):
            sources.extend(self._web_sources_from_results(search_execution.result.get("results", []), limit=3))

        # 生成综合回答
        local_context = self._build_sources_context(rag_results)
        web_context = "\n\n".join([
            s.snippet for s in sources if s.source_type == "web_search"
        ])
        context = f"本地知识库片段：\n{local_context or '无'}\n\n联网搜索片段：\n{web_context or '无'}"
        enriched_question = (
            f"{self._build_enriched_question(request, standalone_question, history)}\n\n"
            "请综合本地知识库片段和联网搜索片段回答；如果联网搜索失败或为空，需要明确说明。"
        )
        if search_execution.trace.status == "success" and search_execution.result.get("success") and self._is_web_first_question(
            standalone_question,
            search_input.get("domains"),
        ):
            search_text = format_search_results(search_execution.result.get("results", []))
            web_first_context = (
                f"联网搜索片段：\n{search_text or '无'}\n\n"
                f"本地知识库片段（仅在相关时辅助，若无关请忽略）：\n{local_context or '无'}"
            )
            system_prompt, prompt = build_web_search_prompt(request.question, web_first_context)
        else:
            system_prompt, prompt = build_rag_prompt(
                enriched_question,
                context,
                is_global=self._is_global_question(request.question, standalone_question),
            )

        start = time.perf_counter()
        answer = await self.llm.generate(prompt, system_prompt=system_prompt, allow_mock=False)
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
