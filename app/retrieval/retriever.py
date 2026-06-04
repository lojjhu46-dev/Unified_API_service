"""检索器"""

import asyncio
import re
from typing import List
from app.schemas import SourceItem
from app.config import settings
from app.observability.logging import get_logger

logger = get_logger(__name__)


class Retriever:
    """检索器"""

    def __init__(self):
        self._use_chroma = False
        # 不在服务启动/import 阶段加载 HuggingFace embedding，避免网络重试阻塞 /health 和飞书 challenge。
        self._chroma_checked = False

    def refresh(self) -> bool:
        """刷新Chroma可用状态。上传成功后调用，确保后续查询优先使用真实向量库。"""
        self._chroma_checked = True
        try:
            import chromadb
            from app.retrieval.vector_store import get_vector_store
            get_vector_store()
            self._use_chroma = True
            logger.info("使用Chroma向量存储")
            return True
        except Exception:
            self._use_chroma = False
            logger.info("使用Mock检索器")
            return False

    async def search(
        self,
        query: str,
        top_k: int = 5,
        knowledge_scope: List[str] = None,
        owner_open_id: str | None = None,
        subquestions: List[str] | None = None,
    ) -> List[SourceItem]:
        """检索"""
        if self._use_chroma or (not self._chroma_checked and self.refresh()):
            try:
                return await self._chroma_search(query, top_k, knowledge_scope, owner_open_id, subquestions)
            except Exception as e:
                logger.error(f"Chroma检索失败，降级到Mock检索器: {e}")
                self._use_chroma = False
        return await self._mock_search(query, top_k)

    async def _chroma_search(
        self,
        query: str,
        top_k: int,
        knowledge_scope: List[str] = None,
        owner_open_id: str | None = None,
        subquestions: List[str] | None = None,
    ) -> List[SourceItem]:
        """Chroma检索"""
        from app.retrieval.opensearch_store import async_opensearch_search_with_status
        from app.retrieval.vector_store import async_get_document_chunks, async_keyword_search, async_similarity_search

        scopes = self._normalize_scopes(knowledge_scope)
        query_variants = self._build_query_variants(query, subquestions)
        chunk_cache = {}
        neighbor_window = self._neighbor_window_for_query(query)
        if self._is_code_request(query):
            neighbor_window = max(neighbor_window, 2)

        async def run_search_group(query_variant: str, scope: str, metadata_filter: dict | None) -> dict | None:
            ranking_query = self._ranking_query(query_variant, query)
            query_terms = self._extract_query_terms(ranking_query)
            per_filter_k = self._candidate_k(top_k, len(search_plan), len(query_variants))
            keyword_metadata_filter = self._keyword_metadata_filter(scope, metadata_filter)
            filter_results, allow_owner_mismatch = await self._search_with_owner_fallback(
                async_similarity_search,
                query_variant,
                per_filter_k,
                scope,
                metadata_filter,
            )
            opensearch_status = await async_opensearch_search_with_status(
                ranking_query,
                query_terms,
                k=max(per_filter_k, settings.opensearch_lexical_top_k),
                metadata_filter=keyword_metadata_filter,
            )
            opensearch_results = opensearch_status.results
            keyword_results = []
            keyword_allow_owner_mismatch = False
            if opensearch_status.succeeded:
                logger.info(
                    "OpenSearch keyword retrieval succeeded; local fuzzy skipped",
                    extra={
                        "keyword_backend": "opensearch",
                        "local_fuzzy_skipped": True,
                        "opensearch_result_count": len(opensearch_results),
                    },
                )
            else:
                fallback_reason = "opensearch_unavailable" if not opensearch_status.available else "opensearch_error"
                logger.info(
                    "OpenSearch keyword retrieval unavailable; using local fuzzy fallback",
                    extra={
                        "keyword_backend": "local_fuzzy",
                        "local_fuzzy_skipped": False,
                        "fallback_reason": fallback_reason,
                        "opensearch_error": opensearch_status.error,
                    },
                )
                if scope == "personal" and filter_results and metadata_filter:
                    keyword_results = await async_keyword_search(
                        query_terms,
                        k=per_filter_k,
                        metadata_filter=keyword_metadata_filter,
                    )
                else:
                    keyword_results, keyword_allow_owner_mismatch = await self._search_with_owner_fallback(
                        async_keyword_search,
                        query_terms,
                        per_filter_k,
                        scope,
                        keyword_metadata_filter,
                    )
                allow_owner_mismatch = allow_owner_mismatch or keyword_allow_owner_mismatch
            filter_results = self._merge_hybrid_results(filter_results, keyword_results, opensearch_results)
            sources = []

            for doc, score in filter_results:
                metadata = doc.metadata if isinstance(doc.metadata, dict) else {}
                if not self._source_allowed(metadata, [scope], owner_open_id, allow_owner_mismatch):
                    continue
                content = await self._expand_neighbor_content(
                    doc.page_content,
                    metadata,
                    chunk_cache,
                    async_get_document_chunks,
                    neighbor_window,
                )
                sources.append(SourceItem(
                    title=metadata.get("source", "未知来源"),
                    url="",
                    source_type="knowledge_base",
                    snippet=self._build_snippet(doc.page_content, ranking_query),
                    content=content,
                    score=self._normalize_distance_score(score),
                    metadata=metadata,
                ))

            deduped_sources = self._dedupe_sources(sources)
            if self._is_code_request(ranking_query):
                target_sources = [
                    source for source in deduped_sources
                    if self._has_target_evidence(ranking_query, source)
                ]
                if target_sources:
                    deduped_sources = target_sources

            return {
                "query": ranking_query,
                "scope": scope,
                "sources": self._rank_sources_for_query(ranking_query, deduped_sources),
            }

        search_plan = self._build_search_plan(scopes, owner_open_id)
        if not search_plan:
            return []
        tasks = [
            run_search_group(query_variant, scope, metadata_filter)
            for query_variant in query_variants
            for scope, metadata_filter in search_plan
        ]
        logger.info(
            "Knowledge base retrieval planned",
            extra={
                "parallel_retrieval": len(tasks) > 1,
                "subquestion_count": len(query_variants),
                "search_task_count": len(tasks),
            },
        )
        task_results = await asyncio.gather(*tasks, return_exceptions=True)
        search_groups = []
        for result in task_results:
            if isinstance(result, Exception):
                logger.warning(
                    "Knowledge base subquestion retrieval failed",
                    extra={"parallel_retrieval": True, "error": str(result)},
                )
                continue
            if result is not None:
                search_groups.append(result)

        if not search_groups:
            return []
        if len(search_groups) == 1:
            return search_groups[0]["sources"][:top_k]
        return self._merge_scoped_sources(search_groups, top_k)

    def _normalize_scopes(self, knowledge_scope: List[str] = None) -> list[str]:
        scopes = knowledge_scope or ["default"]
        if "default" in scopes:
            scopes = ["enterprise", "personal"]
        return scopes

    def _build_search_plan(self, scopes: List[str], owner_open_id: str | None = None) -> list[tuple[str, dict | None]]:
        plan = []
        if "enterprise" in scopes:
            plan.append(("enterprise", None))
        if "personal" in scopes and owner_open_id:
            plan.append(("personal", {"owner_open_id": owner_open_id}))
        return plan

    def _keyword_metadata_filter(self, scope: str, metadata_filter: dict | None) -> dict | None:
        if scope == "enterprise":
            return {"knowledge_base_type": "enterprise", "tenant_id": settings.default_tenant_id}
        if metadata_filter:
            return {**metadata_filter, "tenant_id": settings.default_tenant_id}
        return metadata_filter

    def _ranking_query(self, query_variant: str, original_query: str) -> str:
        if query_variant != original_query and self._is_code_request(original_query) and not self._is_code_request(query_variant):
            return f"{query_variant} python 代码片段"
        return query_variant

    def _build_query_variants(self, query: str, subquestions: List[str] | None = None) -> list[str]:
        """组合问题优先拆分引号内目标，避免单个长 query 被第一个目标主导。"""
        variants = []
        for candidate in subquestions or []:
            candidate = (candidate or "").strip()
            if 2 <= len(candidate) <= 200 and candidate not in variants:
                variants.append(candidate)

        patterns = [
            r"“([^”]+)”",
            r'"([^"]+)"',
            r"‘([^’]+)’",
            r"'([^']+)'",
        ]
        for pattern in patterns:
            for match in re.finditer(pattern, query):
                candidate = match.group(1).strip()
                if 2 <= len(candidate) <= 200 and candidate not in variants:
                    variants.append(candidate)

        if not variants and self._should_rule_split_query(query):
            for candidate in re.split(r"[、；;]+", query):
                candidate = self._clean_split_candidate(candidate)
                if 2 <= len(candidate) <= 200 and candidate not in variants:
                    variants.append(candidate)

        if not variants:
            return [query]

        if query not in variants:
            variants.append(query)
        return variants[:max(int(settings.rag_parallel_subquestion_max), 1) + 1]

    def _should_rule_split_query(self, query: str) -> bool:
        text = query or ""
        intent_words = ["查找", "检索", "找出", "比较", "总结", "列出", "相关文本", "代码片段"]
        return any(word in text for word in intent_words) and any(separator in text for separator in ["、", "；", ";"])

    def _clean_split_candidate(self, text: str) -> str:
        candidate = (text or "").strip()
        candidate = re.sub(r"^(请|帮我|查询|查找|检索|找出|列出|比较|总结)[：:，,\s]*", "", candidate)
        candidate = re.sub(r"(的)?(相关文本|相关内容|代码片段)$", "", candidate).strip()
        generic_phrases = [
            "企业知识库",
            "个人知识库",
            "企业和个人知识库结合",
            "企业与个人知识库结合",
            "知识库结合",
        ]
        for phrase in generic_phrases:
            candidate = candidate.replace(phrase, "").strip(" ，,")
        return candidate

    def _candidate_k(self, top_k: int, plan_count: int, query_count: int) -> int:
        """扩大候选集，给后续词面重排留出空间。"""
        if plan_count * query_count == 1:
            return min(settings.global_max_chunks, max(top_k, 30))
        return min(settings.global_max_chunks, max(top_k * 10, 50))

    async def _search_with_owner_fallback(
        self,
        similarity_search,
        query: str,
        k: int,
        scope: str,
        metadata_filter: dict | None,
    ) -> tuple[list, bool]:
        results = await similarity_search(query, k=k, metadata_filter=metadata_filter)
        results = sorted(results, key=lambda item: self._safe_float(item[1]))
        if (
            scope != "personal"
            or results
            or settings.personal_kb_strict_owner_filter
        ):
            return results, False

        fallback_results = await similarity_search(
            query,
            k=k,
            metadata_filter={"knowledge_base_type": "personal"},
        )
        fallback_results = sorted(fallback_results, key=lambda item: self._safe_float(item[1]))
        if fallback_results:
            logger.info("个人知识库按 owner 未命中，已降级为个人库宽松检索")
        return fallback_results, True

    def _merge_search_results(self, primary_results: list, supplement_results: list) -> list:
        merged = []
        positions = {}
        for doc, score in list(primary_results or []) + list(supplement_results or []):
            metadata = doc.metadata if isinstance(getattr(doc, "metadata", None), dict) else {}
            key = (
                metadata.get("document_id"),
                metadata.get("chunk_index"),
                metadata.get("source"),
                doc.page_content[:80],
            )
            if key in positions:
                index = positions[key]
                if self._safe_float(score) < self._safe_float(merged[index][1]):
                    merged[index] = (doc, score)
                continue
            positions[key] = len(merged)
            merged.append((doc, score))
        return merged

    def _merge_hybrid_results(self, *ranked_result_lists: list) -> list:
        combined = {}
        rrf_k = max(self._safe_int(settings.opensearch_rrf_k) or 60, 1)
        for result_list in ranked_result_lists:
            ordered_results = sorted(result_list or [], key=lambda item: self._safe_float(item[1]))
            for rank, (doc, score) in enumerate(ordered_results, start=1):
                metadata = doc.metadata if isinstance(getattr(doc, "metadata", None), dict) else {}
                key = self._result_key(doc, metadata)
                entry = combined.setdefault(key, {
                    "doc": doc,
                    "best_distance": self._safe_float(score),
                    "rrf_score": 0.0,
                })
                if self._safe_float(score) < entry["best_distance"]:
                    entry["doc"] = doc
                    entry["best_distance"] = self._safe_float(score)
                entry["rrf_score"] += 1.0 / (rrf_k + rank)

        merged = sorted(
            combined.values(),
            key=lambda item: (-item["rrf_score"], item["best_distance"]),
        )
        return [
            (item["doc"], max(0.0, 1.0 - min(item["rrf_score"] * rrf_k, 1.0)))
            for item in merged
        ]

    def _result_key(self, doc, metadata: dict) -> tuple:
        return (
            metadata.get("document_id"),
            metadata.get("chunk_index"),
            metadata.get("source"),
            doc.page_content[:80],
        )

    def _build_metadata_filters(self, scopes: List[str], owner_open_id: str | None = None) -> list[dict | None]:
        if "personal" in scopes and "enterprise" in scopes:
            return [
                None,
                {"owner_open_id": owner_open_id or ""},
            ]
        if "personal" in scopes:
            return [{"owner_open_id": owner_open_id or ""}]
        return [None]

    def _source_allowed(
        self,
        metadata: dict,
        scopes: List[str],
        owner_open_id: str | None = None,
        allow_owner_mismatch: bool = False,
    ) -> bool:
        kb_type = metadata.get("knowledge_base_type") or "enterprise"
        if kb_type == "personal":
            if allow_owner_mismatch:
                return "personal" in scopes
            owner = owner_open_id or ""
            metadata_owner = metadata.get("owner_user_id") or metadata.get("owner_open_id") or ""
            return "personal" in scopes and metadata_owner == owner
        return "enterprise" in scopes

    async def _expand_neighbor_content(
        self,
        page_content: str,
        metadata: dict,
        chunk_cache: dict,
        get_document_chunks,
        neighbor_window: int,
    ) -> str:
        """补充同文档相邻切块，避免标题命中但正文证据落在下一块。"""
        document_id = metadata.get("document_id")
        chunk_index = self._safe_int(metadata.get("chunk_index"))
        if not document_id or chunk_index is None or neighbor_window <= 0:
            return page_content

        try:
            if document_id not in chunk_cache:
                chunk_cache[document_id] = await get_document_chunks(document_id)
            chunks = chunk_cache[document_id]
        except Exception as e:
            logger.warning(f"补充相邻切块失败，使用原命中切块: {e}")
            return page_content

        min_index = chunk_index - neighbor_window
        max_index = chunk_index + neighbor_window
        selected = []
        for chunk in chunks:
            index = self._safe_int(chunk.get("metadata", {}).get("chunk_index"))
            if index is not None and min_index <= index <= max_index:
                selected.append(chunk.get("content", ""))
        selected = [item for item in selected if item]
        return "\n\n".join(selected) if selected else page_content

    def _neighbor_window_for_query(self, query: str) -> int:
        """列表/汇总类问题通常跨多个切块，需要比普通问答更多相邻上下文。"""
        global_keywords = ["主要内容", "时代演变", "历史影响", "有哪些", "列出", "简介", "概括"]
        if any(keyword in query for keyword in global_keywords):
            return settings.rag_global_neighbor_window
        return settings.rag_neighbor_window

    async def _mock_search(self, query: str, top_k: int) -> List[SourceItem]:
        """模拟检索"""
        await asyncio.sleep(0.05)
        return [
            SourceItem(
                title=f"文档片段 {i+1}",
                url="",
                source_type="knowledge_base",
                snippet=f"这是关于'{query}'的模拟检索结果 {i+1}",
                score=0.9 - i * 0.1,
                metadata={"page": i + 1},
            )
            for i in range(min(top_k, 3))
        ]

    def _rank_sources_for_query(self, query: str, sources: List[SourceItem]) -> List[SourceItem]:
        return sorted(
            sources,
            key=lambda source: (self._lexical_relevance(query, source), source.score),
            reverse=True,
        )

    def _lexical_relevance(self, query: str, source: SourceItem) -> float:
        from app.retrieval.vector_store import expand_search_terms, normalize_search_text

        snippet_text = f"{source.title}\n{source.snippet}".lower()
        content_text = (source.content or "").lower()
        terms = self._extract_query_terms(query)
        expanded_terms = expand_search_terms(terms)
        score = 0.0
        query_text = query.strip().lower()
        normalized_snippet = normalize_search_text(snippet_text)
        normalized_content = normalize_search_text(content_text)
        normalized_query = normalize_search_text(query_text)
        is_code_request = self._is_code_request(query)

        if query_text and query_text in snippet_text:
            score += 30.0
        elif query_text and query_text in content_text:
            score += 8.0
        if normalized_query and normalized_query in normalized_snippet:
            score += 12.0
        elif normalized_query and normalized_query in normalized_content:
            score += 4.0
        for term in expanded_terms:
            if not term:
                continue
            normalized_term = term.lower()
            if normalized_term in snippet_text:
                score += min(len(term), 20)
            elif normalized_term in content_text:
                score += min(len(term), 20) * 0.25
            else:
                compact_term = normalize_search_text(normalized_term)
                if compact_term and compact_term in normalized_snippet:
                    score += min(len(compact_term), 20) * 0.8
                elif compact_term and compact_term in normalized_content:
                    score += min(len(compact_term), 20) * 0.2

        if is_code_request:
            snippet_code_score = self._code_signal_score(snippet_text)
            content_code_score = self._code_signal_score(content_text)
            score += snippet_code_score * 8.0
            score += min(content_code_score, 6.0)
            if snippet_code_score <= 0 and content_code_score <= 0:
                score -= 6.0
        return score

    def _extract_query_terms(self, query: str) -> list[str]:
        terms = []
        for term in re.findall(r"[A-Za-z_][A-Za-z0-9_]*|[\u4e00-\u9fff]{2,}", query):
            if term not in terms:
                terms.append(term)

        if "排序算法" in query or "快速排序" in query:
            for term in ["排序算法", "快速排序", "quick_sort", "quicksort"]:
                if term not in terms:
                    terms.append(term)
        if "正弦曲线" in query:
            for term in ["正弦曲线", "np.sin", "sin(x)", "ax.plot", "plt.plot"]:
                if term not in terms:
                    terms.append(term)
        if "大模型" in query or "大语言模型" in query:
            for term in ["大模型", "大语言模型", "LLM", "large language model"]:
                if term not in terms:
                    terms.append(term)
            if "训练" in query:
                for term in ["大模型训练", "大语言模型训练", "大语言模型的训练"]:
                    if term not in terms:
                        terms.append(term)
        if "圆点" in query and "动画" in query:
            for term in ["圆点", "坐标", "动画", "曲线", "FuncAnimation", "point", "coord_text"]:
                if term not in terms:
                    terms.append(term)
        focus_terms = [
            "天气",
            "毕业选题",
            "毕设选题",
            "历年",
            "选题",
        ]
        for term in focus_terms:
            if term in query and term not in terms:
                terms.append(term)
        return terms

    def _is_code_request(self, query: str) -> bool:
        text = (query or "").lower()
        code_words = ["python", "代码", "代码片段", "实现", "算法", "函数", "示例代码"]
        return any(word in text for word in code_words)

    def _has_target_evidence(self, query: str, source: SourceItem) -> bool:
        from app.retrieval.vector_store import expand_search_terms, normalize_search_text

        target_terms = self._target_terms(query)
        if not target_terms:
            return True
        text = f"{source.title}\n{source.snippet}\n{source.content}".lower()
        normalized_text = normalize_search_text(text)
        for term in expand_search_terms(target_terms):
            if term.lower() in text:
                return True
            normalized_term = normalize_search_text(term)
            if normalized_term and normalized_term in normalized_text:
                return True
        return False

    def _target_terms(self, query: str) -> list[str]:
        generic_terms = {
            "python",
            "代码",
            "代码片段",
            "示例代码",
            "函数",
            "实现",
            "找出",
            "检索",
            "知识库",
            "个人知识库",
            "企业知识库",
            "企业和个人知识库结合检索",
        }
        terms = []
        for term in self._extract_query_terms(query):
            normalized = term.strip()
            if not normalized or normalized.lower() in generic_terms or normalized in generic_terms:
                continue
            if normalized not in terms:
                terms.append(normalized)
        return terms

    def _code_signal_score(self, text: str) -> float:
        patterns = [
            r"\bdef\s+[A-Za-z_][A-Za-z0-9_]*\s*\(",
            r"\bclass\s+[A-Za-z_][A-Za-z0-9_]*",
            r"\bimport\s+[A-Za-z_][A-Za-z0-9_]*",
            r"\bfrom\s+[A-Za-z_][A-Za-z0-9_.]*\s+import\b",
            r"\breturn\b",
            r"\bfor\s+[A-Za-z_][A-Za-z0-9_]*\s+in\b",
            r"\bif\s+.+:",
            r"\bnp\.",
            r"\bplt\.",
            r"\bax\.",
            r"\bFuncAnimation\b",
        ]
        return float(sum(1 for pattern in patterns if re.search(pattern, text)))

    def _build_snippet(self, text: str, query: str) -> str:
        from app.retrieval.vector_store import expand_search_terms, normalize_search_text

        limit = settings.rag_display_snippet_chars
        if len(text) <= limit:
            return text

        lowered = text.lower()
        positions = []
        for term in expand_search_terms(self._extract_query_terms(query)):
            if not term:
                continue
            index = lowered.find(term.lower())
            if index >= 0:
                positions.append(index)
                continue
            normalized_term = normalize_search_text(term)
            if normalized_term and normalized_term in normalize_search_text(text):
                raw_position = self._approximate_normalized_position(text, normalized_term)
                if raw_position is not None:
                    positions.append(raw_position)
        if self._is_code_request(query):
            for pattern in [r"\bdef\s+", r"\bimport\s+", r"\bfrom\s+", r"\bnp\.", r"\bplt\.", r"\bax\."]:
                match = re.search(pattern, text)
                if match:
                    positions.append(match.start())

        if not positions:
            return text[:limit]

        anchor = min(positions)
        start = max(anchor - limit // 4, 0)
        end = start + limit
        if end > len(text):
            end = len(text)
            start = max(end - limit, 0)
        return text[start:end]

    def _approximate_normalized_position(self, text: str, normalized_term: str) -> int | None:
        from app.retrieval.vector_store import normalize_search_text

        for index in range(len(text)):
            candidate = normalize_search_text(text[index:index + len(normalized_term) + 8])
            if normalized_term in candidate:
                return index
        return None

    def _dedupe_sources(self, sources: List[SourceItem]) -> List[SourceItem]:
        """去重来源"""
        seen = set()
        deduped = []
        for s in sources:
            key = (s.snippet[:50], s.title)
            if key not in seen:
                seen.add(key)
                deduped.append(s)
        return deduped

    def _merge_scoped_sources(self, search_groups: list[dict], top_k: int) -> List[SourceItem]:
        """组合检索先保底每个明确目标的每个知识库类型，再补充剩余结果。"""
        merged = []
        seen = set()

        query_order = []
        for group in search_groups:
            if group["query"] not in query_order:
                query_order.append(group["query"])

        # 先为每个引号目标的每个知识库类型保留一个最佳来源，确保企业知识库结果不丢失。
        for query in query_order:
            for scope in ["enterprise", "personal"]:
                best_source = None
                best_rank = None
                for group in search_groups:
                    if group["query"] != query or group["scope"] != scope:
                        continue
                    for source in group["sources"]:
                        lexical_score = self._lexical_relevance(query, source)
                        if self._is_code_request(query) and (
                            lexical_score <= 0 or not self._has_target_evidence(query, source)
                        ):
                            continue
                        key = (source.snippet[:50], source.title)
                        if key in seen:
                            continue
                        rank = (lexical_score, source.score)
                        if best_rank is None or rank > best_rank:
                            best_source = source
                            best_rank = rank
                if best_source is not None:
                    key = (best_source.snippet[:50], best_source.title)
                    seen.add(key)
                    merged.append(best_source)
                    if len(merged) >= top_k:
                        return merged

        max_len = max((len(group["sources"]) for group in search_groups), default=0)
        for index in range(max_len):
            for group in search_groups:
                sources = group["sources"]
                if index < len(sources):
                    source = sources[index]
                    if self._is_code_request(group["query"]) and (
                        self._lexical_relevance(group["query"], source) <= 0
                        or not self._has_target_evidence(group["query"], source)
                    ):
                        continue
                    key = (source.snippet[:50], source.title)
                    if key in seen:
                        continue
                    seen.add(key)
                    merged.append(source)
                    if len(merged) >= top_k:
                        return merged
        return merged

    def _safe_int(self, value) -> int | None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _safe_float(self, value) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("inf")

    def _normalize_distance_score(self, distance) -> float:
        """将Chroma距离转成稳定的0~1相关度，避免返回负数或超过1。"""
        if distance is None:
            return 0.0
        try:
            safe_distance = max(float(distance), 0.0)
        except (TypeError, ValueError):
            return 0.0
        return round(1.0 / (1.0 + safe_distance), 4)


retriever = Retriever()
