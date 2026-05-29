"""检索器"""

import asyncio
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
    ) -> List[SourceItem]:
        """检索"""
        if self._use_chroma or (not self._chroma_checked and self.refresh()):
            try:
                return await self._chroma_search(query, top_k)
            except Exception as e:
                logger.error(f"Chroma检索失败，降级到Mock检索器: {e}")
                self._use_chroma = False
        return await self._mock_search(query, top_k)

    async def _chroma_search(self, query: str, top_k: int) -> List[SourceItem]:
        """Chroma检索"""
        from app.retrieval.vector_store import async_get_document_chunks, async_similarity_search

        results = await async_similarity_search(query, k=top_k)
        sources = []
        chunk_cache = {}
        neighbor_window = self._neighbor_window_for_query(query)

        for doc, score in results:
            metadata = doc.metadata if isinstance(doc.metadata, dict) else {}
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
                snippet=doc.page_content[:settings.rag_display_snippet_chars],
                content=content,
                score=self._normalize_distance_score(score),
                metadata=metadata,
            ))

        return self._dedupe_sources(sources)

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

    def _safe_int(self, value) -> int | None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

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
