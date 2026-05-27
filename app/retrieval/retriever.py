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
        self.refresh()

    def refresh(self) -> bool:
        """刷新Chroma可用状态。上传成功后调用，确保后续查询优先使用真实向量库。"""
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
        if self._use_chroma or self.refresh():
            try:
                return await self._chroma_search(query, top_k)
            except Exception as e:
                logger.error(f"Chroma检索失败，降级到Mock检索器: {e}")
                self._use_chroma = False
        return await self._mock_search(query, top_k)

    async def _chroma_search(self, query: str, top_k: int) -> List[SourceItem]:
        """Chroma检索"""
        from app.retrieval.vector_store import async_similarity_search

        results = await async_similarity_search(query, k=top_k)
        sources = []

        for doc, score in results:
            sources.append(SourceItem(
                title=doc.metadata.get("source", "未知来源"),
                url="",
                source_type="knowledge_base",
                snippet=doc.page_content[:200],
                score=self._normalize_distance_score(score),
                metadata=doc.metadata,
            ))

        return self._dedupe_sources(sources)

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
