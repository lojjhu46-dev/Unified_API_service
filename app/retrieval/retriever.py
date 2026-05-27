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
        self._check_chroma()

    def _check_chroma(self):
        """检查Chroma是否可用"""
        try:
            import chromadb
            from app.retrieval.vector_store import get_vector_store
            get_vector_store()
            self._use_chroma = True
            logger.info("使用Chroma向量存储")
        except Exception:
            logger.info("使用Mock检索器")

    async def search(
        self,
        query: str,
        top_k: int = 5,
        knowledge_scope: List[str] = None,
    ) -> List[SourceItem]:
        """检索"""
        if self._use_chroma:
            return await self._chroma_search(query, top_k)
        return await self._mock_search(query, top_k)

    async def _chroma_search(self, query: str, top_k: int) -> List[SourceItem]:
        """Chroma检索"""
        try:
            from app.retrieval.vector_store import async_similarity_search

            results = await async_similarity_search(query, k=top_k)
            sources = []

            for doc, score in results:
                sources.append(SourceItem(
                    title=doc.metadata.get("source", "未知来源"),
                    url="",
                    source_type="knowledge_base",
                    snippet=doc.page_content[:200],
                    score=float(1 - score) if score else 0.0,
                    metadata=doc.metadata,
                ))

            return self._dedupe_sources(sources)
        except Exception as e:
            logger.error(f"Chroma检索失败: {e}")
            return []

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


retriever = Retriever()
