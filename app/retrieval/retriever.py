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
        self.provider = "mock"

    async def search(
        self,
        query: str,
        top_k: int = 5,
        knowledge_scope: List[str] = None,
    ) -> List[SourceItem]:
        """检索"""
        if self.provider == "mock":
            return await self._mock_search(query, top_k)
        else:
            raise ValueError(f"不支持的检索器: {self.provider}")

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


# 全局实例
retriever = Retriever()
