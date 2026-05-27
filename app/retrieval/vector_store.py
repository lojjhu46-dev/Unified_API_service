"""向量存储适配器"""

import asyncio
from typing import List, Optional
from functools import lru_cache
from app.config import settings
from app.observability.logging import get_logger

logger = get_logger(__name__)


def _get_embeddings():
    """获取Embedding模型"""
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name=settings.embedding_model_name)
    except Exception as e:
        logger.error(f"加载Embedding模型失败: {e}")
        raise


@lru_cache(maxsize=1)
def _get_cached_embeddings():
    """缓存Embedding模型"""
    return _get_embeddings()


def get_vector_store():
    """获取向量存储实例"""
    try:
        from langchain_community.vectorstores import Chroma
        embeddings = _get_cached_embeddings()
        return Chroma(
            persist_directory=settings.chroma_persist_dir,
            embedding_function=embeddings,
        )
    except Exception as e:
        logger.error(f"获取向量存储失败: {e}")
        raise


def add_documents(documents: list) -> int:
    """添加文档到向量存储"""
    try:
        from langchain_community.vectorstores import Chroma
        embeddings = _get_cached_embeddings()
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=settings.chroma_persist_dir,
        )
        return len(documents)
    except Exception as e:
        logger.error(f"添加文档失败: {e}")
        raise


async def async_similarity_search(query: str, k: int = 5) -> list:
    """异步相似度搜索"""
    return await asyncio.to_thread(_sync_similarity_search, query, k)


def _sync_similarity_search(query: str, k: int) -> list:
    """同步相似度搜索"""
    try:
        vector_store = get_vector_store()
        results = vector_store.similarity_search_with_score(query, k=k)
        return results
    except Exception as e:
        logger.error(f"相似度搜索失败: {e}")
        return []
