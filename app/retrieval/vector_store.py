"""向量存储适配器"""

import asyncio
from functools import lru_cache
from app.config import settings
from app.observability.logging import get_logger

logger = get_logger(__name__)


def _get_embeddings():
    """获取Embedding模型"""
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name=settings.embedding_model_name,
            model_kwargs={"local_files_only": True},
        )
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


async def async_similarity_search(query: str, k: int = 5, metadata_filter: dict | None = None) -> list:
    """异步相似度搜索"""
    return await asyncio.to_thread(_sync_similarity_search, query, k, metadata_filter)


async def async_get_document_chunks(document_id: str) -> list[dict]:
    """按文档ID读取已入库切块，用于补充命中切块的相邻上下文。"""
    return await asyncio.to_thread(_sync_get_document_chunks, document_id)


async def async_keyword_search(terms: list[str], k: int = 5, metadata_filter: dict | None = None) -> list:
    """按关键词精确匹配已入库切块，补充向量召回遗漏。"""
    return await asyncio.to_thread(_sync_keyword_search, terms, k, metadata_filter)


def _sync_similarity_search(query: str, k: int, metadata_filter: dict | None = None) -> list:
    """同步相似度搜索"""
    try:
        vector_store = get_vector_store()
        kwargs = {"filter": metadata_filter} if metadata_filter else {}
        results = vector_store.similarity_search_with_score(query, k=k, **kwargs)
        return results
    except Exception as e:
        logger.error(f"相似度搜索失败: {e}")
        raise


def _sync_keyword_search(terms: list[str], k: int, metadata_filter: dict | None = None) -> list:
    """同步关键词搜索。"""
    try:
        from langchain_core.documents import Document

        normalized_terms = []
        for term in terms or []:
            normalized = str(term).strip().lower()
            if len(normalized) >= 2 and normalized not in normalized_terms:
                normalized_terms.append(normalized)
        if not normalized_terms:
            return []

        vector_store = get_vector_store()
        kwargs = {"where": metadata_filter} if metadata_filter and metadata_filter != {"knowledge_base_type": "enterprise"} else {}
        result = vector_store.get(include=["documents", "metadatas"], **kwargs)
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []
        matched = []
        for content, metadata in zip(documents, metadatas):
            safe_content = content or ""
            safe_metadata = metadata if isinstance(metadata, dict) else {}
            if not _metadata_matches_keyword_filter(safe_metadata, metadata_filter):
                continue
            search_text = f"{safe_metadata.get('original_filename', '')}\n{safe_content}".lower()
            hit_terms = [term for term in normalized_terms if term in search_text]
            if not hit_terms:
                continue
            distance = -sum(min(len(term), 20) for term in set(hit_terms))
            matched.append((
                Document(page_content=safe_content, metadata=safe_metadata),
                distance,
            ))

        matched.sort(key=lambda item: item[1])
        return matched[:k]
    except Exception as e:
        logger.error(f"关键词搜索失败: {e}")
        raise


def _metadata_matches_keyword_filter(metadata: dict, metadata_filter: dict | None) -> bool:
    if not metadata_filter:
        return True
    if metadata_filter == {"knowledge_base_type": "enterprise"}:
        return (metadata.get("knowledge_base_type") or "enterprise") == "enterprise"
    for key, expected in metadata_filter.items():
        if metadata.get(key) != expected:
            return False
    return True


def _sync_get_document_chunks(document_id: str) -> list[dict]:
    """同步读取同一文档的所有切块。"""
    try:
        vector_store = get_vector_store()
        result = vector_store.get(
            where={"document_id": document_id},
            include=["documents", "metadatas"],
        )
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []
        chunks = []
        for content, metadata in zip(documents, metadatas):
            chunks.append({
                "content": content or "",
                "metadata": metadata if isinstance(metadata, dict) else {},
            })
        return sorted(chunks, key=lambda item: _safe_chunk_index(item["metadata"]))
    except Exception as e:
        logger.error(f"读取文档切块失败: {e}")
        raise


def _safe_chunk_index(metadata: dict) -> int:
    try:
        return int(metadata.get("chunk_index", 0))
    except (TypeError, ValueError):
        return 0
