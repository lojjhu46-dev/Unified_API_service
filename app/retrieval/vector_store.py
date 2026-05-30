"""向量存储适配器"""

import asyncio
import re
from functools import lru_cache
from app.config import settings
from app.observability.logging import get_logger

logger = get_logger(__name__)

_STOP_PHRASES = [
    "帮我",
    "请帮我",
    "麻烦",
    "查找",
    "检索",
    "搜索",
    "关于",
    "文本",
    "内容",
    "相关",
    "一下",
]
_STOP_CHARS = set("的地得了着过吗呢啊吧呀嘛么")


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

        search_terms = expand_search_terms(terms)
        if not search_terms:
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
            normalized_search_text = normalize_search_text(search_text)
            best_distance = None
            hit_terms = []
            for term in search_terms:
                match_score = _keyword_match_score(term, search_text, normalized_search_text)
                if match_score is None:
                    continue
                hit_terms.append(term)
                best_distance = match_score if best_distance is None else min(best_distance, match_score)
            if best_distance is None:
                continue
            distance = best_distance - sum(min(len(term), 20) * 0.01 for term in set(hit_terms))
            matched.append((
                Document(page_content=safe_content, metadata=safe_metadata),
                distance,
            ))

        matched.sort(key=lambda item: item[1])
        return matched[:k]
    except Exception as e:
        logger.error(f"关键词搜索失败: {e}")
        raise


def normalize_search_text(text: str) -> str:
    normalized = (text or "").lower()
    for phrase in _STOP_PHRASES:
        normalized = normalized.replace(phrase, "")
    normalized = re.sub(r"[^\w\u4e00-\u9fff]+", "", normalized)
    normalized = "".join(ch for ch in normalized if ch not in _STOP_CHARS)
    return normalized


def expand_search_terms(terms: list[str]) -> list[str]:
    expanded = []
    for term in terms or []:
        raw = str(term).strip()
        if not raw:
            continue
        candidates = [raw, normalize_search_text(raw)]
        candidates.extend(_ai_phrase_expansions(raw))
        for candidate in candidates:
            candidate = str(candidate).strip()
            if len(candidate) >= 2 and candidate not in expanded:
                expanded.append(candidate)
    return expanded


def _ai_phrase_expansions(term: str) -> list[str]:
    text = term or ""
    normalized = normalize_search_text(text)
    expansions = []
    has_big_model = "大模型" in text or "大模型" in normalized
    has_llm_cn = "大语言模型" in text or "大语言模型" in normalized
    has_training = "训练" in text or "训练" in normalized

    if has_big_model:
        expansions.extend(["大语言模型", "LLM", "large language model"])
    if has_llm_cn:
        expansions.extend(["大模型", "LLM", "large language model"])
    if has_training and (has_big_model or has_llm_cn):
        expansions.extend([
            "大模型训练",
            "大语言模型训练",
            "大语言模型的训练",
        ])
    return expansions


def _keyword_match_score(term: str, search_text: str, normalized_search_text: str) -> float | None:
    raw_term = (term or "").lower()
    normalized_term = normalize_search_text(raw_term)
    if not raw_term or not normalized_term:
        return None
    if raw_term in search_text:
        return -100.0 - min(len(raw_term), 20)
    if normalized_term in normalized_search_text:
        return -80.0 - min(len(normalized_term), 20)

    similarity = _best_fuzzy_similarity(normalized_term, normalized_search_text)
    if similarity >= _fuzzy_threshold(normalized_term):
        return -40.0 - similarity
    return None


def _fuzzy_threshold(term: str) -> float:
    length = len(term or "")
    if length < 4:
        return 1.1
    if length <= 8:
        return 0.82
    return 0.78


def _best_fuzzy_similarity(term: str, text: str) -> float:
    term = term or ""
    text = text or ""
    length = len(term)
    if length < 4 or not text:
        return 0.0
    if term in text:
        return 1.0

    best = 0.0
    min_window = max(4, length - 2)
    max_window = min(len(text), length + 2)
    for window_size in range(min_window, max_window + 1):
        for start in range(0, len(text) - window_size + 1):
            window = text[start:start + window_size]
            distance = _levenshtein_distance(term, window)
            similarity = 1.0 - distance / max(length, window_size)
            if similarity > best:
                best = similarity
                if best >= 1.0:
                    return best
    return best


def _levenshtein_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        for j, right_char in enumerate(right, start=1):
            cost = 0 if left_char == right_char else 1
            current.append(min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + cost,
            ))
        previous = current
    return previous[-1]


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
