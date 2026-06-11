"""OpenSearch keyword index adapter."""

import asyncio
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from app.config import settings
from app.observability.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class OpenSearchSearchResult:
    results: list
    available: bool
    succeeded: bool
    error: str | None = None


def build_index_body(use_ik_analyzer: bool | None = None) -> dict:
    use_ik = settings.opensearch_use_ik_analyzer if use_ik_analyzer is None else use_ik_analyzer
    tokenizer = "ik_max_word" if use_ik else "standard"
    search_tokenizer = "ik_smart" if use_ik else "standard"
    return {
        "settings": {
            "analysis": {
                "filter": {
                    "rag_stopwords": {
                        "type": "stop",
                        "stopwords": [
                            "的",
                            "地",
                            "得",
                            "了",
                            "着",
                            "过",
                            "请",
                            "帮我",
                            "查找",
                            "检索",
                            "关于",
                            "内容",
                            "文本",
                            "相关",
                        ],
                    },
                    "rag_synonyms": {
                        "type": "synonym_graph",
                        "synonyms": [
                            "大模型, 大语言模型, LLM, large language model",
                            "大模型训练, 大语言模型训练, 大语言模型的训练",
                            "毕设, 毕业设计, 毕设选题, 毕业选题",
                            "排序算法, 快速排序, quick_sort, quicksort",
                            "正弦曲线, sin(x), np.sin",
                        ],
                    },
                },
                "analyzer": {
                    "rag_index_analyzer": {
                        "type": "custom",
                        "tokenizer": tokenizer,
                        "filter": ["lowercase"],
                    },
                    "rag_search_analyzer": {
                        "type": "custom",
                        "tokenizer": search_tokenizer,
                        "filter": ["lowercase", "rag_synonyms", "rag_stopwords"],
                    },
                },
            },
        },
        "mappings": {
            "properties": {
                "content": {
                    "type": "text",
                    "analyzer": "rag_index_analyzer",
                    "search_analyzer": "rag_search_analyzer",
                    "fields": {"raw": {"type": "keyword", "ignore_above": 32766}},
                },
                "original_filename": {
                    "type": "text",
                    "analyzer": "rag_index_analyzer",
                    "search_analyzer": "rag_search_analyzer",
                    "fields": {"keyword": {"type": "keyword"}},
                },
                "document_id": {"type": "keyword"},
                "chunk_id": {"type": "keyword"},
                "tenant_id": {"type": "keyword"},
                "stored_filename": {"type": "keyword"},
                "chunk_index": {"type": "integer"},
                "knowledge_base_type": {"type": "keyword"},
                "owner_user_id": {"type": "keyword"},
                "owner_open_id": {"type": "keyword"},
                "chat_id": {"type": "keyword"},
                "channel": {"type": "keyword"},
                "source": {"type": "keyword"},
                "sheet": {"type": "keyword"},
                "page": {"type": "keyword"},
            },
        },
    }


@lru_cache(maxsize=1)
def get_opensearch_client():
    if not settings.opensearch_enabled:
        return None
    try:
        from opensearchpy import OpenSearch
    except Exception as e:
        logger.warning(f"OpenSearch client unavailable: {e}")
        return None

    http_auth = None
    if settings.opensearch_username or settings.opensearch_password:
        http_auth = (settings.opensearch_username or "", settings.opensearch_password or "")
    return OpenSearch(
        hosts=[settings.opensearch_url],
        http_auth=http_auth,
        use_ssl=settings.opensearch_use_ssl,
        verify_certs=settings.opensearch_verify_certs,
        timeout=settings.opensearch_timeout_seconds,
    )


def refresh_opensearch_client() -> None:
    get_opensearch_client.cache_clear()


def ensure_index() -> bool:
    client = get_opensearch_client()
    if client is None:
        return False
    index_name = settings.opensearch_index_name
    try:
        if client.indices.exists(index=index_name):
            return True
        client.indices.create(index=index_name, body=build_index_body())
        return True
    except Exception as e:
        logger.warning(f"OpenSearch index init failed: {e}")
        return False


def index_documents(documents: list) -> int:
    if not documents or not ensure_index():
        return 0
    try:
        from opensearchpy.helpers import bulk

        actions = []
        for document in documents:
            metadata = document.metadata if isinstance(getattr(document, "metadata", None), dict) else {}
            document_id = metadata.get("document_id")
            chunk_index = metadata.get("chunk_index")
            if document_id is None or chunk_index is None:
                continue
            actions.append({
                "_op_type": "index",
                "_index": settings.opensearch_index_name,
                "_id": f"{document_id}:{chunk_index}",
                "_source": _document_source(document.page_content or "", metadata),
            })
        if not actions:
            return 0
        success_count, _ = bulk(get_opensearch_client(), actions, raise_on_error=False)
        return int(success_count)
    except Exception as e:
        logger.warning(f"OpenSearch bulk index failed: {e}")
        return 0


async def async_opensearch_search(
    query: str,
    terms: list[str],
    k: int = 5,
    metadata_filter: dict | None = None,
) -> list:
    status = await async_opensearch_search_with_status(query, terms, k, metadata_filter)
    return status.results


async def async_opensearch_search_with_status(
    query: str,
    terms: list[str],
    k: int = 5,
    metadata_filter: dict | None = None,
) -> OpenSearchSearchResult:
    return await asyncio.to_thread(_sync_opensearch_search_with_status, query, terms, k, metadata_filter)


def _sync_opensearch_search(
    query: str,
    terms: list[str],
    k: int,
    metadata_filter: dict | None = None,
) -> list:
    return _sync_opensearch_search_with_status(query, terms, k, metadata_filter).results


def _sync_opensearch_search_with_status(
    query: str,
    terms: list[str],
    k: int,
    metadata_filter: dict | None = None,
) -> OpenSearchSearchResult:
    client = get_opensearch_client()
    if client is None:
        return OpenSearchSearchResult(
            results=[],
            available=False,
            succeeded=False,
            error="opensearch_client_unavailable",
        )
    try:
        from langchain_core.documents import Document

        body = {
            "size": k,
            "track_total_hits": False,
            "query": _build_query(query, terms, metadata_filter),
            "_source": True,
        }
        response = client.search(index=settings.opensearch_index_name, body=body)
        hits = response.get("hits", {}).get("hits", [])
        results = []
        for hit in hits:
            source = hit.get("_source") or {}
            content = source.get("content") or ""
            metadata = _source_metadata(source)
            results.append((Document(page_content=content, metadata=metadata), _score_to_distance(hit.get("_score"))))
        return OpenSearchSearchResult(
            results=results,
            available=True,
            succeeded=True,
        )
    except Exception as e:
        logger.warning(
            f"OpenSearch search failed: {type(e).__name__}: {e}",
            exc_info=True,
            extra={"query": query, "metadata_filter": metadata_filter},
        )
        return OpenSearchSearchResult(
            results=[],
            available=True,
            succeeded=False,
            error=str(e),
        )


def _build_query(query: str, terms: list[str], metadata_filter: dict | None) -> dict:
    from app.retrieval.vector_store import expand_search_terms, normalize_search_text

    should = []
    cleaned_query = (query or "").strip()
    if cleaned_query:
        should.extend([
            {"match_phrase": {"content": {"query": cleaned_query, "boost": 6}}},
            {"match": {"content": {"query": cleaned_query, "operator": "and", "boost": 3}}},
            {"multi_match": {
                "query": cleaned_query,
                "fields": ["content^2", "original_filename"],
                "type": "best_fields",
                "boost": 1.5,
            }},
        ])
        normalized_query = normalize_search_text(cleaned_query)
        if normalized_query and normalized_query != cleaned_query:
            should.append({"match_phrase": {"content": {"query": normalized_query, "boost": 4}}})

    for term in expand_search_terms(terms)[:20]:
        if not term:
            continue
        should.append({"match_phrase": {"content": {"query": term, "boost": 4}}})
        should.append({"match": {"content": {"query": term, "boost": 1.2}}})

    filters = _metadata_filters(metadata_filter)
    return {
        "bool": {
            "filter": filters,
            "should": should or [{"match_all": {}}],
            "minimum_should_match": 1 if should else 0,
        },
    }


def _metadata_filters(metadata_filter: dict | None) -> list[dict]:
    if not metadata_filter:
        return []
    tenant_id = metadata_filter.get("tenant_id") or settings.default_tenant_id
    if metadata_filter.get("knowledge_base_type") == "enterprise":
        return [
            {"term": {"knowledge_base_type": "enterprise"}},
            _tenant_filter(tenant_id),
        ]
    if "owner_open_id" in metadata_filter or "owner_user_id" in metadata_filter:
        owner = metadata_filter.get("owner_user_id") or metadata_filter.get("owner_open_id") or ""
        return [
            {"term": {"knowledge_base_type": "personal"}},
            _tenant_filter(tenant_id),
            {
                "bool": {
                    "should": [
                        {"term": {"owner_user_id": owner}},
                        {"term": {"owner_open_id": owner}},
                    ],
                    "minimum_should_match": 1,
                },
            },
        ]
    return [{"term": {key: value}} for key, value in metadata_filter.items()]


def _tenant_filter(tenant_id: str) -> dict:
    return {
        "bool": {
            "should": [
                {"term": {"tenant_id": tenant_id}},
                {"bool": {"must_not": [{"exists": {"field": "tenant_id"}}]}},
            ],
            "minimum_should_match": 1,
        }
    }


def _document_source(content: str, metadata: dict) -> dict[str, Any]:
    return {
        "content": content,
        "tenant_id": str(metadata.get("tenant_id") or settings.default_tenant_id),
        "document_id": str(metadata.get("document_id") or ""),
        "chunk_id": str(metadata.get("chunk_id") or ""),
        "chunk_index": _safe_int(metadata.get("chunk_index")),
        "original_filename": str(metadata.get("original_filename") or metadata.get("source") or ""),
        "stored_filename": str(metadata.get("stored_filename") or ""),
        "knowledge_base_type": str(metadata.get("knowledge_base_type") or "enterprise"),
        "owner_user_id": str(metadata.get("owner_user_id") or ""),
        "owner_open_id": str(metadata.get("owner_open_id") or ""),
        "chat_id": str(metadata.get("chat_id") or ""),
        "channel": str(metadata.get("channel") or ""),
        "source": str(metadata.get("source") or ""),
        "sheet": str(metadata.get("sheet") or ""),
        "page": str(metadata.get("page") or ""),
    }


def _source_metadata(source: dict) -> dict:
    metadata = {
        "tenant_id": source.get("tenant_id") or settings.default_tenant_id,
        "document_id": source.get("document_id") or "",
        "chunk_id": source.get("chunk_id") or "",
        "chunk_index": source.get("chunk_index", 0),
        "original_filename": source.get("original_filename") or "",
        "stored_filename": source.get("stored_filename") or "",
        "knowledge_base_type": source.get("knowledge_base_type") or "enterprise",
        "owner_user_id": source.get("owner_user_id") or "",
        "owner_open_id": source.get("owner_open_id") or "",
        "chat_id": source.get("chat_id") or "",
        "channel": source.get("channel") or "",
        "source": source.get("source") or source.get("original_filename") or "",
    }
    if source.get("sheet"):
        metadata["sheet"] = source["sheet"]
    if source.get("page"):
        metadata["page"] = source["page"]
    return metadata


def _score_to_distance(score) -> float:
    try:
        safe_score = max(float(score), 0.0)
    except (TypeError, ValueError):
        safe_score = 0.0
    return 1.0 / (1.0 + safe_score)


def _safe_int(value) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
