"""联网搜索工具"""

import asyncio
import re
import httpx
from typing import List
from app.config import settings
from app.observability.logging import get_logger

logger = get_logger(__name__)


DOMAIN_PATTERN = re.compile(r"^(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}$")


def _normalize_domain(domain: str) -> str | None:
    value = (domain or "").strip().lower()
    value = re.sub(r"^https?://", "", value)
    value = value.split("/", 1)[0].split("?", 1)[0].split("#", 1)[0]
    value = value.strip(" .，,。")
    if value.startswith("www."):
        value = value[4:]
    if not value or not DOMAIN_PATTERN.match(value):
        return None
    return value


def _normalize_domains(domains: list[str] | tuple[str, ...] | str | None) -> list[str]:
    if not domains:
        return []
    raw_domains = [domains] if isinstance(domains, str) else list(domains)
    normalized = []
    for item in raw_domains:
        domain = _normalize_domain(str(item))
        if domain and domain not in normalized:
            normalized.append(domain)
    return normalized


def _build_effective_query(query: str, domains: list[str]) -> str:
    cleaned = (query or "").strip()
    if not domains:
        return cleaned
    site_filter = " OR ".join(f"site:{domain}" for domain in domains)
    return f"{cleaned} ({site_filter})"


def _format_structured_result(data: dict) -> List[dict]:
    """格式化结构化答案"""
    results = []

    answer_box = data.get("answerBox")
    if isinstance(answer_box, dict):
        answer = answer_box.get("answer") or answer_box.get("snippet")
        title = answer_box.get("title", "结构化答案")
        link = answer_box.get("link", "")
        if answer:
            results.append({
                "title": title,
                "url": link,
                "snippet": answer,
            })

    knowledge_graph = data.get("knowledgeGraph")
    if isinstance(knowledge_graph, dict):
        title = knowledge_graph.get("title", "知识图谱")
        description = knowledge_graph.get("description")
        if description:
            results.append({
                "title": title,
                "url": "",
                "snippet": description,
            })

    return results


def _format_organic_results(organic: list) -> List[dict]:
    """格式化普通搜索结果"""
    results = []
    for item in organic[:5]:
        results.append({
            "title": item.get("title", "无标题"),
            "url": item.get("link", ""),
            "snippet": item.get("snippet", "无摘要"),
        })
    return results


async def web_search(query: str, domains: list[str] | tuple[str, ...] | str | None = None) -> dict:
    """执行联网搜索"""
    normalized_domains = _normalize_domains(domains)
    effective_query = _build_effective_query(query, normalized_domains)

    if not settings.serper_api_key:
        return {
            "success": False,
            "error": "未配置SERPER_API_KEY",
            "results": [],
            "query": query,
            "effective_query": effective_query,
            "domains": normalized_domains,
        }

    headers = {
        "X-API-KEY": settings.serper_api_key,
        "Content-Type": "application/json",
    }
    payload = {"q": effective_query, "gl": "cn", "hl": "zh-cn"}

    for attempt in range(settings.max_retries + 1):
        try:
            async with httpx.AsyncClient(timeout=settings.search_timeout) as client:
                resp = await client.post(settings.serper_url, json=payload, headers=headers)

                if resp.status_code == 429:
                    if attempt < settings.max_retries:
                        wait_time = 2 ** attempt
                        await asyncio.sleep(wait_time)
                        continue
                    return {
                        "success": False,
                        "error": "搜索请求过于频繁（HTTP 429），请稍后重试",
                        "results": [],
                        "query": query,
                        "effective_query": effective_query,
                        "domains": normalized_domains,
                    }

                if resp.status_code == 401:
                    return {
                        "success": False,
                        "error": "搜索服务认证失败（HTTP 401），SERPER_API_KEY 无效",
                        "results": [],
                        "query": query,
                        "effective_query": effective_query,
                        "domains": normalized_domains,
                    }

                resp.raise_for_status()
                data = resp.json()

                results = []
                results.extend(_format_structured_result(data))
                results.extend(_format_organic_results(data.get("organic", [])))

                if not results:
                    return {
                        "success": True,
                        "error": None,
                        "results": [],
                        "message": f"针对「{effective_query}」的搜索未找到相关结果",
                        "query": query,
                        "effective_query": effective_query,
                        "domains": normalized_domains,
                    }

                return {
                    "success": True,
                    "error": None,
                    "results": results,
                    "query": query,
                    "effective_query": effective_query,
                    "domains": normalized_domains,
                }

        except httpx.TimeoutException:
            if attempt < settings.max_retries:
                continue
            return {
                "success": False,
                "error": f"搜索请求超时（等待了 {settings.search_timeout} 秒）",
                "results": [],
                "query": query,
                "effective_query": effective_query,
                "domains": normalized_domains,
            }
        except httpx.ConnectError:
            return {
                "success": False,
                "error": "无法连接搜索服务，请检查网络连接",
                "results": [],
                "query": query,
                "effective_query": effective_query,
                "domains": normalized_domains,
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"搜索请求异常: {type(e).__name__}",
                "results": [],
                "query": query,
                "effective_query": effective_query,
                "domains": normalized_domains,
            }

    return {
        "success": False,
        "error": f"搜索失败，已重试 {settings.max_retries} 次",
        "results": [],
        "query": query,
        "effective_query": effective_query,
        "domains": normalized_domains,
    }


def format_search_results(results: List[dict]) -> str:
    """格式化搜索结果为文本"""
    lines = []
    for i, item in enumerate(results[:5], 1):
        title = item.get("title", "无标题")
        snippet = item.get("snippet", "无摘要")
        link = item.get("url", "")
        if link:
            lines.append(f"{i}. {title}\n   {snippet}\n   来源: {link}")
        else:
            lines.append(f"{i}. {title}\n   {snippet}")
    return "\n\n".join(lines)
