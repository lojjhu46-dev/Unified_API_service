"""联网搜索工具"""

import time
import httpx
from typing import List
from app.schemas import SourceItem
from app.config import settings
from app.observability.logging import get_logger

logger = get_logger(__name__)


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


async def web_search(query: str) -> dict:
    """执行联网搜索"""
    if not settings.serper_api_key:
        return {
            "success": False,
            "error": "未配置SERPER_API_KEY",
            "results": [],
        }

    headers = {
        "X-API-KEY": settings.serper_api_key,
        "Content-Type": "application/json",
    }
    payload = {"q": query, "gl": "cn", "hl": "zh-cn"}

    for attempt in range(settings.max_retries + 1):
        try:
            async with httpx.AsyncClient(timeout=settings.search_timeout) as client:
                resp = await client.post(settings.serper_url, json=payload, headers=headers)

                if resp.status_code == 429:
                    if attempt < settings.max_retries:
                        wait_time = 2 ** attempt
                        time.sleep(wait_time)
                        continue
                    return {
                        "success": False,
                        "error": "搜索请求过于频繁（HTTP 429），请稍后重试",
                        "results": [],
                    }

                if resp.status_code == 401:
                    return {
                        "success": False,
                        "error": "搜索服务认证失败（HTTP 401），SERPER_API_KEY 无效",
                        "results": [],
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
                        "message": f"针对「{query}」的搜索未找到相关结果",
                    }

                return {
                    "success": True,
                    "error": None,
                    "results": results,
                }

        except httpx.TimeoutException:
            if attempt < settings.max_retries:
                continue
            return {
                "success": False,
                "error": f"搜索请求超时（等待了 {settings.search_timeout} 秒）",
                "results": [],
            }
        except httpx.ConnectError:
            return {
                "success": False,
                "error": "无法连接搜索服务，请检查网络连接",
                "results": [],
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"搜索请求异常: {type(e).__name__}",
                "results": [],
            }

    return {
        "success": False,
        "error": f"搜索失败，已重试 {settings.max_retries} 次",
        "results": [],
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
