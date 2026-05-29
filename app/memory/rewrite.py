"""追问改写模块"""

from typing import List
from app.llm.gateway import llm_gateway
from app.observability.logging import get_logger

logger = get_logger(__name__)


def _history_to_text(history: List[dict]) -> str:
    """将历史消息转为文本"""
    if not history:
        return "无历史对话。"
    lines = []
    for item in history:
        role = "用户" if item["role"] == "user" else "助手"
        lines.append(f"{role}: {item['content']}")
    return "\n".join(lines)


async def rewrite_question(question: str, history: List[dict]) -> str:
    """将追问改写成独立问题"""
    if not history:
        return question
    if llm_gateway.provider == "mock":
        return question

    history_text = _history_to_text(history)
    system_prompt = (
        "你是对话式检索系统中的问题改写器。"
        "请根据历史对话理解代词、省略和指代关系，把用户本轮问题改写成一个可独立检索的问题。"
        "遇到“它、他、这个、那、其、时代演变、主要内容、历史影响”等省略追问时，"
        "必须结合最近用户主题补全核心对象。"
        "只输出改写后的问题，不要解释。"
    )
    user_prompt = (
        f"历史对话：\n{history_text}\n\n"
        f"用户本轮问题：\n{question}\n\n"
        f"独立检索问题："
    )

    try:
        rewritten = await llm_gateway.generate(
            user_prompt,
            system_prompt=system_prompt,
            max_tokens=200,
            temperature=0.3,
        )
        return rewritten.strip() if rewritten else question
    except Exception as e:
        logger.warning(f"追问改写失败，使用原问题: {e}")
        return question
