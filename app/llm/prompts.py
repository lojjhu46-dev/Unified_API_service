"""Prompt模板"""

SYSTEM_PROMPT = """你是一个严谨的中文问答助手。
请优先依据给定的文档片段回答问题；如果文档中没有答案，请明确说明没有在文档中找到依据。
回答要简洁、准确。"""


RAG_PROMPT_TEMPLATE = """基于以下文档回答问题。

文档片段：
{context}

问题：{question}

请回答："""


DIRECT_PROMPT_TEMPLATE = """请直接回答以下问题：

{question}"""


GLOBAL_SEARCH_PROMPT = """本轮是全局汇总任务，请从所有给定片段中抽取完整列表。
只要部分片段包含答案，就不要因为其他片段无关而回答没有找到。"""


def build_rag_prompt(question: str, context: str, is_global: bool = False) -> tuple[str, str]:
    """构建RAG提示词"""
    system = SYSTEM_PROMPT
    if is_global:
        system += "\n\n" + GLOBAL_SEARCH_PROMPT

    user = RAG_PROMPT_TEMPLATE.format(context=context, question=question)
    return system, user


def build_direct_prompt(question: str) -> tuple[str, str]:
    """构建直接问答提示词"""
    system = "你是一个有用的中文助手。"
    user = DIRECT_PROMPT_TEMPLATE.format(question=question)
    return system, user
