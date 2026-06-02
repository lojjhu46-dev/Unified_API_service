"""Prompt模板"""

SYSTEM_PROMPT = """你是一个严谨的中文问答助手。
请优先依据给定的文档片段回答问题；只要任一片段包含依据，就应基于该片段作答，不要因为其他片段无关而整体否定。
如果所有文档片段都没有答案，请明确说明没有在文档中找到依据。
当用户要求代码或代码片段时，如果文档片段包含 def、import、return、np.、plt.、ax. 等代码标记，应直接提取相关代码并用代码块回答；不要因为片段前半部分是说明文字或其他片段无关而回答未找到。
回答要简洁、准确。"""


RAG_PROMPT_TEMPLATE = """基于以下文档回答问题。

文档片段：
{context}

问题：{question}

请回答："""


WEB_SEARCH_PROMPT_TEMPLATE = """基于以下联网搜索结果回答问题。

搜索片段：
{context}

问题：{question}

要求：
1. 优先基于搜索结果作答，不要编造未出现的信息。
2. 如果有来源链接，请在回答中保留或列出来源。
3. 对天气、新闻、论坛、技术方案、指定网站内容，优先相信联网结果。

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


def build_web_search_prompt(question: str, context: str) -> tuple[str, str]:
    """构建联网搜索回答提示词。"""
    system = "你是一个有用的中文助手。回答时必须保留来源链接。"
    user = WEB_SEARCH_PROMPT_TEMPLATE.format(context=context, question=question)
    return system, user
