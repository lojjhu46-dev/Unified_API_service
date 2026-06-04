"""Uploaded file summarization helpers."""

import os
import tempfile
from pathlib import Path

from app.llm.gateway import LLMGatewayError, llm_gateway
from app.retrieval.ingest import load_document, sanitize_filename, validate_file_extension

UNSUPPORTED_SUMMARY_TEXT = "仅支持 PDF、DOCX、TXT、XLSX 文件。"
MAX_SUMMARY_TEXT_CHARS = 12000
MAX_SAMPLE_TEXT_CHARS = 2000


def _truncate(text: str, limit: int) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    return text[:limit], True


def _document_text_and_outline(documents: list) -> tuple[str, list[str]]:
    parts = []
    outline = []
    for index, doc in enumerate(documents, start=1):
        content = (getattr(doc, "page_content", "") or "").strip()
        if not content:
            continue
        metadata = getattr(doc, "metadata", {}) if isinstance(getattr(doc, "metadata", None), dict) else {}
        label = metadata.get("sheet") or metadata.get("page") or f"片段{index}"
        outline.append(str(label))
        parts.append(f"[{label}]\n{content}")
    return "\n\n".join(parts), outline


def _extractive_summary(text: str, title: str = "") -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return f"{title or '资源'}未抽取到可总结的文本内容。"
    selected = "\n".join(lines[:8])
    return f"已识别到文本内容，前 {min(len(lines), 8)} 条要点如下：\n{selected}"


async def _generate_summary(text: str, title: str, resource_type: str) -> str:
    prompt = (
        "请基于以下已抽取的文本内容生成简洁中文摘要。"
        "输出包括：1. 内容概述；2. 关键要点；3. 如为表格，说明表头/字段和数据主题。"
        "不要编造未出现的信息。\n\n"
        f"资源类型：{resource_type}\n标题：{title}\n\n内容：\n{text}"
    )
    try:
        return await llm_gateway.generate(
            prompt,
            system_prompt="你是严谨的文件识别与总结助手。",
            max_tokens=800,
            temperature=0.2,
            allow_mock=False,
        )
    except LLMGatewayError:
        return _extractive_summary(text, title)


async def summarize_uploaded_file_content(content: bytes, filename: str) -> dict:
    if not validate_file_extension(filename):
        return {"success": False, "error": UNSUPPORTED_SUMMARY_TEXT}
    if len(content) > 10 * 1024 * 1024:
        return {"success": False, "error": "文件大小不能超过10MB。"}

    safe_filename = sanitize_filename(filename)
    ext = os.path.splitext(safe_filename)[1].lower().lstrip(".")
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = Path(temp_dir) / safe_filename
        file_path.write_bytes(content)
        documents = load_document(str(file_path))

    text, outline = _document_text_and_outline(documents)
    sample_text, sample_truncated = _truncate(text, MAX_SAMPLE_TEXT_CHARS)
    summary_input, summary_truncated = _truncate(text, MAX_SUMMARY_TEXT_CHARS)
    summary = await _generate_summary(summary_input, safe_filename, ext)
    warnings = []
    if sample_truncated or summary_truncated:
        warnings.append("内容较长，已截断部分文本用于总结。")

    return {
        "success": True,
        "resource_type": ext,
        "title": safe_filename,
        "summary": summary,
        "outline": outline,
        "sample_text": sample_text,
        "warnings": warnings,
    }
