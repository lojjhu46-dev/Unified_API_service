"""DOCX 文档操作

基于 python-docx 的同步操作实现。
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from docx import Document


def read_structure(file_path: str) -> dict[str, Any]:
    """读取 DOCX 文档结构"""
    doc = Document(file_path)
    paragraphs = doc.paragraphs

    headings = []
    paragraphs_info = []
    max_paragraphs = 200  # 最多返回前 200 段
    max_text_length = 500  # 每段最多 500 字

    for i, p in enumerate(paragraphs[:max_paragraphs]):
        text = p.text
        # 截断过长的文本
        if len(text) > max_text_length:
            text = text[:max_text_length] + "..."

        paragraphs_info.append({
            "index": i,
            "text": text,
        })

        if p.style and p.style.name and p.style.name.startswith("Heading"):
            headings.append(p.text)

    # tables 改为列表格式
    tables_info = []
    for i, table in enumerate(doc.tables):
        tables_info.append({
            "index": i,
            "rows": len(table.rows),
            "cols": len(table.columns),
        })

    return {
        "type": "docx",
        "paragraph_count": len(paragraphs),
        "paragraphs": paragraphs_info,
        "headings": headings,
        "tables": tables_info,
    }


def read_paragraph(file_path: str, index: int) -> str:
    """读取指定段落的文本"""
    doc = Document(file_path)
    paragraphs = doc.paragraphs

    if index < 0 or index >= len(paragraphs):
        raise IndexError(f"段落索引 {index} 超出范围（共 {len(paragraphs)} 段）")

    return paragraphs[index].text


def replace_paragraph(file_path: str, index: int, new_text: str) -> None:
    """替换指定段落的文本"""
    doc = Document(file_path)
    paragraphs = doc.paragraphs

    if index < 0 or index >= len(paragraphs):
        raise IndexError(f"段落索引 {index} 超出范围（共 {len(paragraphs)} 段）")

    # 清除原有内容并添加新文本，保留段落样式
    paragraph = paragraphs[index]
    paragraph.clear()
    paragraph.add_run(new_text)

    doc.save(file_path)


def delete_paragraph(file_path: str, index: int) -> None:
    """删除指定段落"""
    doc = Document(file_path)
    paragraphs = doc.paragraphs

    if index < 0 or index >= len(paragraphs):
        raise IndexError(f"段落索引 {index} 超出范围（共 {len(paragraphs)} 段）")

    # 从 XML 树中移除段落元素
    paragraph = paragraphs[index]
    parent = paragraph._element.getparent()
    if parent is not None:
        parent.remove(paragraph._element)

    doc.save(file_path)


def append_paragraph(file_path: str, text: str) -> None:
    """在文档末尾追加段落"""
    doc = Document(file_path)
    doc.add_paragraph(text)
    doc.save(file_path)


def save_copy(file_path: str, output_path: str) -> str:
    """保存文档副本到指定路径"""
    shutil.copy2(file_path, output_path)
    return output_path
