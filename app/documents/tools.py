"""Phase 8: 文档智能体工具

提供 document_extract / document_plan / document_apply_plan / document_review 四个工具，
供编排器和 API 层调用。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from app.documents.adapters.docx import MockDocxBackend
from app.documents.adapters.pdf import MockPdfBackend
from app.documents.adapters.txt import MockTxtBackend
from app.documents.adapters.xlsx import MockXlsxBackend
from app.documents.executor import DocumentOperationAgent
from app.documents.models import (
    BackendType,
    DocumentIntent,
    DocumentOperationResult,
    DocumentPlan,
    FileType,
)
from app.documents.planner import DocumentPlanningAgent
from app.observability.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# 文件类型推断
# ---------------------------------------------------------------------------

_EXT_TO_FILE_TYPE: dict[str, FileType] = {
    ".docx": FileType.DOCX,
    ".xlsx": FileType.XLSX,
    ".txt": FileType.TXT,
    ".pdf": FileType.PDF,
}


def infer_file_type(file_path: str) -> FileType | None:
    """从文件扩展名推断 FileType"""
    ext = Path(file_path).suffix.lower()
    return _EXT_TO_FILE_TYPE.get(ext)


# ---------------------------------------------------------------------------
# 全局实例（lazy 初始化）
# ---------------------------------------------------------------------------

_planner: DocumentPlanningAgent | None = None
_executor: DocumentOperationAgent | None = None


def _get_planner() -> DocumentPlanningAgent:
    global _planner
    if _planner is None:
        _planner = DocumentPlanningAgent()
    return _planner


def _get_executor() -> DocumentOperationAgent:
    global _executor
    if _executor is None:
        _executor = DocumentOperationAgent()
        # 注册 mock 后端（真实 MCP 后端后续替换）
        _executor.register_backend(BackendType.DOCX_MCP, MockDocxBackend())
        _executor.register_backend(BackendType.XLSX_MCP, MockXlsxBackend())
        _executor.register_backend(BackendType.TEXT_ADAPTER, MockTxtBackend())
        _executor.register_backend(BackendType.PDF_READER, MockPdfBackend())
    return _executor


def set_executor(executor: DocumentOperationAgent) -> None:
    """注入自定义执行器（用于测试或替换后端）"""
    global _executor
    _executor = executor


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

async def document_extract(tool_input: dict) -> dict:
    """提取文档结构信息

    input: { file_path: str, file_type?: str }
    output: { success, structure, file_type }
    """
    file_path = tool_input.get("file_path", "")
    if not file_path:
        return {"success": False, "error": "缺少 file_path"}

    file_type_str = tool_input.get("file_type")
    file_type = FileType(file_type_str) if file_type_str else infer_file_type(file_path)
    if file_type is None:
        return {"success": False, "error": f"无法推断文件类型: {file_path}"}

    backend_type = {
        FileType.DOCX: BackendType.DOCX_MCP,
        FileType.XLSX: BackendType.XLSX_MCP,
        FileType.TXT: BackendType.TEXT_ADAPTER,
        FileType.PDF: BackendType.PDF_READER,
    }.get(file_type)

    executor = _get_executor()
    backend = executor.get_backend(backend_type) if backend_type else None
    if backend is None:
        return {"success": False, "error": f"后端 {backend_type} 未注册"}

    try:
        structure = await backend.read_structure(file_path)
        return {
            "success": True,
            "file_type": file_type.value,
            "structure": structure,
        }
    except FileNotFoundError:
        return {"success": False, "error": f"文件不存在: {file_path}"}
    except Exception as e:
        logger.error(f"文档提取失败: {e}")
        return {"success": False, "error": str(e)}


async def document_plan(tool_input: dict) -> dict:
    """分析用户命令，生成编辑方案

    input: { user_command: str, file_path: str, file_type?: str, structure?: dict }
    output: { success, plan: DocumentPlan.model_dump() }
    """
    user_command = tool_input.get("user_command", "")
    file_path = tool_input.get("file_path", "")
    if not user_command:
        return {"success": False, "error": "缺少 user_command"}
    if not file_path:
        return {"success": False, "error": "缺少 file_path"}

    file_type_str = tool_input.get("file_type")
    file_type = FileType(file_type_str) if file_type_str else infer_file_type(file_path)
    if file_type is None:
        return {"success": False, "error": f"无法推断文件类型: {file_path}"}

    structure = tool_input.get("structure")
    planner = _get_planner()

    try:
        plan = await planner.plan(user_command, file_type, structure)
        return {
            "success": True,
            "plan": plan.model_dump(),
        }
    except Exception as e:
        logger.error(f"文档规划失败: {e}")
        return {"success": False, "error": str(e)}


async def document_apply_plan(tool_input: dict) -> dict:
    """执行已生成的编辑方案

    input: { plan: dict }  (DocumentPlan 的序列化形式)
    output: { success, result: DocumentOperationResult.model_dump() }
    """
    plan_data = tool_input.get("plan")
    if not plan_data:
        return {"success": False, "error": "缺少 plan"}

    try:
        plan = DocumentPlan.model_validate(plan_data)
    except Exception as e:
        return {"success": False, "error": f"plan 解析失败: {e}"}

    # 高风险操作需要确认
    if plan.needs_confirmation:
        confirmed = tool_input.get("confirmed", False)
        if not confirmed:
            return {
                "success": False,
                "error": "高风险操作需要确认",
                "requires_confirmation": True,
                "plan": plan.model_dump(),
            }

    executor = _get_executor()

    try:
        result = await executor.execute(plan)
        return {
            "success": result.success,
            "result": result.model_dump(),
        }
    except Exception as e:
        logger.error(f"文档执行失败: {e}")
        return {"success": False, "error": str(e)}


async def document_review(tool_input: dict) -> dict:
    """审阅文档（只读）

    input: { file_path: str, file_type?: str, user_command?: str }
    output: { success, summary, structure }
    """
    file_path = tool_input.get("file_path", "")
    if not file_path:
        return {"success": False, "error": "缺少 file_path"}

    file_type_str = tool_input.get("file_type")
    file_type = FileType(file_type_str) if file_type_str else infer_file_type(file_path)
    if file_type is None:
        return {"success": False, "error": f"无法推断文件类型: {file_path}"}

    backend_type = {
        FileType.DOCX: BackendType.DOCX_MCP,
        FileType.XLSX: BackendType.XLSX_MCP,
        FileType.TXT: BackendType.TEXT_ADAPTER,
        FileType.PDF: BackendType.PDF_READER,
    }.get(file_type)

    executor = _get_executor()
    backend = executor.get_backend(backend_type) if backend_type else None
    if backend is None:
        return {"success": False, "error": f"后端 {backend_type} 未注册"}

    try:
        plan = DocumentPlan(
            intent=DocumentIntent.REVIEW,
            file_type=file_type,
            file_path=file_path,
            backend_required=backend_type,
        )
        result = await backend.execute(plan)
        return {
            "success": result.success,
            "summary": result.summary,
            "structure": result.verification.get("structure"),
            "error": result.error,
        }
    except Exception as e:
        logger.error(f"文档审阅失败: {e}")
        return {"success": False, "error": str(e)}
