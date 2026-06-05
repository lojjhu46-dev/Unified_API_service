"""Phase 8: 文档智能体工具

提供 document_extract / document_plan / document_apply_plan / document_review 四个工具，
供编排器和 API 层调用。
"""

from __future__ import annotations

from pathlib import Path

from app.documents.executor import DocumentOperationAgent, describe_non_actionable_plan
from app.documents.models import (
    BackendType,
    DocumentIntent,
    DocumentPlan,
    FileType,
)
from app.documents.path_security import validate_file_path
from app.documents.planner import DocumentPlanningAgent
from app.config import settings
from app.observability.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# 文件类型推断与解析
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


def _resolve_file_type(file_path: str, file_type_str: str | None) -> tuple[FileType | None, str | None]:
    """安全解析文件类型，返回 (FileType, error_message)。

    优先使用显式传入的 file_type_str，其次从扩展名推断。
    非法值返回 (None, error_message)，不抛异常。
    """
    if file_type_str:
        try:
            return FileType(file_type_str), None
        except ValueError:
            valid = ", ".join(t.value for t in FileType)
            return None, f"不支持的文件类型: {file_type_str}，可选: {valid}"
    ft = infer_file_type(file_path)
    if ft is None:
        return None, f"无法推断文件类型: {file_path}"
    return ft, None


_BACKEND_BY_FILE_TYPE: dict[FileType, BackendType] = {
    FileType.DOCX: BackendType.DOCX_MCP,
    FileType.XLSX: BackendType.XLSX_MCP,
    FileType.TXT: BackendType.TEXT_ADAPTER,
    FileType.PDF: BackendType.PDF_READER,
}


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
        _executor = _create_default_executor()
    return _executor


def _create_default_executor() -> DocumentOperationAgent:
    """创建默认 executor，注册真实后端。"""
    from app.documents.adapters.pdf_real import RealPdfBackend
    from app.documents.adapters.txt_real import RealTxtBackend

    executor = DocumentOperationAgent()
    executor.register_backend(BackendType.TEXT_ADAPTER, RealTxtBackend())
    executor.register_backend(BackendType.PDF_READER, RealPdfBackend())

    # DOCX/XLSX HTTP MCP 后端（配置启用时注册）
    if settings.document_mcp_enabled:
        if settings.docx_mcp_base_url:
            from app.documents.adapters.docx_http import HttpDocxBackend
            executor.register_backend(
                BackendType.DOCX_MCP,
                HttpDocxBackend(settings.docx_mcp_base_url, settings.document_mcp_timeout_seconds),
            )
        if settings.xlsx_mcp_base_url:
            from app.documents.adapters.xlsx_http import HttpXlsxBackend
            executor.register_backend(
                BackendType.XLSX_MCP,
                HttpXlsxBackend(settings.xlsx_mcp_base_url, settings.document_mcp_timeout_seconds),
            )

    return executor


def set_executor(executor: DocumentOperationAgent) -> None:
    """注入自定义执行器（用于测试或替换后端）"""
    global _executor
    _executor = executor


def _reset_executor() -> None:
    """重置为默认空执行器（测试用）"""
    global _executor
    _executor = None


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

    resolved_path, err = validate_file_path(file_path)
    if err:
        return {"success": False, "error": err}
    file_path = str(resolved_path)

    file_type, err = _resolve_file_type(file_path, tool_input.get("file_type"))
    if err:
        return {"success": False, "error": err}

    backend_type = _BACKEND_BY_FILE_TYPE.get(file_type)
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

    resolved_path, err = validate_file_path(file_path)
    if err:
        return {"success": False, "error": err}
    file_path = str(resolved_path)

    file_type, err = _resolve_file_type(file_path, tool_input.get("file_type"))
    if err:
        return {"success": False, "error": err}

    structure = tool_input.get("structure")
    planner = _get_planner()

    try:
        plan = await planner.plan(user_command, file_type, structure)
        # 写回 file_path，确保 document_plan -> document_apply_plan 链路可用
        plan = plan.model_copy(update={"file_path": file_path})
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

    # 路径安全校验
    if not plan.file_path:
        return {"success": False, "error": "plan 缺少 file_path"}
    resolved_path, err = validate_file_path(plan.file_path)
    if err:
        return {"success": False, "error": err}
    plan = plan.model_copy(update={"file_path": str(resolved_path)})

    if not plan.is_actionable:
        response = {
            "success": False,
            "error": describe_non_actionable_plan(plan),
            "plan": plan.model_dump(),
        }
        if plan.clarification_question:
            response["requires_clarification"] = True
            response["clarification_question"] = plan.clarification_question
        return response

    confirmed = bool(tool_input.get("confirmed", False))
    if plan.intent == DocumentIntent.EDIT and not confirmed:
        error = "高风险操作需要确认" if plan.needs_confirmation else "编辑操作需要确认"
        return {
            "success": False,
            "error": error,
            "requires_confirmation": True,
            "plan": plan.model_dump(),
        }

    # 非编辑但显式标记需要确认的计划，也不能绕过确认。
    if plan.needs_confirmation and not confirmed:
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
    output: { success, summary, structure, error }
    """
    file_path = tool_input.get("file_path", "")
    if not file_path:
        return {"success": False, "error": "缺少 file_path"}

    resolved_path, err = validate_file_path(file_path)
    if err:
        return {"success": False, "error": err}
    file_path = str(resolved_path)

    file_type, err = _resolve_file_type(file_path, tool_input.get("file_type"))
    if err:
        return {"success": False, "error": err}

    backend_type = _BACKEND_BY_FILE_TYPE.get(file_type)

    try:
        plan = DocumentPlan(
            intent=DocumentIntent.REVIEW,
            file_type=file_type,
            file_path=file_path,
            backend_required=backend_type,
        )
        executor = _get_executor()
        result = await executor.execute(plan)
        return {
            "success": result.success,
            "summary": result.summary,
            "structure": result.verification.get("structure"),
            "error": result.error,
        }
    except Exception as e:
        logger.error(f"文档审阅失败: {e}")
        return {"success": False, "error": str(e)}
