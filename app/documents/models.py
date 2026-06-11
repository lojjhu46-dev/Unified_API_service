"""统一文档任务模型

Phase 1: 定义 DocumentPlan / DocumentOperation / DocumentOperationResult，
供规划智能体与执行智能体共用。
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class DocumentIntent(str, Enum):
    """用户对文档的意图"""
    EDIT = "edit"
    REVIEW = "review"
    EXTRACT = "extract"
    SUMMARIZE = "summarize"
    UNSUPPORTED = "unsupported"


class FileType(str, Enum):
    """支持的文件类型"""
    DOCX = "docx"
    XLSX = "xlsx"
    TXT = "txt"
    PDF = "pdf"


class BackendType(str, Enum):
    """执行后端"""
    DOCX_MCP = "docx_mcp"
    XLSX_MCP = "xlsx_mcp"
    TEXT_ADAPTER = "text_adapter"
    PDF_READER = "pdf_reader"


class RiskLevel(str, Enum):
    """操作风险等级"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


BACKEND_BY_FILE_TYPE: dict[FileType, BackendType] = {
    FileType.DOCX: BackendType.DOCX_MCP,
    FileType.XLSX: BackendType.XLSX_MCP,
    FileType.TXT: BackendType.TEXT_ADAPTER,
    FileType.PDF: BackendType.PDF_READER,
}


# ---------------------------------------------------------------------------
# 单条操作
# ---------------------------------------------------------------------------

class DocumentOperation(BaseModel):
    """规划智能体生成的单条结构化操作"""
    action: str = Field(
        ...,
        description="操作动作，如 replace_paragraph, delete_paragraph, append_text, "
                    "modify_cell, delete_row, extract_text 等",
    )
    target: dict[str, Any] = Field(
        default_factory=dict,
        description="定位信息，如 paragraph_index, cell_ref, page_number, "
                    "search_text, line_range 等",
    )
    value: Optional[str] = Field(
        default=None,
        description="写入值（替换文本、单元格内容等），删除操作时为 None",
    )
    description: str = Field(
        default="",
        description="人类可读的操作说明，供确认环节展示",
    )


# ---------------------------------------------------------------------------
# 规划结果
# ---------------------------------------------------------------------------

class DocumentPlan(BaseModel):
    """规划智能体输出：可执行的结构化方案，或澄清问题"""
    intent: DocumentIntent = Field(
        ...,
        description="用户意图",
    )
    file_type: FileType = Field(
        ...,
        description="目标文件类型",
    )
    file_path: str = Field(
        default="",
        description="目标文件路径（由调用方填充）",
    )
    operations: list[DocumentOperation] = Field(
        default_factory=list,
        description="结构化操作列表；intent 为 unsupported 或需要澄清为空",
    )
    backend_required: Optional[BackendType] = Field(
        default=None,
        description="执行所需后端；intent=unsupported 时为 None",
    )
    risk_level: RiskLevel = Field(
        default=RiskLevel.LOW,
        description="操作风险等级",
    )
    requires_confirmation: bool = Field(
        default=False,
        description="高风险操作需要用户确认后才能执行",
    )
    clarification_question: Optional[str] = Field(
        default=None,
        description="当命令定位不清时，向用户提出澄清问题；非空时 operations 为空",
    )
    unsupported_reason: Optional[str] = Field(
        default=None,
        description="intent=unsupported 时的原因说明",
    )
    edit_in_place: bool = Field(
        default=False,
        exclude=True,
        description="内部执行字段：系统生成的编辑副本可原地续编，不由 planner 生成",
    )

    # ---- 安全规则校验 ----

    @model_validator(mode="after")
    def enforce_security_rules(self) -> "DocumentPlan":
        """在模型层强制执行文档安全规则，非法组合直接拒绝。"""
        if self.intent == DocumentIntent.UNSUPPORTED:
            return self

        # PDF 禁止编辑
        if self.file_type == FileType.PDF and self.intent == DocumentIntent.EDIT:
            raise ValueError("PDF 不支持编辑，仅支持 extract / review / summarize")

        required_backend = BACKEND_BY_FILE_TYPE[self.file_type]
        if self.backend_required != required_backend:
            raise ValueError(
                f"{self.file_type.value} 文件必须使用 {required_backend.value} 后端，"
                f"当前为 {self.backend_required}"
            )

        return self

    # ---- 便捷判断 ----

    @property
    def is_actionable(self) -> bool:
        """方案是否可执行（无需澄清、非 unsupported、有后端）。

        - EDIT 意图要求 operations 非空。
        - REVIEW / EXTRACT / SUMMARIZE 可以没有 operations（整个文档即目标）。
        """
        if self.intent == DocumentIntent.UNSUPPORTED:
            return False
        if self.clarification_question is not None:
            return False
        if self.backend_required is None:
            return False
        if self.intent == DocumentIntent.EDIT and len(self.operations) == 0:
            return False
        return True

    @property
    def needs_confirmation(self) -> bool:
        """是否需要用户确认"""
        return self.requires_confirmation or self.risk_level == RiskLevel.HIGH


# ---------------------------------------------------------------------------
# 执行结果
# ---------------------------------------------------------------------------

class DocumentOperationResult(BaseModel):
    """执行智能体返回的执行结果"""
    success: bool = Field(..., description="执行是否成功")
    output_file: Optional[str] = Field(
        default=None,
        description="生成的新文件路径；只读操作或失败时为 None",
    )
    summary: str = Field(
        default="",
        description="执行摘要，供前端展示",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="非致命警告",
    )
    verification: dict[str, Any] = Field(
        default_factory=dict,
        description="校验信息，如原文件未变、目标段落已修改等",
    )
    error: Optional[str] = Field(
        default=None,
        description="失败时的错误信息",
    )
