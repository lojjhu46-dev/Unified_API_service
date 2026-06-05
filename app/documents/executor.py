"""Phase 7: 执行智能体

DocumentOperationAgent 只接收 DocumentPlan，根据 backend_required 分发到对应后端。
不直接解析自然语言，不做规划，只执行。
"""

from __future__ import annotations

from app.documents.adapters.base import DocumentBackend
from app.documents.adapters.docx import DocxBackend
from app.documents.adapters.pdf import PdfBackend
from app.documents.adapters.txt import TxtBackend
from app.documents.adapters.xlsx import XlsxBackend
from app.documents.models import (
    BackendType,
    DocumentIntent,
    DocumentOperationResult,
    DocumentPlan,
    FileType,
)
from app.observability.logging import get_logger

logger = get_logger(__name__)


class DocumentOperationAgent:
    """执行智能体：接收 DocumentPlan，分发到对应后端执行。

    职责单一：
    - 不解析自然语言
    - 不做规划决策
    - 只做路由分发 + 执行后校验
    """

    def __init__(
        self,
        docx_backend: DocxBackend | None = None,
        xlsx_backend: XlsxBackend | None = None,
        txt_backend: TxtBackend | None = None,
        pdf_backend: PdfBackend | None = None,
    ) -> None:
        self._backends: dict[BackendType, DocumentBackend] = {}
        if docx_backend:
            self._backends[BackendType.DOCX_MCP] = docx_backend
        if xlsx_backend:
            self._backends[BackendType.XLSX_MCP] = xlsx_backend
        if txt_backend:
            self._backends[BackendType.TEXT_ADAPTER] = txt_backend
        if pdf_backend:
            self._backends[BackendType.PDF_READER] = pdf_backend

    def register_backend(self, backend_type: BackendType, backend: DocumentBackend) -> None:
        """注册后端实例"""
        self._backends[backend_type] = backend

    def get_backend(self, backend_type: BackendType) -> DocumentBackend | None:
        """获取已注册的后端实例"""
        return self._backends.get(backend_type)

    async def execute(self, plan: DocumentPlan) -> DocumentOperationResult:
        """执行 DocumentPlan。

        校验链：
        1. intent 不为 unsupported
        2. backend_required 非空
        3. backend 已注册且可用
        4. 分发到后端执行
        5. 执行后校验结果
        """
        # 校验 intent
        if plan.intent == DocumentIntent.UNSUPPORTED:
            return DocumentOperationResult(
                success=False,
                error="不支持的操作意图",
            )

        # 校验 backend_required
        if plan.backend_required is None:
            return DocumentOperationResult(
                success=False,
                error="backend_required 为空，无法执行",
            )

        # 查找后端
        backend = self._backends.get(plan.backend_required)
        if backend is None:
            return DocumentOperationResult(
                success=False,
                error=f"后端 {plan.backend_required} 未注册",
            )

        # 分发执行
        try:
            result = await backend.execute(plan)
        except Exception as e:
            logger.error(f"执行异常: {e}")
            return DocumentOperationResult(
                success=False,
                error=f"执行异常: {e}",
            )

        # 执行后校验
        self._validate_result(plan, result)
        return result

    @staticmethod
    def _validate_result(plan: DocumentPlan, result: DocumentOperationResult) -> None:
        """执行后校验：检查结果一致性。

        不修改 result，只记录 warnings。
        """
        warnings = result.warnings

        # 编辑操作成功时，output_file 不应为空
        if plan.intent == DocumentIntent.EDIT and result.success and result.output_file is None:
            warnings.append("编辑操作成功但 output_file 为空")

        # 只读操作不应生成 output_file
        if plan.intent in (DocumentIntent.REVIEW, DocumentIntent.EXTRACT, DocumentIntent.SUMMARIZE):
            if result.output_file is not None:
                warnings.append("只读操作不应生成 output_file")

        # PDF 不应生成 output_file
        if plan.file_type == FileType.PDF and result.output_file is not None:
            warnings.append("PDF 操作不应生成 output_file")
