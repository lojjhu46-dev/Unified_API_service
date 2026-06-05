"""文档后端抽象基类

所有文档后端（DOCX MCP、XLSX MCP、TXT adapter、PDF reader）
实现统一接口，由执行智能体按 backend_required 分发调用。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from app.documents.models import DocumentOperationResult, DocumentPlan


class BackendError(Exception):
    """后端执行异常"""
    pass


class BackendUnavailableError(BackendError):
    """后端不可用（MCP 未连接、服务未启动等）"""
    pass


class DocumentBackend(ABC):
    """文档后端抽象接口"""

    @property
    @abstractmethod
    def name(self) -> str:
        """后端名称，用于日志和错误信息"""
        ...

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """后端是否可用（MCP 已连接、服务已就绪等）"""
        ...

    @abstractmethod
    async def execute(self, plan: DocumentPlan) -> DocumentOperationResult:
        """执行文档操作方案。

        统一契约：
        - 成功或可恢复的失败 → 返回 DocumentOperationResult（success=True/False）。
        - 底层原子操作（read_paragraph 等）允许抛 BackendError / BackendUnavailableError，
          由 execute() 内部捕获并转为 DocumentOperationResult。
        - 调用方不应期望 execute() 向上抛出异常。
        """
        ...

    @abstractmethod
    async def read_structure(self, file_path: str) -> dict[str, Any]:
        """读取文档结构信息

        Args:
            file_path: 文件路径

        Returns:
            结构信息字典，格式因文件类型而异

        Raises:
            BackendUnavailableError: 后端不可用
            BackendError: 读取失败
        """
        ...
