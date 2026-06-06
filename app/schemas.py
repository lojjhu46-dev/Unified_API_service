"""统一请求响应模型"""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field, field_validator
from datetime import datetime


class AskRequest(BaseModel):
    """统一问答请求"""
    channel: str = Field(default="api", description="来源渠道")
    user_id: str = Field(..., description="用户标识")
    session_id: Optional[str] = Field(default=None, description="会话ID")
    question: str = Field(..., min_length=1, description="用户问题")
    knowledge_scope: List[str] = Field(default=["default"], description="知识库范围")
    need_web: Literal["auto", "always", "never"] = Field(default="auto", description="是否需要联网搜索")
    top_k: int = Field(default=5, ge=1, le=20, description="检索返回来源数量")
    return_trace: bool = Field(default=False, description="是否返回工具调用轨迹")
    # 文档工具字段（可选，保持旧请求兼容）
    document_id: Optional[str] = Field(default=None, description="个人知识库文档ID")
    document_file_path: Optional[str] = Field(default=None, description="文档文件路径")
    document_file_type: Optional[Literal["docx", "xlsx", "txt", "pdf"]] = Field(default=None, description="文档文件类型")
    document_action: Literal["auto", "extract", "review", "plan", "apply", "list"] = Field(default="auto", description="文档操作类型")
    document_plan: Optional[dict] = Field(default=None, description="已生成的编辑方案（用于 apply）")
    document_confirmed: bool = Field(default=False, description="是否确认执行高风险操作")

    @field_validator("question")
    @classmethod
    def clean_question(cls, value: str) -> str:
        """清理用户问题，避免纯空白内容进入编排层。"""
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("问题不能为空")
        return cleaned


class SourceItem(BaseModel):
    """来源项"""
    title: str = Field(default="", description="来源标题")
    url: str = Field(default="", description="来源链接")
    source_type: str = Field(default="knowledge_base", description="来源类型")
    snippet: str = Field(default="", description="内容摘要")
    content: str = Field(default="", exclude=True, description="内部完整上下文")
    score: float = Field(default=0.0, description="相关性分数")
    metadata: dict = Field(default_factory=dict, description="元数据")


class ToolTrace(BaseModel):
    """工具调用轨迹"""
    tool_name: str = Field(..., description="工具名称")
    tool_input: dict = Field(default_factory=dict, description="工具输入")
    status: Literal["success", "error"] = Field(default="success", description="执行状态")
    output_preview: str = Field(default="", description="输出预览")
    latency_ms: float = Field(default=0.0, description="耗时(毫秒)")
    error_message: Optional[str] = Field(default=None, description="错误信息")


class TimingInfo(BaseModel):
    """耗时信息"""
    rewrite_ms: float = Field(default=0.0, description="问题改写耗时")
    retrieval_ms: float = Field(default=0.0, description="检索耗时")
    tool_ms: float = Field(default=0.0, description="工具调用耗时")
    llm_ms: float = Field(default=0.0, description="LLM生成耗时")
    total_ms: float = Field(default=0.0, description="总耗时")


class AgentResponse(BaseModel):
    """统一问答响应"""
    request_id: str = Field(..., description="请求ID")
    session_id: str = Field(..., description="会话ID")
    route: Literal["direct", "rag", "web", "tool", "agentic_rag"] = Field(..., description="路由类型")
    answer: str = Field(default="", description="回答内容")
    sources: List[SourceItem] = Field(default_factory=list, description="来源列表")
    tool_trace: List[ToolTrace] = Field(default_factory=list, description="工具调用轨迹")
    timing: TimingInfo = Field(default_factory=TimingInfo, description="耗时信息")
    standalone_question: Optional[str] = Field(default=None, description="改写后的独立问题")


class UploadResponse(BaseModel):
    """文档上传响应"""
    document_id: str = Field(..., description="文档ID")
    filename: str = Field(..., description="文件名")
    chunks: int = Field(..., description="切块数量")
    status: str = Field(default="success", description="状态")
    message: str = Field(default="", description="消息")


class SessionCreateResponse(BaseModel):
    """会话创建响应"""
    session_id: str = Field(..., description="会话ID")


class SessionHistoryResponse(BaseModel):
    """会话历史响应"""
    session_id: str = Field(..., description="会话ID")
    messages: List[dict] = Field(default_factory=list, description="历史消息")


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str = Field(default="ok", description="状态")
    version: str = Field(..., description="版本")
    timestamp: datetime = Field(default_factory=datetime.now, description="时间戳")
    components: dict = Field(default_factory=dict, description="组件健康状态")


class ErrorResponse(BaseModel):
    """错误响应"""
    detail: str = Field(..., description="错误详情")
    request_id: Optional[str] = Field(default=None, description="请求ID")
