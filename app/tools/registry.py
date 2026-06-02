"""工具注册表"""

from dataclasses import dataclass
from typing import List
from app.schemas import ToolTrace
from app.tools.calculator import calculate
from app.tools.search import web_search
from app.observability.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ToolExecution:
    """工具执行结果"""

    trace: ToolTrace
    result: dict


class ToolRegistry:
    """工具注册表"""

    def __init__(self):
        self._tools = {
            "calculator": self._run_calculator,
            "web_search": self._run_web_search,
        }

    async def execute_with_result(self, tool_name: str, tool_input: dict) -> ToolExecution:
        """执行工具并返回业务结果"""
        if tool_name not in self._tools:
            trace = ToolTrace(
                tool_name=tool_name,
                tool_input=tool_input,
                status="error",
                output_preview="",
                latency_ms=0,
                error_message=f"未知工具: {tool_name}",
            )
            return ToolExecution(trace=trace, result={"success": False, "error": trace.error_message})

        import time
        start = time.perf_counter()

        try:
            result = await self._tools[tool_name](tool_input)
            latency_ms = (time.perf_counter() - start) * 1000
            is_error_result = isinstance(result, dict) and result.get("success") is False
            error_message = result.get("error") if is_error_result else None

            trace = ToolTrace(
                tool_name=tool_name,
                tool_input=tool_input,
                status="error" if is_error_result else "success",
                output_preview=str(result)[:200],
                latency_ms=latency_ms,
                error_message=error_message,
            )
            return ToolExecution(trace=trace, result=result)
        except Exception as e:
            latency_ms = (time.perf_counter() - start) * 1000
            logger.error(f"工具 {tool_name} 执行失败: {e}")
            trace = ToolTrace(
                tool_name=tool_name,
                tool_input=tool_input,
                status="error",
                output_preview="",
                latency_ms=latency_ms,
                error_message=str(e),
            )
            return ToolExecution(trace=trace, result={"success": False, "error": str(e)})

    async def execute(self, tool_name: str, tool_input: dict) -> ToolTrace:
        """执行工具，兼容只需要轨迹的调用方"""
        execution = await self.execute_with_result(tool_name, tool_input)
        return execution.trace

    async def _run_calculator(self, tool_input: dict) -> dict:
        """执行计算器"""
        expression = tool_input.get("expression", "")
        return calculate(expression)

    async def _run_web_search(self, tool_input: dict) -> dict:
        """执行联网搜索"""
        query = tool_input.get("query", "")
        return await web_search(query, domains=tool_input.get("domains"))

    def get_available_tools(self) -> List[str]:
        """获取可用工具列表"""
        return list(self._tools.keys())


tool_registry = ToolRegistry()
