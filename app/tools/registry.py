"""工具注册表"""

from typing import List, Optional
from app.schemas import ToolTrace
from app.tools.calculator import calculate
from app.tools.search import web_search, format_search_results
from app.observability.logging import get_logger

logger = get_logger(__name__)


class ToolRegistry:
    """工具注册表"""

    def __init__(self):
        self._tools = {
            "calculator": self._run_calculator,
            "web_search": self._run_web_search,
        }

    async def execute(self, tool_name: str, tool_input: dict) -> ToolTrace:
        """执行工具"""
        if tool_name not in self._tools:
            return ToolTrace(
                tool_name=tool_name,
                tool_input=tool_input,
                status="error",
                output_preview="",
                latency_ms=0,
                error_message=f"未知工具: {tool_name}",
            )

        import time
        start = time.perf_counter()

        try:
            result = await self._tools[tool_name](tool_input)
            latency_ms = (time.perf_counter() - start) * 1000

            return ToolTrace(
                tool_name=tool_name,
                tool_input=tool_input,
                status="success",
                output_preview=str(result)[:200],
                latency_ms=latency_ms,
            )
        except Exception as e:
            latency_ms = (time.perf_counter() - start) * 1000
            logger.error(f"工具 {tool_name} 执行失败: {e}")
            return ToolTrace(
                tool_name=tool_name,
                tool_input=tool_input,
                status="error",
                output_preview="",
                latency_ms=latency_ms,
                error_message=str(e),
            )

    async def _run_calculator(self, tool_input: dict) -> dict:
        """执行计算器"""
        expression = tool_input.get("expression", "")
        return calculate(expression)

    async def _run_web_search(self, tool_input: dict) -> dict:
        """执行联网搜索"""
        query = tool_input.get("query", "")
        return await web_search(query)

    def get_available_tools(self) -> List[str]:
        """获取可用工具列表"""
        return list(self._tools.keys())


tool_registry = ToolRegistry()
