"""LLM网关"""

import asyncio
from typing import Optional
from openai import AsyncOpenAI, APIError, APITimeoutError, AuthenticationError
from app.config import settings
from app.observability.logging import get_logger

logger = get_logger(__name__)


class LLMGatewayError(Exception):
    """LLM网关异常"""
    pass


class LLMGateway:
    """LLM网关"""

    def __init__(self):
        self._client: Optional[AsyncOpenAI] = None

    @property
    def provider(self) -> str:
        """当前LLM提供商。

        每次从配置读取，避免全局单例在导入时把 provider 固化。
        """
        return settings.llm_provider

    @provider.setter
    def provider(self, value: str) -> None:
        """兼容测试中直接设置 provider 的写法。"""
        settings.llm_provider = value

    def _get_client(self) -> AsyncOpenAI:
        """获取或创建客户端"""
        if self._client is None:
            if not settings.deepseek_api_key:
                raise LLMGatewayError("未配置DEEPSEEK_API_KEY")
            self._client = AsyncOpenAI(
                api_key=settings.deepseek_api_key,
                base_url=settings.deepseek_base_url,
                timeout=settings.llm_timeout_seconds,
            )
        return self._client

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        allow_mock: bool = True,
    ) -> str:
        """生成回答"""
        if self.provider == "mock":
            if not allow_mock:
                raise LLMGatewayError("当前为 mock LLM，不能用于正式文档问答")
            return await self._mock_generate(prompt)
        elif self.provider == "deepseek":
            return await self._deepseek_generate(prompt, system_prompt, max_tokens, temperature)
        else:
            raise LLMGatewayError(f"不支持的LLM提供商: {self.provider}")

    async def _mock_generate(self, prompt: str) -> str:
        """模拟生成"""
        await asyncio.sleep(0.1)
        return f"[MOCK回答] 基于问题生成的回答: {prompt[:50]}..."

    async def _deepseek_generate(
        self,
        prompt: str,
        system_prompt: Optional[str],
        max_tokens: int,
        temperature: float,
    ) -> str:
        """调用DeepSeek API"""
        client = self._get_client()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = await client.chat.completions.create(
                model=settings.deepseek_model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            content = response.choices[0].message.content
            if not content:
                logger.warning("DeepSeek返回空内容")
                return "抱歉，无法生成回答。"
            return content
        except AuthenticationError as e:
            logger.error(f"DeepSeek认证失败: {e}")
            raise LLMGatewayError("API Key无效或已过期") from e
        except APITimeoutError as e:
            logger.error(f"DeepSeek请求超时: {e}")
            raise LLMGatewayError(f"请求超时({settings.llm_timeout_seconds}秒)") from e
        except APIError as e:
            logger.error(f"DeepSeek API错误: {e}")
            raise LLMGatewayError(f"API错误: {e.message}") from e
        except Exception as e:
            logger.error(f"DeepSeek调用异常: {e}")
            raise LLMGatewayError(f"调用异常: {str(e)}") from e


llm_gateway = LLMGateway()
