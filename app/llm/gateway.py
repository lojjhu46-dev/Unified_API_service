"""LLM网关"""

import asyncio
from typing import Optional
from app.config import settings
from app.observability.logging import get_logger

logger = get_logger(__name__)


class LLMGateway:
    """LLM网关"""

    def __init__(self):
        self.provider = settings.llm_provider
        self._client = None

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> str:
        """生成回答"""
        if self.provider == "mock":
            return await self._mock_generate(prompt)
        elif self.provider == "deepseek":
            return await self._deepseek_generate(prompt, system_prompt, max_tokens, temperature)
        else:
            raise ValueError(f"不支持的LLM提供商: {self.provider}")

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
        if not settings.deepseek_api_key:
            raise ValueError("未配置DEEPSEEK_API_KEY")

        if self._client is None:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(
                api_key=settings.deepseek_api_key,
                base_url=settings.deepseek_base_url,
            )

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = await self._client.chat.completions.create(
                model=settings.deepseek_model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=settings.llm_timeout_seconds,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"DeepSeek API调用失败: {e}")
            raise


# 全局实例
llm_gateway = LLMGateway()
