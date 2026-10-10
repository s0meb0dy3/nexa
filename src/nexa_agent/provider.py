"""模型 Provider 的统一接口，由可移植的 agent 层拥有。

核心只依赖这个协议，不依赖任何具体 Provider 实现；具体实现在
nexa_ai 里（如 OpenAICompatibleProvider）。这样依赖方向单向：
nexa_ai → nexa_agent。
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Literal, Protocol

from nexa_agent.messages import AgentMessage
from nexa_agent.provider_events import ProviderEvent
from nexa_agent.tools import AgentTool

type ThinkingLevel = Literal["default", "off", "on", "low", "medium", "high", "max"]


class ModelProvider(Protocol):
    """所有模型后端都要实现的最小接口。"""

    default_thinking: ThinkingLevel

    def context_window(self, model: str) -> int | None:
        """模型官方上下文上限；未知时返回 None。"""
        ...

    def thinking_options(self, model: str) -> tuple[ThinkingLevel, ...]:
        """当前模型支持的设置；default 表示不指定 API 参数。"""
        ...

    def with_thinking(self, model: str, level: ThinkingLevel) -> ModelProvider:
        """验证设置并返回候选 Provider，不修改当前实例。"""
        ...

    def stream_response(
        self,
        *,
        model: str,
        system: str,
        messages: list[AgentMessage],
        tools: list[AgentTool],
    ) -> AsyncGenerator[ProviderEvent, None]:
        """返回一次模型响应的异步事件流。"""

        ...


__all__ = ["ModelProvider", "ThinkingLevel"]
