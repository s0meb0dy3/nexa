"""用于测试的确定性假 Provider。"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Iterable

from nexa_agent.messages import AgentMessage
from nexa_agent.provider import ThinkingLevel
from nexa_agent.provider_events import ProviderEvent
from nexa_agent.tools import AgentTool


class FakeProvider:
    """按调用顺序播放预先准备好的 ProviderEvent。"""

    default_thinking: ThinkingLevel = "default"

    def __init__(self, streams: Iterable[Iterable[ProviderEvent]]) -> None:
        # 每次调用消费一组事件，便于测试多轮模型交互。
        self._streams = [list(stream) for stream in streams]
        self.calls: list[tuple[str, str, list[AgentMessage], list[AgentTool]]] = []

    def context_window(self, model: str) -> int | None:
        return None

    def thinking_options(self, model: str) -> tuple[ThinkingLevel, ...]:
        return ("default",)

    def with_thinking(self, model: str, level: ThinkingLevel) -> FakeProvider:
        if level != "default":
            raise ValueError(f"模型 {model} 不支持思考设置 {level}")
        return self

    def stream_response(
        self,
        *,
        model: str,
        system: str,
        messages: list[AgentMessage],
        tools: list[AgentTool],
    ) -> AsyncGenerator[ProviderEvent, None]:
        """记录请求，并返回下一组预设事件。"""

        self.calls.append((model, system, list(messages), list(tools)))
        stream = self._streams.pop(0) if self._streams else []

        async def iterator() -> AsyncGenerator[ProviderEvent, None]:
            for event in stream:
                yield event

        return iterator()


__all__ = ["FakeProvider"]
