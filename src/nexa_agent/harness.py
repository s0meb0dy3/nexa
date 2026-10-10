"""管理当前对话历史与运行状态；完整消息先通知监听器，再交给界面。"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import aclosing
from dataclasses import dataclass, replace
from inspect import isawaitable
from typing import Protocol

from nexa_agent.events import AgentEvent, MessageEndEvent
from nexa_agent.loop import AgentLoop
from nexa_agent.messages import AgentMessage, UserMessage
from nexa_agent.provider import ModelProvider
from nexa_agent.tools import AgentTool


class EventListener(Protocol):
    """监听器可同步观察事件，也可异步完成保存等工作。"""

    def on_event(self, event: AgentEvent) -> Awaitable[None] | None: ...


@dataclass(frozen=True)
class AgentHarnessConfig:
    provider: ModelProvider
    model: str
    system: str = ""
    tools: tuple[AgentTool, ...] = ()
    max_turns: int = 10


class AgentHarness:
    def __init__(
        self, config: AgentHarnessConfig, *, messages: list[AgentMessage] | None = None
    ) -> None:
        self._config = config
        self._messages = list(messages or [])
        self._listeners: list[EventListener] = []
        self._is_running = False
        self._cancel_requested = False

    @property
    def model(self) -> str:
        return self._config.model

    @property
    def messages(self) -> tuple[AgentMessage, ...]:
        return tuple(self._messages)

    @property
    def is_running(self) -> bool:
        return self._is_running

    def select_model(self, provider: ModelProvider, model: str) -> None:
        if self._is_running:
            raise RuntimeError("运行中不能切换模型")
        self._config = replace(self._config, provider=provider, model=model)

    def append_message(self, message: AgentMessage) -> None:
        self._messages.append(message)

    def replace_messages(self, messages: list[AgentMessage]) -> None:
        self._messages = list(messages)

    def subscribe(self, listener: EventListener) -> Callable[[], None]:
        self._listeners.append(listener)

        def unsubscribe() -> None:
            self._listeners.remove(listener)

        return unsubscribe

    def cancel(self) -> None:
        """在下一事件边界停止；TUI 的 Esc 仍使用任务取消中断等待。"""
        self._cancel_requested = True

    async def prompt(self, content: str) -> AsyncGenerator[AgentEvent, None]:
        async with aclosing(self._run_loop(UserMessage(content=content))) as stream:
            async for event in stream:
                yield event

    async def continue_(self) -> AsyncGenerator[AgentEvent, None]:
        async with aclosing(self._run_loop()) as stream:
            async for event in stream:
                yield event

    async def _notify(self, event: AgentEvent) -> None:
        if isinstance(event, MessageEndEvent):
            self._messages.append(event.message)
        # 按订阅顺序等待；保存失败必须向上传递，不能显示成成功。
        for listener in tuple(self._listeners):
            result = listener.on_event(event)
            if isawaitable(result):
                await result

    async def _run_loop(
        self, user_message: UserMessage | None = None
    ) -> AsyncGenerator[AgentEvent, None]:
        if self._is_running:
            raise RuntimeError("Harness 已经在运行中，不能并发调用 prompt/continue")
        self._is_running = True
        self._cancel_requested = False
        try:
            if user_message is not None:
                # 用户输入也走完整消息边界，API 请求前便已保存。
                event: AgentEvent = MessageEndEvent(message=user_message)
                await self._notify(event)
                yield event
            loop = AgentLoop(self._config.provider, max_turns=self._config.max_turns)
            async with aclosing(
                loop.run(
                    model=self.model,
                    system=self._config.system,
                    messages=self._messages,
                    tools=list(self._config.tools),
                )
            ) as stream:
                async for event in stream:
                    if self._cancel_requested:
                        break
                    await self._notify(event)
                    yield event
        finally:
            self._is_running = False


__all__ = ["AgentHarness", "AgentHarnessConfig", "EventListener"]
