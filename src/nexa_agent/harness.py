"""AgentHarness：把 AgentLoop 包装成可持续对话的"agent brain"。

上层不再直接操作 AgentLoop.run() 和对话列表，而是通过 Harness 的接口：
- prompt() 发送用户消息并等待回复
- continue_() 基于当前历史继续运行
- subscribe() 订阅事件流（用于日志、持久化、TUI）
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Protocol

from nexa_agent.events import AgentEndEvent, AgentEvent
from nexa_agent.loop import AgentLoop
from nexa_agent.messages import AgentMessage, UserMessage
from nexa_agent.provider import ModelProvider
from nexa_agent.tools import AgentTool

# ── 事件监听器协议 ────────────────────────────────────────────────────────────


class EventListener(Protocol):
    """事件监听器接口：任何实现了 on_event 方法的对象都可以作为监听器。"""

    def on_event(self, event: AgentEvent) -> None:
        """收到一个事件时调用。"""
        ...


# ── 配置类 ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class AgentHarnessConfig:
    """AgentHarness 的配置。

    Attributes:
        provider: 模型 Provider（DeepSeek、OpenAI 等）
        model: 模型名称，如 "deepseek-chat"
        system: 系统提示词
        tools: 可用工具列表
        max_turns: 单次 prompt/continue 的最大轮次，默认 10
    """

    provider: ModelProvider
    model: str
    system: str = ""
    tools: tuple[AgentTool, ...] = ()
    max_turns: int = 10


# ── Harness 主类 ──────────────────────────────────────────────────────────────


class AgentHarness:
    """可持续对话的 Agent 封装。

    职责：
    - 管理对话历史（messages）
    - 封装 AgentLoop 的调用
    - 提供事件订阅机制
    - 防止并发运行
    - 支持取消操作

    使用示例：
        config = AgentHarnessConfig(provider=..., model="deepseek-chat")
        harness = AgentHarness(config)

        # 发送消息并等待回复
        async for event in harness.prompt("你好"):
            print(event)

        # 继续对话（不添加用户消息）
        async for event in harness.continue_():
            print(event)

        # 订阅事件（用于日志、持久化）
        unsubscribe = harness.subscribe(my_listener)
    """

    def __init__(
        self,
        config: AgentHarnessConfig,
        *,
        messages: list[AgentMessage] | None = None,
    ) -> None:
        """初始化 Harness。

        Args:
            config: 配置（provider、model、tools 等）
            messages: 初始消息列表（用于会话恢复），默认为空
        """
        self._config = config
        # 内部保存可变对话历史
        self._messages: list[AgentMessage] = list(messages) if messages else []
        # 事件监听器集合
        self._listeners: set[EventListener] = set()
        # 当前是否正在运行
        self._is_running: bool = False
        # 取消标志
        self._cancel_requested: bool = False
        # 当前运行的任务（用于取消）
        self._current_task: asyncio.Task | None = None

    # ── 属性 ──────────────────────────────────────────────────────────────────

    @property
    def messages(self) -> tuple[AgentMessage, ...]:
        """只读返回当前对话历史。

        返回 tuple 而不是 list，防止外部直接修改内部状态。
        """
        return tuple(self._messages)

    @property
    def is_running(self) -> bool:
        """当前是否正在运行 prompt() 或 continue_()。"""
        return self._is_running

    # ── 核心方法 ──────────────────────────────────────────────────────────────

    async def prompt(self, content: str) -> AsyncIterator[AgentEvent]:
        """发送用户消息并运行 Agent 循环。

        Args:
            content: 用户消息内容

        Yields:
            AgentEvent: 循环过程中产生的事件

        Raises:
            RuntimeError: 如果已经在运行中
        """
        if self._is_running:
            raise RuntimeError("Harness 已经在运行中，不能并发调用 prompt/continue")

        # 重置取消标志
        self._cancel_requested = False

        # 添加用户消息到历史
        self._messages.append(UserMessage(content=content))

        # 运行循环
        async for event in self._run_loop():
            yield event

    async def continue_(self) -> AsyncIterator[AgentEvent]:
        """基于当前历史继续运行 Agent 循环（不添加用户消息）。

        用于：
        - 工具执行后让 Agent 继续
        - 恢复会话后继续对话

        Yields:
            AgentEvent: 循环过程中产生的事件

        Raises:
            RuntimeError: 如果已经在运行中
        """
        if self._is_running:
            raise RuntimeError("Harness 已经在运行中，不能并发调用 prompt/continue")

        # 重置取消标志
        self._cancel_requested = False

        # 运行循环
        async for event in self._run_loop():
            yield event

    def cancel(self) -> None:
        """请求取消当前运行。

        设置取消标志，循环会在下一个事件检查点退出。
        不会立即中断，而是让循环安全地结束。
        """
        self._cancel_requested = True

    # ── 会话恢复方法 ──────────────────────────────────────────────────────────

    def append_message(self, message: AgentMessage) -> None:
        """追加一条消息到历史（用于会话恢复）。

        Args:
            message: 要追加的消息
        """
        self._messages.append(message)

    def replace_messages(self, messages: list[AgentMessage]) -> None:
        """替换整个对话历史（用于会话恢复）。

        Args:
            messages: 新的消息列表
        """
        self._messages = list(messages)

    # ── 事件订阅 ──────────────────────────────────────────────────────────────

    def subscribe(self, listener: EventListener) -> Callable[[], None]:
        """注册事件监听器。

        Args:
            listener: 实现了 on_event 方法的对象

        Returns:
            取消订阅的函数，调用后不再收到事件

        使用示例：
            unsubscribe = harness.subscribe(my_listener)
            # ... 收到事件 ...
            unsubscribe()  # 取消订阅
        """
        self._listeners.add(listener)

        def unsubscribe() -> None:
            self._listeners.discard(listener)

        return unsubscribe

    # ── 内部方法 ──────────────────────────────────────────────────────────────

    async def _run_loop(self) -> AsyncIterator[AgentEvent]:
        """运行 AgentLoop，并通知所有监听器。"""

        # 标记为运行中
        self._is_running = True

        try:
            # 创建 AgentLoop 实例
            loop = AgentLoop(self._config.provider, max_turns=self._config.max_turns)

            # 运行循环
            async for event in loop.run(
                model=self._config.model,
                system=self._config.system,
                messages=self._messages,
                tools=list(self._config.tools),
            ):
                # 检查取消标志
                if self._cancel_requested:
                    break

                # 通知所有监听器
                for listener in self._listeners:
                    with contextlib.suppress(Exception):
                        # 监听器出错不应该影响主流程
                        listener.on_event(event)

                # 从 AgentEndEvent 中更新历史
                if isinstance(event, AgentEndEvent):
                    self._messages = list(event.messages)

                # 产出事件给主调用者
                yield event

        finally:
            # 无论成功还是失败，都标记为未运行
            self._is_running = False
            self._current_task = None


__all__ = ["AgentHarness", "AgentHarnessConfig", "EventListener"]
