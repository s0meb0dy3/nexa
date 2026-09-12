"""事件 → 状态翻译：把 AgentEvent 流翻译成 TuiState 的变化（不依赖 Textual）。

这是"事件驱动前端"的翻译层：每个 AgentEvent 对应 TuiState 的一次变化，
UI 只管渲染 TuiState。翻译层不碰界面，所以能脱离终端单独测试。
"""

from __future__ import annotations

from nexa_agent.events import (
    AgentEndEvent,
    AgentEvent,
    AgentStartEvent,
    MessageEndEvent,
    MessageStartEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolExecutionUpdateEvent,
)
from nexa_agent.messages import AssistantMessage
from nexa_coding.tui.state import TuiState


class TuiEventAdapter:
    """把 AgentEvent 流应用到 TuiState。"""

    def __init__(self, state: TuiState) -> None:
        """绑定要更新的状态对象。"""
        self._state = state

    def apply(self, event: AgentEvent) -> None:
        """处理一个事件，更新状态。

        单个事件处理失败不应拖垮整个 UI——异常记录到 state.error。
        """

        try:
            self._apply_one(event)
        except Exception as error:  # noqa: BLE001 - 渲染层容错，不让单个事件搞挂界面
            self._state.error = f"事件处理失败: {error}"

    # ── 内部实现 ──────────────────────────────────────────────────────────

    def _apply_one(self, event: AgentEvent) -> None:
        """按事件类型分派到 TuiState 的方法。"""

        if isinstance(event, AgentStartEvent):
            self._state.set_running(True)
        elif isinstance(event, AgentEndEvent):
            # 只停止运行状态；最终消息已经由 MessageEndEvent 合并，
            # 这里不反查 messages，避免重复显示。
            self._state.set_running(False)
        elif isinstance(event, MessageStartEvent):
            if event.message.role == "assistant" and event.message.text:
                self._state.start_assistant()
        elif isinstance(event, MessageEndEvent):
            # 整条消息粒度：消息到达时把思考、正文分别合并成记录。
            message = event.message
            if isinstance(message, AssistantMessage):
                # 先记思考（推理发生在正文之前），再记正文。
                if message.thinking:
                    self._state.add_thinking(message.thinking)
                if message.text:
                    self._state.end_assistant(message.text)
        elif isinstance(event, ToolExecutionStartEvent):
            self._state.add_tool(event.tool_name, "开始")
        elif isinstance(event, ToolExecutionUpdateEvent):
            self._state.add_tool(event.tool_name, "更新")
        elif isinstance(event, ToolExecutionEndEvent):
            status = "失败" if event.is_error else "结束"
            self._state.add_tool(event.tool_name, status, error=event.is_error)
        # TurnStartEvent / TurnEndEvent 对显示无意义，忽略。


__all__ = ["TuiEventAdapter"]
