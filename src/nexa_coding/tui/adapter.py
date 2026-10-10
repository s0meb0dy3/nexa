"""事件 → 状态翻译：把 AgentEvent 流翻译成 TuiState 的变化（不依赖 Textual）。

这是"事件驱动前端"的翻译层：每个 AgentEvent 对应 TuiState 的一次变化，
UI 只管渲染 TuiState。翻译层不碰界面，所以能脱离终端单独测试。
"""

from __future__ import annotations

from nexa_agent.events import (
    AgentEndEvent,
    AgentEvent,
    AgentStartEvent,
    MessageDeltaEvent,
    MessageEndEvent,
    MessageStartEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolExecutionUpdateEvent,
)
from nexa_agent.messages import AssistantMessage, ToolResultMessage
from nexa_coding.tui.state import TuiState


class TuiEventAdapter:
    """把 AgentEvent 流应用到 TuiState。"""

    def __init__(self, state: TuiState) -> None:
        """绑定要更新的状态对象。"""
        self._state = state

    def apply(self, event: AgentEvent) -> None:
        """只转换显示状态；异常交给 app 的运行边界报告。"""

        if isinstance(event, AgentStartEvent):
            self._state.set_running(True)
        elif isinstance(event, AgentEndEvent):
            # 只停止运行状态；最终消息已经由 MessageEndEvent 合并，
            # 这里不反查 messages，避免重复显示。
            self._state.set_running(False)
        elif isinstance(event, MessageStartEvent):
            # 助手开始输出（流式时携带空消息）：进入流式状态准备累积。
            if event.message.role == "assistant":
                self._state.start_assistant()
        elif isinstance(event, MessageDeltaEvent):
            # 流式片段：按 kind 分别追加到正文 / 思考缓冲。
            if event.kind == "reasoning":
                self._state.append_thinking_delta(event.delta)
            else:
                self._state.append_assistant_delta(event.delta)
        elif isinstance(event, MessageEndEvent):
            # 完整消息到达：以它为准提交思考与正文，并重置流式状态。
            message = event.message
            if isinstance(message, AssistantMessage):
                if message.usage is not None:
                    self._state.usage = message.usage
                # 先记思考（推理发生在正文之前），再记正文。
                if message.thinking:
                    self._state.add_thinking(message.thinking)
                self._state.end_assistant(message.text)
            elif isinstance(message, ToolResultMessage):
                self._state.update_tool(
                    message.tool_call_id,
                    message.tool_name,
                    "失败" if message.is_error else "结束",
                    output=message.text,
                    error=message.is_error,
                )
        elif isinstance(event, ToolExecutionStartEvent):
            self._state.update_tool(event.tool_call_id, event.tool_name, "执行中", args=event.args)
        elif isinstance(event, ToolExecutionUpdateEvent):
            self._state.update_tool(
                event.tool_call_id,
                event.tool_name,
                "执行中",
                args=event.args,
                output=event.partial_result.text,
            )
        elif isinstance(event, ToolExecutionEndEvent):
            status = "失败" if event.is_error else "结束"
            self._state.update_tool(
                event.tool_call_id,
                event.tool_name,
                status,
                output=event.result.text,
                error=event.is_error,
            )
        # TurnStartEvent / TurnEndEvent 对显示无意义，忽略。


__all__ = ["TuiEventAdapter"]
