"""transcript 模式渲染器：过程 + 结果都显示。

助手消息 → stdout；工具开始/结束/失败 → stderr（过程细节不污染 stdout）。
这是最接近人类看的"运行过程实录"。
"""

from __future__ import annotations

from typing import TextIO

from nexa_agent.events import (
    AgentEndEvent,
    AgentEvent,
    MessageEndEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
)
from nexa_agent.messages import AssistantMessage


class TranscriptRenderer:
    """把工具过程打到 stderr，最终答案打到 stdout。"""

    def __init__(self, *, stdout: TextIO, stderr: TextIO) -> None:
        """指定两个输出流：stdout 放结果，stderr 放过程。"""
        self._stdout = stdout
        self._stderr = stderr
        self._final_answer = ""
        self._failed = False

    def render(self, event: AgentEvent) -> None:
        """分流事件：助手答案存起来，思考与工具过程立即打印。"""

        if isinstance(event, ToolExecutionStartEvent):
            print(f"工具开始：{event.tool_name}", file=self._stderr)
        elif isinstance(event, ToolExecutionEndEvent):
            prefix = "工具失败" if event.is_error else "工具结束"
            print(f"{prefix}：{event.tool_name}", file=self._stderr)
            self._failed |= event.is_error
        elif isinstance(event, MessageEndEvent) and isinstance(event.message, AssistantMessage):
            # 思考属于"过程"，打到 stderr，不污染 stdout 的最终答案。
            if event.message.thinking:
                print(f"思考：{event.message.thinking}", file=self._stderr)
        elif isinstance(event, AgentEndEvent):
            # 从后往前找第一条有文字的助手消息，作为最终答案。
            for message in reversed(event.messages):
                if message.role == "assistant" and message.text:
                    self._final_answer = message.text
                    break

    def finish(self) -> bool:
        """打印最终答案，返回是否成功（工具失败过则算失败）。"""

        if self._failed:
            return False
        if not self._final_answer or self._final_answer.startswith("错误: "):
            return False
        print(self._final_answer, file=self._stdout)
        return True


__all__ = ["TranscriptRenderer"]
