"""TUI 的纯显示状态（不依赖 Textual）。

这里只描述"界面上该显示什么"，不关心怎么渲染。Textual 渲染层
（app.py）读这个状态来画界面，测试也可以直接断言它——这就是
逻辑层和渲染层解耦的意义。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from time import monotonic


class ChatItemKind(StrEnum):
    """一条聊天记录的类别。"""

    # 用户输入
    user = "user"
    # 助手回复
    assistant = "assistant"
    # 思考内容（推理模型的推理过程）
    thinking = "thinking"
    # 工具执行
    tool = "tool"


@dataclass
class ChatItem:
    """一条聊天记录。

    Attributes:
        kind: 记录类别（user / assistant / tool）。
        text: 显示文本。
        error: 是否表示失败（工具失败时 True）。
    """

    kind: ChatItemKind
    text: str
    error: bool = False
    tool_call_id: str | None = None
    output: str = ""
    started_at: float | None = None
    elapsed: float | None = None
    status: str = ""
    historical: bool = False


@dataclass
class TuiState:
    """TUI 界面的纯状态。"""

    # 全部聊天记录，按顺序渲染。
    chat_items: list[ChatItem] = field(default_factory=list)
    # 当前正在流式输出的正文（尚未提交成 ChatItem）。
    streaming_text: str = ""
    # 当前正在流式输出的思考（尚未提交成 ChatItem）。
    streaming_thinking: str = ""
    # 是否正处于"助手消息输出中"（MessageStart 之后、MessageEnd 之前）。
    streaming_started: bool = False
    # agent 是否正在运行。
    running: bool = False
    # 最近一次错误，None 表示无错误。
    error: str | None = None
    # 是否展开显示思考内容；默认折叠（推理往往很长）。
    show_thinking: bool = False

    # ── 状态修改方法 ──────────────────────────────────────────────────────

    def add_user(self, text: str) -> None:
        """记录一条用户消息。"""

        self.chat_items.append(ChatItem(kind=ChatItemKind.user, text=text))

    def start_assistant(self) -> None:
        """助手开始回复：清空流式缓冲，进入输出中状态。"""

        self.streaming_text = ""
        self.streaming_thinking = ""
        self.streaming_started = True

    def append_assistant_delta(self, delta: str) -> None:
        """追加一段正文流式文本。"""

        self.streaming_text += delta

    def append_thinking_delta(self, delta: str) -> None:
        """追加一段思考流式文本。"""

        self.streaming_thinking += delta

    def end_assistant(self, text: str) -> None:
        """助手回复结束：提交整条消息，并重置流式状态。

        text 为空（例如只调用工具、没有文字）时只重置状态，不产生记录。
        """

        self.streaming_text = ""
        self.streaming_thinking = ""
        self.streaming_started = False
        if text:
            self.chat_items.append(ChatItem(kind=ChatItemKind.assistant, text=text))

    def add_tool(self, name: str, status: str, *, error: bool = False) -> None:
        """记录一条工具执行消息。"""

        self.chat_items.append(
            ChatItem(kind=ChatItemKind.tool, text=f"{name} — {status}", error=error)
        )

    def update_tool(
        self,
        call_id: str,
        name: str,
        status: str,
        *,
        args: dict | None = None,
        output: str = "",
        error: bool = False,
    ) -> None:
        item = next((i for i in self.chat_items if i.tool_call_id == call_id), None)
        if item is None:
            detail = (args or {}).get("path") or (args or {}).get("command") or ""
            item = ChatItem(
                kind=ChatItemKind.tool,
                text=f"{name}  {detail}",
                tool_call_id=call_id,
                started_at=monotonic(),
            )
            self.chat_items.append(item)
        item.status = status
        item.error = error
        item.output = output
        if status in ("结束", "失败", "已中断"):
            item.elapsed = monotonic() - item.started_at

    def interrupt_tools(self) -> None:
        for item in self.chat_items:
            if item.tool_call_id and item.elapsed is None:
                item.status = "已中断"
                item.elapsed = monotonic() - item.started_at

    def add_thinking(self, text: str) -> None:
        """记录一条思考内容。"""

        self.chat_items.append(ChatItem(kind=ChatItemKind.thinking, text=text))

    def toggle_thinking(self) -> None:
        """切换思考内容的展开/折叠。"""

        self.show_thinking = not self.show_thinking

    def set_running(self, running: bool) -> None:
        """更新运行状态。"""

        self.running = running


__all__ = ["ChatItem", "ChatItemKind", "TuiState"]
