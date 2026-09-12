"""TUI 的纯显示状态（不依赖 Textual）。

这里只描述"界面上该显示什么"，不关心怎么渲染。Textual 渲染层
（app.py）读这个状态来画界面，测试也可以直接断言它——这就是
逻辑层和渲染层解耦的意义。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


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


@dataclass
class TuiState:
    """TUI 界面的纯状态。"""

    # 全部聊天记录，按顺序渲染。
    chat_items: list[ChatItem] = field(default_factory=list)
    # 当前正在流式输出的文本（Phase 12 不用，保留给将来真流式）。
    streaming_text: str = ""
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
        """助手开始回复：清空流式缓冲，等待消息结束合并。"""

        self.streaming_text = ""

    def append_assistant_delta(self, delta: str) -> None:
        """追加一段流式文本（Phase 12 不调用，供将来真流式用）。"""

        self.streaming_text += delta

    def end_assistant(self, text: str) -> None:
        """助手回复结束：把整条消息合并进聊天记录。"""

        self.streaming_text = ""
        self.chat_items.append(ChatItem(kind=ChatItemKind.assistant, text=text))

    def add_tool(self, name: str, status: str, *, error: bool = False) -> None:
        """记录一条工具执行消息。"""

        self.chat_items.append(
            ChatItem(kind=ChatItemKind.tool, text=f"{name} — {status}", error=error)
        )

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
