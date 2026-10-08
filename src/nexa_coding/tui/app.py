"""NEXA 的最简 TUI（唯一依赖 Textual 的模块）。

三层分工：state.py 是纯状态，adapter.py 是事件→状态翻译，app.py 只负责
把 TuiState 画到屏幕上。worker 在后台跑 CodingSession.prompt()，事件流
翻译成状态变化后再刷新界面，UI 因此不卡顿。
"""

from __future__ import annotations

import time
from pathlib import Path

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Footer, Header, Input, RichLog, Static
from textual.worker import Worker

from nexa_agent.events import MessageDeltaEvent
from nexa_coding.session import CodingSession, CodingSessionConfig
from nexa_coding.tui.adapter import TuiEventAdapter
from nexa_coding.tui.state import ChatItemKind, TuiState

# 流式刷新节流间隔（秒）：delta 来得很快，但界面没必要每个字都重画一次，
# 否则全量重绘会退化成 O(n²) 并产生闪烁。MessageEnd 时会强制刷新收尾。
_STREAM_REFRESH_INTERVAL = 0.04


class NexaTuiApp(App):
    """最简交互式 coding agent 界面。

    用法：nexa（或在源码目录运行 uv run nexa）
    """

    # 键盘绑定：Escape 中断当前运行，Ctrl+T 展开/折叠思考。
    BINDINGS = [
        Binding("escape", "cancel", "取消"),
        Binding("ctrl+t", "toggle_thinking", "思考"),
        Binding("ctrl+q", "quit", "退出"),
    ]

    # 简单布局：Header / 对话记录 / 运行状态 / 输入框 / Footer。
    CSS = """
    Screen {
        layout: vertical;
    }
    #transcript {
        height: 1fr;
        border: solid $accent;
        padding: 0 1;
    }
    #status {
        height: 1;
        content-align: left middle;
    }
    #prompt-input {
        dock: bottom;
    }
    """

    def __init__(self, provider, *, model: str, cwd: Path) -> None:
        """初始化 TUI。

        Args:
            provider: 模型 Provider（通常来自 resolve_provider()）。
            model: 模型名（由配置档案解析得到，必填）。
            cwd: 工具可访问的项目目录，也是会话隔离的依据。
        """
        super().__init__()
        self._provider = provider
        self._model = model
        self._cwd = cwd
        # 纯状态 + 翻译层（都不碰 Textual，可单独测试）。
        self._state = TuiState()
        self._adapter = TuiEventAdapter(self._state)
        # 当前正在运行的 worker，用于 Escape 取消。
        self._current_worker: Worker | None = None
        # 上次流式刷新的时刻（单调时钟），用于节流。
        self._last_refresh = 0.0

    def compose(self) -> ComposeResult:
        """搭建界面组件。"""
        yield Header()
        yield RichLog(id="transcript", wrap=True, markup=False)
        yield Static("待机", id="status")
        yield Input(placeholder="输入你的问题，回车运行，Escape 取消", id="prompt-input")
        yield Footer()

    # ── 输入提交：起 worker 跑一轮 ──────────────────────────────────────

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """用户回车：把输入交给 agent 跑一轮。"""

        text = event.value.strip()
        if not text or self._state.running:
            return
        event.input.value = ""

        # 记录用户消息，立即可见。
        self._state.add_user(text)
        self._refresh()

        # 起 worker 在后台跑，避免阻塞 UI。
        # @work 让方法调用立即返回 Worker 并启动，无需再用 run_worker 包一层。
        self._current_worker = self._run_prompt(text)

    # ── 后台任务 ────────────────────────────────────────────────────────

    @work(exclusive=True)
    async def _run_prompt(self, text: str) -> None:
        """在 worker 里跑一轮 CodingSession.prompt()，把事件翻译成状态。"""

        session = CodingSession.load(
            CodingSessionConfig(provider=self._provider, model=self._model, cwd=self._cwd)
        )

        async for event in session.prompt(text):
            self._adapter.apply(event)
            # 流式 delta 用节流刷新（字太快，没必要每个都重画）；
            # 其他事件（尤其是收尾的 message_end）立即刷新。
            if isinstance(event, MessageDeltaEvent):
                self._refresh_throttled()
            else:
                self._refresh()

    # ── 刷新 ─────────────────────────────────────────────────────────────

    def _refresh_throttled(self) -> None:
        """按固定间隔节流刷新，避免每个 token 都全量重绘。"""

        now = time.monotonic()
        if now - self._last_refresh >= _STREAM_REFRESH_INTERVAL:
            self._last_refresh = now
            self._refresh()

    # ── 键盘动作 ─────────────────────────────────────────────────────────

    def action_cancel(self) -> None:
        """Escape：取消当前运行。"""

        if self._current_worker is not None:
            self._current_worker.cancel()
            self._state.set_running(False)
            self._state.add_tool("运行", "已取消")
            self._refresh()

    def action_toggle_thinking(self) -> None:
        """Ctrl+T：展开/折叠所有思考内容。"""

        self._state.toggle_thinking()
        self._refresh()

    # ── 刷新 ─────────────────────────────────────────────────────────────

    def _refresh(self) -> None:
        """把 TuiState 重新画到界面上。"""

        transcript = self.query_one("#transcript", RichLog)
        status = self.query_one("#status", Static)

        # 清空重画（消息量小，简单实现即可）。
        transcript.clear()
        for item in self._state.chat_items:
            if item.kind == ChatItemKind.user:
                transcript.write(f"你：{item.text}")
            elif item.kind == ChatItemKind.assistant:
                transcript.write(f"🤖 {item.text}")
            elif item.kind == ChatItemKind.thinking:
                # 思考默认折叠成一行摘要，Ctrl+T 展开看全文。
                if self._state.show_thinking:
                    transcript.write(f"💭 {item.text}")
                else:
                    transcript.write(f"💭 思考了 {len(item.text)} 字（Ctrl+T 展开）")
            else:
                prefix = "⚠" if item.error else "🔧"
                transcript.write(f"{prefix} {item.text}")

        # 正在进行中的流式内容（尚未提交）：思考在前、正文在后。
        if self._state.streaming_started:
            if self._state.streaming_thinking:
                if self._state.show_thinking:
                    transcript.write(f"💭 {self._state.streaming_thinking}")
                else:
                    transcript.write(
                        f"💭 思考中… {len(self._state.streaming_thinking)} 字（Ctrl+T 展开）"
                    )
            if self._state.streaming_text:
                transcript.write(f"🤖 {self._state.streaming_text}")

        transcript.scroll_end()

        if self._state.error:
            status.update(f"错误：{self._state.error}")
        elif self._state.running:
            status.update("运行中…（Escape 取消）")
        else:
            status.update("待机")


__all__ = ["NexaTuiApp"]
