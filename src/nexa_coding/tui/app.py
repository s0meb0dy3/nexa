"""NEXA 的最简 TUI（唯一依赖 Textual 的模块）。

三层分工：state.py 是纯状态，adapter.py 是事件→状态翻译，app.py 只负责
把 TuiState 画到屏幕上。worker 在后台跑 CodingSession.prompt()，事件流
翻译成状态变化后再刷新界面，UI 因此不卡顿。
"""

from __future__ import annotations

from pathlib import Path

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Footer, Header, Input, RichLog, Static
from textual.worker import Worker

from nexa_coding.session import CodingSession, CodingSessionConfig
from nexa_coding.tui.adapter import TuiEventAdapter
from nexa_coding.tui.state import ChatItemKind, TuiState


class NexaTuiApp(App):
    """最简交互式 coding agent 界面。

    用法：uv run nexa --tui
    """

    # 键盘绑定：Escape 中断当前运行。
    BINDINGS = [
        Binding("escape", "cancel", "取消"),
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
            self._refresh()

    # ── 键盘动作 ─────────────────────────────────────────────────────────

    def action_cancel(self) -> None:
        """Escape：取消当前运行。"""

        if self._current_worker is not None:
            self._current_worker.cancel()
            self._state.set_running(False)
            self._state.add_tool("运行", "已取消")
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
            else:
                prefix = "⚠" if item.error else "🔧"
                transcript.write(f"{prefix} {item.text}")
        transcript.scroll_end()

        if self._state.error:
            status.update(f"错误：{self._state.error}")
        elif self._state.running:
            status.update("运行中…（Escape 取消）")
        else:
            status.update("待机")


__all__ = ["NexaTuiApp"]
