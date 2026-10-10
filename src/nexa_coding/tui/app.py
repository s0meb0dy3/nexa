"""Textual 界面：组件保留在屏幕上，只更新当前变化的消息。"""

from __future__ import annotations

import asyncio
import contextlib
import time
from datetime import datetime
from pathlib import Path

from rich.markdown import Markdown as RichMarkdown
from rich.text import Text
from textual import events, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, VerticalScroll
from textual.message import Message
from textual.widgets import Collapsible, Markdown, OptionList, Static, TextArea
from textual.widgets.option_list import Option
from textual.worker import Worker, WorkerCancelled

from nexa_agent.events import MessageDeltaEvent
from nexa_agent.session.storage import JsonlStorage
from nexa_coding.commands import COMMANDS, completions, parse_command
from nexa_coding.config import load_config
from nexa_coding.paths import NexaPaths
from nexa_coding.providers import build_provider
from nexa_coding.session import CodingSession, CodingSessionConfig
from nexa_coding.session_manager import SessionManager
from nexa_coding.tui.adapter import TuiEventAdapter
from nexa_coding.tui.dialogs import ChoiceScreen
from nexa_coding.tui.state import ChatItemKind, TuiState

_STREAM_REFRESH_INTERVAL = 0.04


class PromptInput(TextArea):
    """保留 TextArea 的编辑和粘贴能力，只替换提交／换行按键。"""

    class Submitted(Message):
        def __init__(self, text: str) -> None:
            super().__init__()
            self.text = text

    async def _on_key(self, event: events.Key) -> None:
        if event.key == "enter":
            event.stop()
            event.prevent_default()
            self.post_message(self.Submitted(self.text))
        elif event.key in ("tab", "up", "down") and self.app._completion_items:
            event.stop()
            event.prevent_default()
            self.app.action_completion(event.key)
        elif event.key in ("shift+enter", "ctrl+j"):
            event.stop()
            event.prevent_default()
            self.insert("\n")
        else:
            await super()._on_key(event)


class ToolDetail(Collapsible):
    """展开结果时保持阅读位置，避免失败事件把用户拉到屏幕底部。"""

    def _watch_collapsed(self, collapsed: bool) -> None:
        self._update_collapsed(collapsed)
        self.post_message(self.Collapsed(self) if collapsed else self.Expanded(self))


class NexaTuiApp(App):
    BINDINGS = [
        Binding("escape", "cancel", "取消", priority=True, show=False),
        Binding("ctrl+t", "toggle_thinking", "展开思考", show=False),
        Binding("ctrl+end", "jump_bottom", "到底部", priority=True, show=False),
        Binding("ctrl+q", "quit", "退出", show=False),
    ]

    CSS = """
    Screen { layout: vertical; background: #0d1418; color: #d8e2e7; }
    #transcript { height: 1fr; padding: 1 2; }
    #transcript > Markdown { margin: 0 0 1 0; padding: 0; }
    .user-message { height: auto; border-left: thick #79d6cf; padding: 0 1;
        margin: 1 0; color: #a7e6e0; }
    .notice { height: auto; margin-bottom: 1; color: #8fa5b2; }
    .thinking { height: auto; margin-bottom: 1; color: #8fa5b2; }
    .tool { height: auto; margin-bottom: 1; padding: 0; background: #142026;
        border: none; }
    .tool-output { height: auto; padding: 0 1 1 1; }
    .failed { border-left: thick #eb9a89; }
    #status { height: 1; margin: 0 2; color: #8fa5b2; }
    #completions { height: auto; max-height: 6; margin: 0 2; border: none; background: #142026; }
    #prompt-input { height: 3; max-height: 8; margin: 1 2 0 2;
        border: solid #38565e; background: #142026; }
    #input-meta { height: 1; margin: 0 2; }
    #input-help { height: 1; width: auto; max-width: 50%; margin-left: 2; color: #8fa5b2; }
    #environment { height: 1; width: 1fr; color: #79d6cf; }
    #context-usage { height: 1; width: auto; margin-left: 2; color: #79d6cf; }
    """

    def __init__(self, provider, *, model: str, cwd: Path, paths: NexaPaths | None = None) -> None:
        super().__init__()
        self._provider = provider
        self._model = model
        self._cwd = cwd
        self._paths = paths or NexaPaths()
        self._sessions = SessionManager(
            self._paths.project_session_dir(cwd.resolve()), config_file=self._paths.config_file
        )
        self._session_path = self._sessions.current_path()
        self._session: CodingSession | None = None
        self._command_busy = False
        self._completion_items: list[str] = []
        self._branch = ""
        self._state = TuiState()
        self._adapter = TuiEventAdapter(self._state)
        self._current_worker: Worker | None = None
        self._last_refresh = 0.0
        # 每条完成的消息对应一个组件；缓存内容避免重复重排 Markdown。
        self._item_widgets: list = []
        self._item_versions: dict[int, tuple] = {}
        self._tool_outputs: dict[int, Static] = {}
        self._stream_text = ""
        self._stream_reasoning = ""

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="transcript"):
            yield Static("", id="stream-thinking", classes="thinking", markup=False)
            # 流式预览保留一个组件；Markdown.update 会拆除代码块子组件，
            # 与 Textual 的鼠标文本选择产生竞争。完整回复仍使用 Markdown。
            yield Static("", id="stream-answer", markup=False)
        yield Static("", id="status", markup=False)
        yield OptionList(id="completions")
        yield PromptInput(
            placeholder="Enjoy your coding journey! 输入 /help 查看命令",
            id="prompt-input",
            show_line_numbers=False,
            highlight_cursor_line=False,
        )
        with Horizontal(id="input-meta"):
            yield Static("", id="environment", markup=False)
            yield Static("", id="context-usage", markup=False)
            yield Static("Enter 发送 · Shift+Enter 换行", id="input-help")

    @property
    def _view(self):
        # 弹窗在栈顶，后台刷新仍然指向主界面。
        return self.screen_stack[0]

    def on_mount(self) -> None:
        self._view.query_one(PromptInput).focus()
        self._view.query_one("#transcript", VerticalScroll).anchor()
        self._view.query_one("#completions").display = False
        self._view.query_one("#status").display = False
        self._view.query_one("#stream-thinking").display = False
        self._view.query_one("#stream-answer").display = False
        self._refresh_environment()
        self._load_branch()
        self.set_interval(0.2, self._tick)
        if self._session_path.exists():
            self._command_busy = True
            self._restore_startup()

    @work(group="environment", exclusive=True)
    async def _load_branch(self) -> None:
        proc = None
        try:
            proc = await asyncio.create_subprocess_exec(
                "git",
                "rev-parse",
                "--abbrev-ref",
                "HEAD",
                cwd=self._cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), 2)
            if proc.returncode == 0:
                self._branch = stdout.decode(errors="replace").strip()
        except (OSError, TimeoutError):
            pass
        finally:
            if proc is not None and proc.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    proc.kill()
                await proc.wait()
        self._refresh_environment()

    def _refresh_environment(self) -> None:
        if not self._view.query("#environment"):
            return
        path = str(self._cwd)
        home = str(Path.home())
        if path.startswith(home + "/"):
            path = "~" + path[len(home) :]
        provider = getattr(self._provider, "name", type(self._provider).__name__)
        thinking = self._session.thinking if self._session else self._provider.default_thinking
        label = "模型默认" if thinking == "default" else thinking
        model = f"{provider} / {self._model} · 思考 {label}"
        self._view.query_one("#context-usage", Static).update(self._context_status())
        # 窄窗口优先保留实测统计，快捷键仍可通过 /help 查看。
        self._view.query_one("#input-help").display = self.size.width >= 160
        branch = f" · {self._branch}" if self._branch else ""
        branch += f" · 会话 {self._session_path.stem[:8]}"
        width = self._view.query_one("#environment").size.width
        budget = max(12, width - len(model) - len(branch) - 3)
        if len(path) > budget:
            path = "…" + path[-(budget - 1) :]
        self._view.query_one("#environment", Static).update(f"{path}{branch}   {model}")

    def _context_status(self, *, detailed: bool = False) -> str:
        usage = self._state.usage
        limit = self._provider.context_window(self._model)
        if not detailed:
            if usage is None:
                return "上下文 —"

            def compact(tokens: int) -> str:
                if tokens >= 1_000_000:
                    return f"{tokens / 1_000_000:.1f}".rstrip("0").rstrip(".") + "M"
                if tokens >= 1_000:
                    return f"{tokens / 1_000:.1f}".rstrip("0").rstrip(".") + "k"
                return str(tokens)

            used = compact(usage.total_tokens)
            if limit is None:
                return f"上下文 {used} · 上限未知"
            return f"上下文 {usage.total_tokens / limit:.1%} · {used}/{compact(limit)}"

        used = f"{usage.total_tokens:,}" if usage else "—"
        if limit is None:
            return f"已测上下文 {used} · 上限未知"
        percent = f" · {usage.total_tokens / limit:.1%}" if usage else ""
        return f"已测上下文 {used} / {limit:,}{percent}"

    def on_resize(self) -> None:
        self.call_after_refresh(self._refresh_environment)

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        editor = self._view.query_one(PromptInput)
        editor.styles.height = min(8, max(3, editor.wrapped_document.height + 2))
        self._completion_items = completions(editor.text)
        menu = self._view.query_one("#completions", OptionList)
        menu.clear_options()
        menu.add_options(
            [Option(Text(f"/{name}  {COMMANDS[name]}"), id=name) for name in self._completion_items]
        )
        menu.display = bool(self._completion_items)
        if self._completion_items:
            menu.highlighted = 0

    def action_completion(self, key: str) -> None:
        menu = self._view.query_one("#completions", OptionList)
        if key == "tab":
            self._accept_completion(menu.highlighted or 0)
        elif key == "up":
            menu.action_cursor_up()
        else:
            menu.action_cursor_down()

    def _accept_completion(self, index: int) -> None:
        editor = self._view.query_one(PromptInput)
        text = f"/{self._completion_items[index]} "
        editor.load_text(text)
        editor.move_cursor((0, len(text)))
        editor.focus()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_list.id == "completions":
            self._accept_completion(event.option_index)

    def on_prompt_input_submitted(self, event: PromptInput.Submitted) -> None:
        text = event.text
        if not text.strip():
            return
        try:
            command = parse_command(text)
        except ValueError as error:
            self._state.add_tool("命令", str(error), error=True)
            self._refresh()
            return
        if command is not None:
            name, argument = command
            if self._command_busy or (
                name in ("new", "resume", "model", "thinking", "tree") and self._state.running
            ):
                self._state.add_tool("命令", "请等待当前任务完成，或按 Esc 取消")
                self._refresh()
                return
            self._view.query_one(PromptInput).load_text("")
            self._command_busy = True
            self._execute_command(name, argument)
            return
        if self._command_busy or self._state.running or self._current_worker is not None:
            self._view.query_one("#input-help", Static).update(
                "当前任务仍在运行，输入已保留；结束后按 Enter 发送"
            )
            return
        self._view.query_one(PromptInput).load_text("")
        self._view.query_one("#input-help", Static).update("Enter 发送 · Shift+Enter 换行")
        self._state.error = None
        self._state.set_running(True)
        self._state.add_user(text)
        self._refresh()
        self._current_worker = self._run_prompt(text)

    def _load_session(self, path: Path, *, fresh: bool = False):
        session = CodingSession.load(
            CodingSessionConfig(
                provider=self._provider,
                model=self._model,
                cwd=self._cwd,
                storage=JsonlStorage(path),
                thinking=self._session.thinking if fresh and self._session else "default",
                resolve_provider=self._sessions.resolve_provider,
            )
        )
        return session, session.provider

    def _ensure_session(self) -> CodingSession:
        if self._session is None:
            session, provider = self._load_session(self._session_path)
            if self._session_path.exists():
                self._sessions.select(self._session_path)
            self._session = session
            self._provider = provider
            self._state.usage = session.latest_usage
        return self._session

    async def _activate_session(self, path: Path, *, fresh: bool = False) -> None:
        session, provider = self._load_session(path, fresh=fresh)
        self._sessions.select(path)
        self._session, self._provider = session, provider
        self._session_path, self._model = path, session.model
        transcript = self._view.query_one("#transcript", VerticalScroll)
        if self._item_widgets:
            await transcript.remove_children(self._item_widgets)
        self._item_widgets.clear()
        self._item_versions.clear()
        self._tool_outputs.clear()
        self._state = TuiState.from_messages(session.messages)
        self._adapter = TuiEventAdapter(self._state)
        self._state.usage = session.latest_usage
        self._state.add_tool("会话", f"{'新建' if fresh else '恢复'} {path.stem}")
        if session.thinking_notice:
            self._state.add_tool("思考", session.thinking_notice)
        self._refresh_environment()
        self._refresh()
        self.action_jump_bottom()

    @work(group="commands", exclusive=True)
    async def _restore_startup(self) -> None:
        try:
            await self._activate_session(self._session_path)
        except Exception as error:
            self._state.add_tool(
                "恢复失败", f"{error}\n可用 /model 选择配置中的模型，保留原会话历史", error=True
            )
            self._refresh()
        finally:
            self._command_busy = False

    @work(group="commands", exclusive=True)
    async def _execute_command(self, name: str, argument: str) -> None:
        try:
            if name == "help":
                self._state.add_tool(
                    "帮助",
                    "\n".join(
                        f"/{command} — {description}" for command, description in COMMANDS.items()
                    )
                    + "\nTab 补全 · ↑↓ 选择 · Enter 执行 · Esc 取消 / 关闭窗口",
                )
            elif name == "quit":
                worker = self._current_worker
                if worker is not None:
                    self.action_cancel()
                    with contextlib.suppress(WorkerCancelled):
                        await worker.wait()
                self.exit()
                return
            elif name == "new":
                current = self._ensure_session()
                self._model = current.model
                await self._activate_session(self._sessions.new_path(), fresh=True)
            elif name == "resume":
                if not argument:
                    choices = [
                        (
                            s.id,
                            f"{s.title} · {s.messages} 条 · "
                            f"{datetime.fromtimestamp(s.updated_at):%m-%d %H:%M} · {s.id}",
                        )
                        for s in self._sessions.list_sessions()
                    ]
                    if not choices:
                        raise ValueError("当前项目还没有可恢复的会话")
                    argument = await self.push_screen_wait(ChoiceScreen("恢复会话", choices))
                if argument:
                    await self._activate_session(self._sessions.path_for(argument))
            elif name == "tree":
                session = self._ensure_session()
                if not argument:
                    choices = session.branch_choices()
                    if not choices:
                        raise ValueError("当前会话还没有可选择的历史消息")
                    argument = await self.push_screen_wait(
                        ChoiceScreen("会话树 · 从此节点继续，保留原分支", choices)
                    )
                if argument:
                    session.branch_to_entry(argument)
                    await self._activate_session(self._session_path)
                    self._state.add_tool("分支", f"已回到 {argument}，后续消息从这里继续")
            elif name == "model":
                config = load_config(self._paths.config_file)
                if not config.providers:
                    raise ValueError(f"没有可选模型，请编辑 {config.path}")
                if not argument:
                    argument = await self.push_screen_wait(
                        ChoiceScreen(
                            "选择供应商 / 模型",
                            [(p.name, f"{p.name} / {p.model}") for p in config.providers.values()],
                        )
                    )
                if argument:
                    profiles = [
                        p
                        for p in config.providers.values()
                        if argument in (p.name, p.model, f"{p.name}/{p.model}")
                    ]
                    if len(profiles) != 1:
                        raise ValueError("模型选择无效或不唯一；请使用配置中的供应商档案名")
                    profile = profiles[0]
                    candidate = build_provider(profile)
                    # 用户已经选定新模型，不需要先成功加载旧模型。
                    session = self._session or CodingSession.load(
                        CodingSessionConfig(
                            provider=candidate,
                            model=profile.model,
                            cwd=self._cwd,
                            storage=JsonlStorage(self._session_path),
                            resolve_provider=self._sessions.resolve_provider,
                        ),
                        model_override=profile.model,
                    )
                    reset = session.select_model(candidate, profile.model, profile.name)
                    if self._session is None:
                        await self._activate_session(self._session_path)
                    self._state.error = None
                    if profile.model != self._model or profile.name != getattr(
                        self._provider, "name", None
                    ):
                        self._state.usage = None
                    self._provider, self._model = session.provider, profile.model
                    if reset:
                        self._state.add_tool(
                            "思考", f"新模型不支持原思考设置，已切换为 {session.thinking}"
                        )
                    self._refresh_environment()
                    self._state.add_tool("模型", f"已切换为 {profile.name} / {profile.model}")
            elif name == "usage":
                self._ensure_session()
                self._state.add_tool(
                    "上下文统计",
                    self._context_status(detailed=True)
                    + "\n基于最近完整响应的 API total_tokens（输入历史 + 本次输出）。"
                    "\n不累加历次请求，不包含随后尚未发送的内容；暂无统计时显示 —。",
                )
            elif name == "thinking":
                session = self._ensure_session()
                if argument:
                    session.set_thinking(argument)
                    self._provider = session.provider
                label = "模型默认" if session.thinking == "default" else session.thinking
                self._state.add_tool(
                    "模型思考",
                    f"当前：{label}；可用：{', '.join(session.thinking_options)}"
                    "\n使用 /thinking 等级 修改；Ctrl+T 只展开或隐藏内容",
                )
                self._refresh_environment()
            self._refresh()
        except Exception as error:
            self._state.add_tool("命令失败", str(error), error=True)
            self._refresh()
        finally:
            self._command_busy = False
            if self._view.query(PromptInput):
                self._view.query_one(PromptInput).focus()

    @work(exclusive=True)
    async def _run_prompt(self, text: str) -> None:
        try:
            session = self._ensure_session()
            self._model = getattr(session, "model", self._model)
            self._refresh_environment()
            async for event in session.prompt(text):
                self._adapter.apply(event)
                if isinstance(event, MessageDeltaEvent):
                    self._refresh_throttled()
                else:
                    self._refresh()
        except asyncio.CancelledError:
            self._state.add_tool("运行", "已取消")
            raise
        except Exception as error:
            self._state.error = f"{type(error).__name__}: {error}"
            self._state.add_tool("运行", self._state.error, error=True)
        finally:
            self._state.interrupt_tools()
            self._state.set_running(False)
            self._state.end_assistant("")
            self._current_worker = None
            self._refresh()

    def _refresh_throttled(self) -> None:
        now = time.monotonic()
        if now - self._last_refresh >= _STREAM_REFRESH_INTERVAL:
            self._last_refresh = now
            self._refresh()

    def _tick(self) -> None:
        # 退出时界面先卸载，已经排队的定时回调仍可能执行。
        if not self._view.query("#transcript"):
            return
        if self._state.running:
            self._refresh()

    def action_cancel(self) -> None:
        if isinstance(self.screen, ChoiceScreen):
            self.screen.dismiss(None)
            return
        if self._current_worker is not None and not self._current_worker.is_cancelled:
            self._current_worker.cancel()
            self._state.add_tool("运行", "正在取消")
            self._refresh()

    def action_toggle_thinking(self) -> None:
        self._state.toggle_thinking()
        self._refresh()

    def action_jump_bottom(self) -> None:
        self._view.query_one("#transcript", VerticalScroll).scroll_end(animate=False)

    def _refresh(self) -> None:
        if not self._view.query("#transcript"):
            return
        transcript = self._view.query_one("#transcript", VerticalScroll)
        for index, item in enumerate(self._state.chat_items):
            if index >= len(self._item_widgets):
                if item.tool_call_id:
                    output_widget = Static("", classes="tool-output", markup=False)
                    self._tool_outputs[index] = output_widget
                    widget = ToolDetail(
                        output_widget,
                        title=item.text,
                        classes="tool",
                    )
                elif item.kind == ChatItemKind.assistant:
                    widget = Markdown(item.text)
                else:
                    widget = Static(
                        "",
                        markup=False,
                        classes=("user-message" if item.kind == ChatItemKind.user else "notice"),
                    )
                self._item_widgets.append(widget)
                transcript.mount(widget, before="#stream-thinking")
            widget = self._item_widgets[index]
            elapsed = item.elapsed
            if item.tool_call_id and elapsed is None:
                elapsed = time.monotonic() - item.started_at
            version = (
                item.text,
                item.status,
                item.output,
                item.error,
                round(elapsed, 1) if elapsed is not None else None,
                self._state.show_thinking if item.kind == ChatItemKind.thinking else None,
            )
            if self._item_versions.get(index) == version:
                continue
            self._item_versions[index] = version
            widget.set_class(item.error, "failed")
            if item.tool_call_id:
                symbol = "✗" if item.error else ("⏳" if item.elapsed is None else "✓")
                if item.status == "已中断":
                    symbol = "■"
                timing = "历史记录" if item.historical else f"{elapsed:.1f}s"
                widget.title = f"{symbol} {item.text} · {item.status} · {timing}"
                self._tool_outputs[index].update(item.output or "等待工具结果…")
                if item.error:
                    widget.collapsed = False
            elif item.kind == ChatItemKind.assistant:
                if widget.is_mounted:
                    widget.update(item.text)
            elif item.kind == ChatItemKind.thinking:
                widget.update(
                    f"思考 · {item.text}"
                    if self._state.show_thinking
                    else f"思考 · {len(item.text)} 字（Ctrl+T 展开）"
                )
            else:
                widget.update(f"你：{item.text}" if item.kind == ChatItemKind.user else item.text)

        thinking = self._view.query_one("#stream-thinking", Static)
        answer = self._view.query_one("#stream-answer", Static)
        reasoning = self._state.streaming_thinking
        text = self._state.streaming_text
        thinking.display = bool(reasoning)
        answer.display = bool(text)
        shown_reasoning = (
            reasoning if self._state.show_thinking else f"思考中 · {len(reasoning)} 字"
        )
        if self._stream_reasoning != shown_reasoning:
            thinking.update(shown_reasoning)
            self._stream_reasoning = shown_reasoning
        if self._stream_text != text:
            answer.update(RichMarkdown(text) if text else "")
            self._stream_text = text
        status = ""
        if self._state.running:
            active = next(
                (
                    i
                    for i in reversed(self._state.chat_items)
                    if i.tool_call_id and i.elapsed is None
                ),
                None,
            )
            if active:
                status = f"执行中：{active.text}"
            elif text:
                status = "正在回答…"
            elif reasoning:
                status = "正在思考…"
            else:
                status = "正在连接模型…"
        if self._state.error:
            status = f"错误：{self._state.error}"
        status_widget = self._view.query_one("#status", Static)
        status_widget.update(status)
        status_widget.display = bool(status)
        self._refresh_environment()


__all__ = ["NexaTuiApp"]
