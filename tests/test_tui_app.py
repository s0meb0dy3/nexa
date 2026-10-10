"""验证真实 Textual Worker 的异常收尾和取消。"""

import asyncio

import pytest
from textual.worker import WorkerCancelled

from nexa_agent.events import AgentStartEvent, MessageDeltaEvent, MessageStartEvent
from nexa_agent.messages import AssistantMessage
from nexa_ai.fake import FakeProvider
from nexa_coding.session import CodingSession
from nexa_coding.tui.app import NexaTuiApp


@pytest.mark.asyncio
@pytest.mark.parametrize("load_fails", [False, True])
async def test_worker_failure_resets_ui(tmp_path, monkeypatch, load_fails):
    class BrokenSession:
        provider = FakeProvider([])
        thinking = "default"
        latest_usage = None

        async def prompt(self, text):
            yield AgentStartEvent()
            yield MessageStartEvent(message=AssistantMessage())
            yield MessageDeltaEvent(kind="text", delta="部分输出")
            raise OSError("会话故障")

    def load(config):
        if load_fails:
            raise OSError("会话故障")
        return BrokenSession()

    monkeypatch.setattr(CodingSession, "load", load)
    app = NexaTuiApp(FakeProvider([]), model="test", cwd=tmp_path)
    async with app.run_test():
        app._state.set_running(True)
        app._current_worker = app._run_prompt("测试")
        await app._current_worker.wait()
        assert not app._state.running
        assert not app._state.streaming_started
        assert app._state.streaming_text == ""
        assert "会话故障" in app._state.error
        assert app._current_worker is None


@pytest.mark.asyncio
async def test_worker_cancellation_waits_for_cleanup(tmp_path, monkeypatch):
    started = asyncio.Event()
    cleaning = asyncio.Event()
    finish_cleanup = asyncio.Event()

    class SlowSession:
        provider = FakeProvider([])
        thinking = "default"
        latest_usage = None

        async def prompt(self, text):
            yield AgentStartEvent()
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleaning.set()
                await finish_cleanup.wait()

    monkeypatch.setattr(CodingSession, "load", lambda config: SlowSession())
    app = NexaTuiApp(FakeProvider([]), model="test", cwd=tmp_path)
    async with app.run_test():
        worker = app._run_prompt("测试")
        app._current_worker = worker
        await asyncio.wait_for(started.wait(), 2)
        app.action_cancel()
        await asyncio.wait_for(cleaning.wait(), 2)
        assert app._state.running
        app.action_cancel()  # 重复 Escape 不应中断正在进行的清理。
        finish_cleanup.set()
        with pytest.raises(WorkerCancelled):
            await worker.wait()
        assert not app._state.running
        assert app._current_worker is None
        assert app._state.chat_items[-1].text == "运行 — 已取消"


@pytest.mark.asyncio
async def test_multiline_input_preserves_draft_while_running(tmp_path, monkeypatch):
    from textual import events
    from textual.widgets import Static

    from nexa_coding.tui.app import PromptInput

    received = []
    started = asyncio.Event()
    done = asyncio.Event()

    class Session:
        provider = FakeProvider([])
        thinking = "default"
        latest_usage = None
        model = "restored-model"

        async def prompt(self, text):
            received.append(text)
            yield AgentStartEvent()
            started.set()
            await done.wait()

    monkeypatch.setattr(CodingSession, "load", lambda config: Session())
    app = NexaTuiApp(FakeProvider([]), model="test", cwd=tmp_path)
    async with app.run_test(size=(100, 32)) as pilot:
        editor = app.query_one(PromptInput)
        editor.load_text("first")
        editor.move_cursor((0, 5))
        await pilot.press("shift+enter", "s", "e", "c", "o", "n", "d")
        assert editor.text == "first\nsecond"
        await pilot.press("ctrl+j", "x", "enter")
        await asyncio.wait_for(started.wait(), 2)
        assert received == ["first\nsecond\nx"]
        assert editor.text == ""
        app.post_message(events.Paste("  draft\nnext  "))
        await pilot.pause()
        await pilot.press("enter")
        assert editor.text == "  draft\nnext  "
        assert len(received) == 1
        assert app._model == "restored-model"
        assert "restored-model" in str(app.query_one("#environment", Static).render())
        worker = app._current_worker
        done.set()
        await worker.wait()
        await pilot.press("enter")
        assert received[-1] == "  draft\nnext  "


@pytest.mark.asyncio
async def test_incremental_messages_tools_and_scroll(tmp_path):
    from textual.containers import VerticalScroll
    from textual.widgets import Collapsible, Markdown, Static

    app = NexaTuiApp(FakeProvider([]), model="test", cwd=tmp_path)
    async with app.run_test(size=(90, 26)) as pilot:
        for i in range(20):
            app._state.add_user(f"问题 {i}\n第二行")
            app._state.end_assistant(f"## 回复 {i}\n\n```python\nprint({i})\n```")
        app._refresh()
        await pilot.pause()
        transcript = app.query_one("#transcript", VerticalScroll)
        assert transcript.is_vertical_scroll_end
        first = app._item_widgets[1]
        assert isinstance(first, Markdown)
        assert first.source.startswith("## 回复 0")
        assert len(first.children) > 0
        transcript.scroll_to(y=0, animate=False)
        await pilot.pause()
        app._state.start_assistant()
        app._state.append_assistant_delta("# 流式回复")
        app._refresh()
        await pilot.pause()
        assert transcript.scroll_y == 0
        assert not app.query("#jump-bottom")
        app._state.end_assistant("# 完整回复\n\n- 完成")
        app._state.update_tool("c1", "read", "执行中", args={"path": "README.md"})
        app._refresh()
        await pilot.pause()
        tool = app.query_one(Collapsible)
        assert "README.md" in tool.title
        assert tool.collapsed
        app._state.update_tool("c1", "read", "失败", output="文件不存在", error=True)
        app._refresh()
        await pilot.pause()
        assert len(app.query(Collapsible)) == 1
        assert not tool.collapsed
        assert "失败" in tool.title
        assert "文件不存在" in str(tool.query_one(".tool-output", Static).render())
        assert app._item_widgets[1] is first
        assert not app.query_one("#stream-answer").display
        assert transcript.scroll_y == 0
        app.action_jump_bottom()
        await pilot.pause()
        assert transcript.is_vertical_scroll_end
        app._state.add_user("新的问题")
        app._refresh()
        await pilot.pause()
        assert transcript.is_vertical_scroll_end
        await pilot.resize_terminal(45, 18)
        assert app.query_one("#environment").size.height == 1
        assert app.query_one("#input-meta").size.height == 1
        assert app.query_one("#environment").region.y == app.query_one("#context-usage").region.y


@pytest.mark.asyncio
async def test_clicking_streaming_code_during_updates(tmp_path):
    from textual import events
    from textual.widgets import Markdown, Static

    app = NexaTuiApp(FakeProvider([]), model="test", cwd=tmp_path)
    async with app.run_test(size=(100, 32)) as pilot:
        app._state.start_assistant()
        app._state.append_assistant_delta("```python\nprint('first')\n```\n")
        app._refresh()
        await pilot.pause()
        preview = app.query_one("#stream-answer")
        for i in range(8):
            # 模拟鼠标仍指向旧布局时，新输出已经更新了代码块。
            x, y = preview.region.x + 3, preview.region.y + 1
            app._state.append_assistant_delta(f"\n```python\nprint({i})\n```\n")
            app._refresh()
            app.post_message(events.MouseDown(None, x, y, 0, 0, 1, False, False, False))
            app.post_message(events.MouseUp(None, x, y, 0, 0, 1, False, False, False))
            await pilot.pause()
            assert app.query_one("#stream-answer") is preview
        app._state.end_assistant("```python\nprint('complete')\n```")
        app._refresh()
        await pilot.pause()
        completed = app._item_widgets[-1]
        assert isinstance(completed, Markdown)
        assert "complete" in completed.source
        assert not app.query_one("#stream-answer", Static).display
        await pilot.click(completed.query_one("#code-content"))


@pytest.mark.asyncio
async def test_status_only_shows_active_work(tmp_path):
    from textual.widgets import Button, Footer, Static

    app = NexaTuiApp(FakeProvider([]), model="test", cwd=tmp_path)
    async with app.run_test() as pilot:
        status = app.query_one("#status", Static)
        assert not app.query(Footer)
        assert not app.query(Button)
        assert not status.display
        app._state.set_running(True)
        app._refresh()
        assert "正在连接模型" in str(status.render())
        app._state.start_assistant()
        app._state.append_thinking_delta("思考")
        app._refresh()
        assert "正在思考" in str(status.render())
        app._state.append_assistant_delta("回答")
        app._refresh()
        assert "正在回答" in str(status.render())
        app._state.set_running(False)
        app._refresh()
        assert not status.display
        before = app._state.show_thinking
        await pilot.press("ctrl+t")
        assert app._state.show_thinking != before
