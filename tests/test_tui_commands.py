"""本地命令不请求模型；切换后的会话、Provider 与账本保持一致。"""

import asyncio

import pytest
from textual.widgets import Input, OptionList
from textual.worker import WorkerCancelled

from nexa_agent.messages import AssistantMessage
from nexa_agent.provider_events import ProviderResponseEndEvent
from nexa_agent.session.memory import SessionState
from nexa_agent.session.storage import JsonlStorage
from nexa_ai.fake import FakeProvider
from nexa_coding.commands import completions, parse_command
from nexa_coding.paths import NexaPaths
from nexa_coding.tui.app import NexaTuiApp, PromptInput
from nexa_coding.tui.dialogs import ChoiceScreen


def test_command_parsing():
    assert parse_command("/resume\tdefault") == ("resume", "default")
    assert parse_command("/skill:review file") is None
    assert parse_command("/tmp/example.py 看看这个文件") is None
    assert parse_command("ordinary") is None
    assert parse_command("/quit") == ("quit", "")
    assert parse_command("/exit") == ("quit", "")
    with pytest.raises(ValueError, match="未知命令"):
        parse_command("/unknown")
    with pytest.raises(ValueError, match="不接受参数"):
        parse_command("/new argument")
    assert completions("/re") == ["resume"]
    assert completions("/resume default") == []


def provider(name):
    fake = FakeProvider(
        [[ProviderResponseEndEvent(message=AssistantMessage(content="完成"))] for _ in range(5)]
    )
    fake.name = name
    return fake


async def submit(app, pilot, text):
    editor = app.query_one(PromptInput)
    editor.load_text(text)
    await pilot.press("enter")
    await pilot.pause()


@pytest.mark.asyncio
async def test_new_resume_model_and_restart(tmp_path, monkeypatch):
    paths = NexaPaths(home=tmp_path / "home")
    paths.home.mkdir()
    paths.config_file.write_text("""
[providers.a]
base_url = "https://a"
model = "model-a"
api_key = "unused"
[providers.b]
base_url = "https://b"
model = "model-b"
api_key = "unused"
""")
    a, b = provider("a"), provider("b")
    monkeypatch.setattr("nexa_coding.tui.app.build_provider", lambda p: {"a": a, "b": b}[p.name])
    monkeypatch.setattr(
        "nexa_coding.session_manager.build_provider", lambda p: {"a": a, "b": b}[p.name]
    )
    app = NexaTuiApp(a, model="model-a", cwd=tmp_path, paths=paths)
    async with app.run_test(size=(100, 32)) as pilot:
        await submit(app, pilot, "/help")
        assert not a.calls
        await submit(app, pilot, "/re")
        assert not a.calls
        assert "未知命令" in app._state.chat_items[-1].text
        app.query_one(PromptInput).load_text("/re")
        await pilot.press("tab")
        assert app.query_one(PromptInput).text == "/resume "
        await submit(app, pilot, "first task")
        original = app._session_path
        assert original.name == "default.jsonl"
        await submit(app, pilot, "/new")
        new = app._session_path
        assert new != original
        assert not app._session.messages
        await submit(app, pilot, "second task")
        assert len(a.calls[-1][2]) == 1
        assert app._sessions.current_path() == new
        await submit(app, pilot, "/resume default")
        assert app._session_path == original
        assert app._session.messages[0].text == "first task"
        assert any("first task" in item.text for item in app._state.chat_items)
        await submit(app, pilot, "/model b")
        assert app._provider is b
        assert app._model == "model-b"
        await submit(app, pilot, "third task")
        assert b.calls[-1][0] == "model-b"
        assert b.calls[-1][2][0].text == "first task"
        restored = SessionState.from_entries(JsonlStorage(original).read_all())
        assert restored.provider == "b"
        assert restored.model == "model-b"
        assert len(JsonlStorage(new).read_all()) < len(JsonlStorage(original).read_all())
        await submit(app, pilot, "/resume")
        assert isinstance(app.screen, ChoiceScreen)
        await pilot.pause(0.3)  # 主界面计时器不能查询弹窗上的组件。
        await pilot.press("escape")
        await pilot.pause()
        await submit(app, pilot, "/model")
        assert isinstance(app.screen, ChoiceScreen)
        await pilot.press("escape")
        await pilot.pause()
        await submit(app, pilot, "/resume ../other")
        assert app._session_path == original
        assert "格式无效" in app._state.chat_items[-1].text
        await submit(app, pilot, "/resume")
        app.screen.query_one(Input).value = new.stem
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        assert app._session_path == new
        assert app._provider is a
        await submit(app, pilot, "/model")
        app.screen.query_one(OptionList).highlighted = 1
        await pilot.press("enter")
        await pilot.pause()
        assert app._provider is b
        assert app._model == "model-b"
        await submit(app, pilot, "/resume default")
    restarted = NexaTuiApp(a, model="model-a", cwd=tmp_path, paths=paths)
    async with restarted.run_test() as pilot:
        await pilot.pause()
        assert restarted._session_path == original
        assert restarted._provider is b
        assert restarted._model == "model-b"


@pytest.mark.asyncio
async def test_model_write_failure_keeps_current_provider(tmp_path, monkeypatch):
    from nexa_coding.session import CodingSession, CodingSessionConfig

    class Storage:
        entries = []
        fail = False

        def read_all(self):
            return list(self.entries)

        def append(self, entry):
            if self.fail:
                raise OSError("磁盘故障")
            self.entries.append(entry)

    storage = Storage()
    a, b = provider("a"), provider("b")
    session = CodingSession.load(
        CodingSessionConfig(provider=a, model="model-a", cwd=tmp_path, storage=storage)
    )
    storage.fail = True
    with pytest.raises(OSError, match="磁盘故障"):
        session.select_model(b, "model-b", "b")
    storage.fail = False
    async for _ in session.prompt("still a"):
        pass
    assert a.calls[-1][0] == "model-a"
    assert not b.calls


@pytest.mark.asyncio
async def test_exit_waits_for_cleanup_and_blocks_switching(tmp_path, monkeypatch):
    from nexa_agent.events import AgentStartEvent
    from nexa_coding.session import CodingSession

    started, cleaning, finish = asyncio.Event(), asyncio.Event(), asyncio.Event()

    class Session:
        provider = FakeProvider([])
        thinking = "default"
        latest_usage = None
        model = "test"

        async def prompt(self, text):
            yield AgentStartEvent()
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleaning.set()
                await finish.wait()

    monkeypatch.setattr(CodingSession, "load", lambda c: Session())
    app = NexaTuiApp(
        provider("a"), model="test", cwd=tmp_path, paths=NexaPaths(home=tmp_path / "home")
    )
    exited = []
    monkeypatch.setattr(app, "exit", lambda: exited.append(True))
    async with app.run_test() as pilot:
        await submit(app, pilot, "start")
        await asyncio.wait_for(started.wait(), 2)
        old = app._session_path
        await submit(app, pilot, "/new")
        assert app._session_path == old
        assert "等待" in app._state.chat_items[-1].text
        running = app._current_worker
        await submit(app, pilot, "/quit")
        await asyncio.wait_for(cleaning.wait(), 2)
        assert not exited
        finish.set()
        with pytest.raises(WorkerCancelled):
            await running.wait()
        await pilot.pause()
        assert exited == [True]


@pytest.mark.asyncio
async def test_tree_picker_switches_branch_and_restores_after_restart(tmp_path):
    paths = NexaPaths(home=tmp_path / "home")
    fake = provider("a")
    app = NexaTuiApp(fake, model="model-a", cwd=tmp_path, paths=paths)
    async with app.run_test(size=(120, 32)) as pilot:
        await submit(app, pilot, "first")
        first_tip = app._session.active_leaf_id
        await submit(app, pilot, "old branch")
        old_tip = app._session.active_leaf_id
        await submit(app, pilot, "/tree")
        assert isinstance(app.screen, ChoiceScreen)
        app.screen.query_one(Input).value = first_tip
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        assert app._session.active_leaf_id == first_tip
        assert not any(item.text == "old branch" for item in app._state.chat_items)
        await submit(app, pilot, "new branch")
        assert [m.text for m in app._session.messages] == ["first", "完成", "new branch", "完成"]
        await submit(app, pilot, f"/tree {old_tip}")
        assert [m.text for m in app._session.messages] == ["first", "完成", "old branch", "完成"]
    restarted = NexaTuiApp(fake, model="model-a", cwd=tmp_path, paths=paths)
    async with restarted.run_test() as pilot:
        await pilot.pause()
        assert restarted._session.active_leaf_id == old_tip
        assert any(item.text == "old branch" for item in restarted._state.chat_items)
        assert not any(item.text == "new branch" for item in restarted._state.chat_items)
