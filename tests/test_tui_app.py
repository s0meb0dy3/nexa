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
