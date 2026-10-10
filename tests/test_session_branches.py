"""监听器保存边界、取消恢复与会话分支。"""

import asyncio
from contextlib import aclosing

import pytest

from nexa_agent.events import MessageEndEvent, ToolExecutionEndEvent
from nexa_agent.harness import AgentHarness, AgentHarnessConfig
from nexa_agent.messages import AssistantMessage, ToolCall, ToolResultMessage, UserMessage
from nexa_agent.provider_events import ProviderResponseEndEvent
from nexa_agent.session.entries import LeafEntry, MessageEntry
from nexa_agent.session.memory import SessionState
from nexa_agent.session.storage import JsonlStorage
from nexa_agent.session.tree import TreeError, path_to_entry
from nexa_ai.fake import FakeProvider
from nexa_coding.session import CodingSession, CodingSessionConfig


def reply(text):
    return [ProviderResponseEndEvent(message=AssistantMessage(content=text))]


def config(tmp_path, provider):
    return CodingSessionConfig(
        provider=provider,
        model="model-a",
        system="test",
        cwd=tmp_path,
        storage=JsonlStorage(tmp_path / "session.jsonl"),
    )


async def send(session, text):
    return [event async for event in session.prompt(text)]


@pytest.mark.asyncio
async def test_async_listener_runs_before_yield_and_sees_updated_history():
    harness = AgentHarness(AgentHarnessConfig(provider=FakeProvider([reply("answer")]), model="a"))
    saved = []

    class Listener:
        async def on_event(self, event):
            if isinstance(event, MessageEndEvent):
                assert harness.messages[-1] == event.message
                await asyncio.sleep(0)
                saved.append(event.message)

    harness.subscribe(Listener())
    async for event in harness.prompt("question"):
        if isinstance(event, MessageEndEvent):
            assert saved[-1] == event.message
    assert [message.text for message in saved] == ["question", "answer"]


@pytest.mark.asyncio
async def test_completed_tool_saved_before_consumer_closes(tmp_path):
    (tmp_path / "file.txt").write_text("content")
    provider = FakeProvider(
        [
            [
                ProviderResponseEndEvent(
                    message=AssistantMessage(
                        content=[ToolCall(id="read-1", name="read", arguments={"path": "file.txt"})]
                    )
                )
            ],
            reply("done"),
        ]
    )
    cfg = config(tmp_path, provider)
    session = CodingSession.load(cfg)
    async with aclosing(session.prompt("read file")) as stream:
        async for event in stream:
            if isinstance(event, ToolExecutionEndEvent):
                restored = CodingSession.load(cfg)
                assert isinstance(restored.messages[-1], ToolResultMessage)
                assert restored.messages[-1].text
                break
    assert not session._harness.is_running
    assert len(provider.calls) == 1
    # 新进程带着已完成的工具结果继续，不会重复执行工具。
    cfg.provider = FakeProvider([reply("continued")])
    restored = CodingSession.load(cfg)
    async for _ in restored.continue_():
        pass
    assert restored.messages[-1].text == "continued"
    assert isinstance(cfg.provider.calls[0][2][-1], ToolResultMessage)


@pytest.mark.asyncio
async def test_cancel_during_next_api_keeps_finished_messages(tmp_path):
    waiting = asyncio.Event()

    class WaitingProvider(FakeProvider):
        async def stream_response(self, **kwargs):
            if self.calls:
                waiting.set()
                await asyncio.Event().wait()
            async for event in super().stream_response(**kwargs):
                yield event

    provider = WaitingProvider(
        [
            [
                ProviderResponseEndEvent(
                    message=AssistantMessage(
                        content=[ToolCall(id="read-1", name="read", arguments={"path": "missing"})]
                    )
                )
            ],
        ]
    )
    cfg = config(tmp_path, provider)
    session = CodingSession.load(cfg)
    task = asyncio.create_task(send(session, "read"))
    await asyncio.wait_for(waiting.wait(), timeout=2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not session._harness.is_running
    restored = CodingSession.load(cfg)
    assert [message.role for message in restored.messages] == ["user", "assistant", "toolResult"]
    assert restored.messages == session.messages


@pytest.mark.asyncio
async def test_message_write_failure_stops_before_tool_and_requires_reload(tmp_path, monkeypatch):
    provider = FakeProvider(
        [
            [
                ProviderResponseEndEvent(
                    message=AssistantMessage(
                        content=[
                            ToolCall(
                                id="write-1",
                                name="write",
                                arguments={"path": "new.txt", "content": "x"},
                            )
                        ]
                    )
                )
            ],
        ]
    )
    cfg = config(tmp_path, provider)
    session = CodingSession.load(cfg)
    append = cfg.storage.append

    def fail_assistant(entry):
        if isinstance(entry, MessageEntry) and isinstance(entry.message, AssistantMessage):
            raise OSError("disk full")
        append(entry)

    monkeypatch.setattr(cfg.storage, "append", fail_assistant)
    with pytest.raises(OSError, match="disk full"):
        await send(session, "write")
    assert not (tmp_path / "new.txt").exists()
    assert session.messages[-1].tool_calls
    assert not session._harness.is_running
    assert [message.role for message in CodingSession.load(cfg).messages] == ["user"]
    with pytest.raises(RuntimeError, match="重新加载"):
        await send(session, "again")
    assert len(provider.calls) == 1


@pytest.mark.asyncio
async def test_branch_preserves_both_paths_and_model_settings(tmp_path):
    a, b = FakeProvider([reply("answer-a"), reply("new-branch")]), FakeProvider([reply("answer-b")])
    a.name, b.name = "a", "b"
    cfg = config(tmp_path, a)
    cfg.resolve_provider = {"a": a, "b": b}.__getitem__
    session = CodingSession.load(cfg)
    await send(session, "first")
    first = session.active_leaf_id
    session.select_model(b, "model-b", "b")
    await send(session, "second")
    old_tip = session.active_leaf_id
    before = cfg.storage.read_all()
    session.branch_to_entry(first)
    assert session.provider is a
    assert session.model == "model-a"
    assert [m.text for m in session.messages] == ["first", "answer-a"]
    assert cfg.storage.read_all()[: len(before)] == before
    await send(session, "alternative")
    new_tip = session.active_leaf_id
    restored = CodingSession.load(cfg)
    assert restored.active_leaf_id == new_tip
    assert [m.text for m in restored.messages] == ["first", "answer-a", "alternative", "new-branch"]
    assert restored.model == "model-a"
    restored.branch_to_entry(old_tip)
    assert restored.provider is b
    assert restored.model == "model-b"
    assert [m.text for m in restored.messages] == ["first", "answer-a", "second", "answer-b"]
    assert new_tip in dict(restored.branch_choices())
    with pytest.raises(ValueError):
        restored.branch_to_entry("not-a-node")
    restored._harness._is_running = True
    with pytest.raises(RuntimeError, match="运行中"):
        restored.branch_to_entry(first)


@pytest.mark.asyncio
async def test_incomplete_tool_group_cannot_branch_or_send(tmp_path):
    provider = FakeProvider(
        [
            [
                ProviderResponseEndEvent(
                    message=AssistantMessage(
                        content=[
                            ToolCall(id="one", name="read", arguments={"path": "missing"}),
                            ToolCall(id="two", name="read", arguments={"path": "missing"}),
                        ]
                    )
                )
            ],
        ]
    )
    cfg = config(tmp_path, provider)
    session = CodingSession.load(cfg)
    async with aclosing(session.prompt("read")) as stream:
        async for event in stream:
            if isinstance(event, ToolExecutionEndEvent):
                break
    entries = [e for e in cfg.storage.read_all() if isinstance(e, MessageEntry)]
    assert dict(session.branch_choices()).keys() == {entries[0].id}
    with pytest.raises(ValueError, match="全部结果"):
        session.branch_to_entry(entries[-1].id)
    with pytest.raises(ValueError, match="/tree"):
        await send(session, "continue")
    session.branch_to_entry(entries[0].id)
    assert session.messages == (entries[0].message,)


def test_legacy_leaf_and_latest_setting_replay():
    from nexa_agent.session.entries import ModelChangeEntry

    entries = [
        ModelChangeEntry(id="model", model="a"),
        MessageEntry(id="message-1", parent_id="model", message=UserMessage(content="hi")),
        LeafEntry(id="leaf-2", parent_id="message-1", target_id="message-1"),
        ModelChangeEntry(id="model-3", parent_id="leaf-2", model="b"),
    ]
    assert SessionState.from_entries(entries).model == "b"
    entries.append(LeafEntry(id="leaf-4", parent_id="message-1", target_id="message-1"))
    assert SessionState.from_entries(entries).model == "a"
    with pytest.raises(TreeError, match="循环"):
        path_to_entry(
            [MessageEntry(id="cycle", parent_id="cycle", message=UserMessage(content="hi"))],
            "cycle",
        )


def test_branch_restores_thinking_and_usage(tmp_path):
    from nexa_agent.messages import TokenUsage
    from nexa_agent.session.entries import ModelChangeEntry, ThinkingChangeEntry
    from nexa_ai.deepseek import DeepSeekProvider

    provider = DeepSeekProvider(name="deepseek", api_key="unused")
    cfg = config(tmp_path, provider)
    cfg.model = "deepseek-flash"
    usage = TokenUsage(model=cfg.model, input_tokens=10, output_tokens=5, total_tokens=15)
    for entry in [
        ModelChangeEntry(id="model", model=cfg.model, provider="deepseek", thinking="high"),
        MessageEntry(
            id="answer-1", parent_id="model", message=AssistantMessage(content="first", usage=usage)
        ),
        ThinkingChangeEntry(id="thinking-2", parent_id="answer-1", thinking="low"),
        MessageEntry(
            id="answer-3", parent_id="thinking-2", message=AssistantMessage(content="second")
        ),
    ]:
        cfg.storage.append(entry)
    session = CodingSession.load(cfg)
    assert session.thinking == "low"
    session.branch_to_entry("answer-1")
    assert session.thinking == "high"
    assert session.provider.thinking == "high"
    assert session.latest_usage == usage
    restored = CodingSession.load(cfg)
    assert restored.thinking == "high"
    assert len(restored.messages) == 1


@pytest.mark.asyncio
async def test_closing_display_stream_closes_provider_immediately(tmp_path):
    from nexa_agent.events import MessageDeltaEvent
    from nexa_agent.provider_events import ProviderDeltaEvent

    closed = False

    class StreamingProvider(FakeProvider):
        async def stream_response(self, **kwargs):
            nonlocal closed
            try:
                yield ProviderDeltaEvent(delta="partial")
                await asyncio.Event().wait()
            finally:
                closed = True

    cfg = config(tmp_path, StreamingProvider([]))
    session = CodingSession.load(cfg)
    async with aclosing(session.prompt("question")) as stream:
        async for event in stream:
            if isinstance(event, MessageDeltaEvent):
                break
    assert closed
    assert not session._harness.is_running
    assert [m.text for m in CodingSession.load(cfg).messages] == ["question"]
