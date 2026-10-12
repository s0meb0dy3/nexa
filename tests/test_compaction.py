"""主动、自动压缩与分支恢复；全部使用确定性 Provider，不访问真实 API。"""

import asyncio

import pytest

from nexa_agent.messages import (
    AssistantMessage,
    TokenUsage,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from nexa_agent.provider_events import ProviderErrorEvent, ProviderResponseEndEvent
from nexa_agent.session.entries import CompactionEntry
from nexa_agent.session.jsonl import entry_from_line, entry_to_line
from nexa_agent.session.storage import JsonlStorage
from nexa_ai.fake import FakeProvider
from nexa_coding.compaction import SUMMARY_SYSTEM, cut_point
from nexa_coding.session import CodingSession, CodingSessionConfig


def reply(text, usage=None):
    return [ProviderResponseEndEvent(message=AssistantMessage(content=text, usage=usage))]


def config(tmp_path, provider):
    return CodingSessionConfig(
        provider=provider,
        model="test",
        system="rules",
        cwd=tmp_path,
        skills=[],
        storage=JsonlStorage(tmp_path / "session.jsonl"),
    )


async def send(session, text):
    return [event async for event in session.prompt(text)]


class BoundedProvider(FakeProvider):
    def context_window(self, model):
        return 30000


def usage(total):
    return TokenUsage(model="test", input_tokens=total - 100, output_tokens=100, total_tokens=total)


async def test_manual_repeat_restore_and_branch(tmp_path):
    provider = FakeProvider([reply("原回答"), reply("摘要一"), reply("新回答"), reply("摘要二")])
    cfg = config(tmp_path, provider)
    session = CodingSession.load(cfg)
    await send(session, "原问题" * 3000)
    old_tip = session.active_leaf_id
    before = cfg.storage.read_all()
    assert await session.compact("保留测试结果") == "摘要一"
    first_compact = session.compaction_id
    assert cfg.storage.read_all()[:-1] == before
    assert isinstance(cfg.storage.read_all()[-1], CompactionEntry)
    assert (
        entry_from_line(entry_to_line(cfg.storage.read_all()[-1]), 1) == cfg.storage.read_all()[-1]
    )
    assert "保留测试结果" in provider.calls[1][2][0].text
    assert not provider.calls[1][3]
    restored = CodingSession.load(cfg)
    assert restored.messages == session.messages
    assert restored.transcript == session.transcript
    assert restored.transcript[0].text == "原问题" * 3000
    assert restored.compaction_id == first_compact
    await send(restored, "第二个长问题" * 5000)
    new_tip = restored.active_leaf_id
    await restored.compact()
    assert "摘要一" in provider.calls[-1][2][0].text
    assert restored.messages[0].text == "历史任务摘要：\n摘要二"
    assert restored.latest_usage is None
    restored.branch_to_entry(old_tip)
    assert restored.compaction_id is None
    assert restored.messages[0].text == "原问题" * 3000
    restored.branch_to_entry(new_tip)
    assert restored.compaction_id == first_compact
    assert restored.messages[0].text == "历史任务摘要：\n摘要一"


async def test_automatic_before_new_prompt_uses_measured_usage(tmp_path):
    provider = BoundedProvider(
        [reply("回答", usage(25000)), reply("交接摘要"), reply("继续完成", usage(300))]
    )
    session = CodingSession.load(config(tmp_path, provider))
    await send(session, "问题" * 2500)
    await send(session, "继续")
    assert len(provider.calls) == 3
    assert provider.calls[1][1] == SUMMARY_SYSTEM
    assert provider.calls[2][2][0].text == "历史任务摘要：\n交接摘要"
    assert provider.calls[2][2][-1].text == "继续"
    assert session.latest_usage.total_tokens == 300


async def test_automatic_between_tool_batch_and_next_request(tmp_path):
    call = ToolCall(id="read", name="read", arguments={"path": "missing"})
    provider = BoundedProvider(
        [
            [
                ProviderResponseEndEvent(
                    message=AssistantMessage(content=[call], usage=usage(25000))
                )
            ],
            reply("任务摘要"),
            reply("完成"),
        ]
    )
    session = CodingSession.load(config(tmp_path, provider))
    await send(session, "读文件" * 2000)
    assert provider.calls[1][1] == SUMMARY_SYSTEM
    messages = provider.calls[2][2]
    assert messages[0].text == "历史任务摘要：\n任务摘要"
    assert messages[-2].tool_calls == (call,)
    assert isinstance(messages[-1], ToolResultMessage)
    assert len([m for m in session.messages if isinstance(m, ToolResultMessage)]) == 1


async def test_unknown_limit_does_not_automatically_compact(tmp_path):
    provider = FakeProvider([reply("回答", usage(999999)), reply("继续")])
    session = CodingSession.load(config(tmp_path, provider))
    await send(session, "问题")
    await send(session, "继续")
    assert session.compaction_id is None
    assert len(provider.calls) == 2


async def test_failed_auto_compaction_stops_before_next_model_request(tmp_path):
    provider = BoundedProvider(
        [reply("回答", usage(25000)), [ProviderErrorEvent(message="摘要失败")]]
    )
    cfg = config(tmp_path, provider)
    session = CodingSession.load(cfg)
    await send(session, "问题" * 2500)
    with pytest.raises(RuntimeError, match="摘要失败"):
        await send(session, "继续")
    assert len(provider.calls) == 2
    assert not any(isinstance(entry, CompactionEntry) for entry in cfg.storage.read_all())
    assert session.messages[-1].text == "继续"
    assert not session._harness.is_running


@pytest.mark.parametrize("failure", ["error", "empty", "larger", "write"])
async def test_failed_compaction_does_not_change_state(tmp_path, failure, monkeypatch):
    end = (
        [ProviderErrorEvent(message="网络失败")]
        if failure == "error"
        else reply("" if failure == "empty" else "x" * 100000 if failure == "larger" else "摘要")
    )
    provider = FakeProvider([reply("回答"), end])
    cfg = config(tmp_path, provider)
    session = CodingSession.load(cfg)
    await send(session, "问题" * 2000)
    before, messages, tip = cfg.storage.read_all(), session.messages, session.active_leaf_id
    if failure == "write":

        def fail(entry):
            raise OSError("disk full")

        monkeypatch.setattr(cfg.storage, "append", fail)
    with pytest.raises((RuntimeError, ValueError, OSError)):
        await session.compact()
    assert cfg.storage.read_all() == before
    assert session.messages == messages
    assert session.active_leaf_id == tip
    assert not session.compacting


async def test_cancelled_compaction_keeps_history(tmp_path):
    started = asyncio.Event()

    class WaitingProvider(FakeProvider):
        async def stream_response(self, **kwargs):
            if kwargs["system"] == SUMMARY_SYSTEM:
                started.set()
                await asyncio.Event().wait()
            async for event in super().stream_response(**kwargs):
                yield event

    cfg = config(tmp_path, WaitingProvider([reply("回答")]))
    session = CodingSession.load(cfg)
    await send(session, "问题" * 2000)
    before = cfg.storage.read_all()
    task = asyncio.create_task(session.compact())
    await asyncio.wait_for(started.wait(), 2)
    with pytest.raises(RuntimeError, match="压缩"):
        await send(session, "其他输入")
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cfg.storage.read_all() == before
    assert not session.compacting


def test_cut_point_never_splits_tool_group():
    call = ToolCall(id="read", name="read", arguments={})
    messages = [
        UserMessage(content="任务"),
        AssistantMessage(content=[call]),
        ToolResultMessage(tool_call_id="read", tool_name="read", content="x" * 10000),
    ]
    assert cut_point(messages, 50) == 1
    with pytest.raises(ValueError, match="尚未完成"):
        cut_point(messages[:-1], 50)
