"""测试 CodingSession：持久化会话层的编码应用。"""

from __future__ import annotations

import pytest

from nexa_agent.messages import (
    AssistantMessage,
    TextContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from nexa_agent.provider_events import ProviderResponseEndEvent, ProviderResponseStartEvent
from nexa_agent.session.entries import (
    Entry,
    LeafEntry,
    MessageEntry,
    ModelChangeEntry,
    SessionInfoEntry,
)
from nexa_agent.session.storage import JsonlStorage
from nexa_ai.fake import FakeProvider
from nexa_coding.session import CodingSession, CodingSessionConfig

# ── 内存存储 ──────────────────────────────────────────────────────────────────


class InMemorySessionStorage:
    """把条目存在内存列表里，避免测试碰磁盘。"""

    def __init__(self) -> None:
        self._entries: list[Entry] = []

    def append(self, entry: Entry) -> None:
        self._entries.append(entry)

    def read_all(self) -> list[Entry]:
        return list(self._entries)


# ── 辅助函数 ──────────────────────────────────────────────────────────────────


def _make_config(storage: InMemorySessionStorage, provider: FakeProvider) -> CodingSessionConfig:
    """创建测试用配置。"""
    return CodingSessionConfig(
        provider=provider,
        model="test-model",
        system="你是助手",
        storage=storage,
        cwd=".",
    )


def _plain_stream(text: str) -> list:
    """构造一个"模型直接回复文字"的 Provider 事件流。"""
    return [
        ProviderResponseStartEvent(model="test-model"),
        ProviderResponseEndEvent(
            message=AssistantMessage(content=[TextContent(text=text)]),
            finish_reason="stop",
        ),
    ]


async def _run_prompt(session: CodingSession, text: str) -> None:
    """跑一轮 prompt，消费所有事件。"""
    async for _ in session.prompt(text):
        pass


async def _run_continue(session: CodingSession) -> None:
    """跑一轮 continue_，消费所有事件。"""
    async for _ in session.continue_():
        pass


# ── 测试用例 ──────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_empty_session_initializes_metadata():
    """空会话 load 后应追加 SessionInfoEntry + ModelChangeEntry。"""
    storage = InMemorySessionStorage()
    provider = FakeProvider([_plain_stream("你好！")])

    session = CodingSession.load(_make_config(storage, provider))

    entries = storage.read_all()
    # 空会话初始化：info + model 两条。
    assert len(entries) == 2
    assert isinstance(entries[0], SessionInfoEntry)
    assert isinstance(entries[1], ModelChangeEntry)
    assert entries[1].model == "test-model"
    # 初始历史为空。
    assert session.messages == ()


@pytest.mark.asyncio
async def test_prompt_persists_new_messages():
    """prompt 跑完后，账本里应有新 MessageEntry + LeafEntry。"""
    storage = InMemorySessionStorage()
    provider = FakeProvider([_plain_stream("这是回答")])
    session = CodingSession.load(_make_config(storage, provider))

    await _run_prompt(session, "你好")

    entries = storage.read_all()
    # info + model + user message + assistant message + leaf = 5
    message_entries = [e for e in entries if isinstance(e, MessageEntry)]
    leaf_entries = [e for e in entries if isinstance(e, LeafEntry)]
    assert len(message_entries) == 2
    assert isinstance(message_entries[0].message, UserMessage)
    assert message_entries[0].message.text == "你好"
    assert isinstance(message_entries[1].message, AssistantMessage)
    assert message_entries[1].message.text == "这是回答"
    # 有一条叶子指针，指向最新消息。
    assert len(leaf_entries) == 1
    assert leaf_entries[0].target_id == message_entries[-1].id


@pytest.mark.asyncio
async def test_load_recovers_transcript():
    """load 后 harness 消息应等于账本里的消息。"""
    storage = InMemorySessionStorage()
    provider = FakeProvider([_plain_stream("第一条回答"), _plain_stream("第二条回答")])
    session = CodingSession.load(_make_config(storage, provider))

    await _run_prompt(session, "问题一")
    expected = [m.text for m in session.messages]

    # 重新 load：从账本恢复历史。
    provider2 = FakeProvider([_plain_stream("继续回答")])
    session2 = CodingSession.load(_make_config(storage, provider2))

    assert [m.text for m in session2.messages] == expected


@pytest.mark.asyncio
async def test_continue_persists_second_round():
    """continue_ 跑完后，第二轮消息也应落盘，不丢失第一轮。"""
    storage = InMemorySessionStorage()
    provider = FakeProvider([_plain_stream("第一轮回答"), _plain_stream("第二轮回答")])
    session = CodingSession.load(_make_config(storage, provider))

    await _run_prompt(session, "问题一")
    await _run_continue(session)

    entries = storage.read_all()
    message_entries = [e for e in entries if isinstance(e, MessageEntry)]
    # 第一轮 user + assistant，continue 只追加 assistant（不加用户消息），共 3 条。
    assert len(message_entries) == 3
    texts = [e.message.text for e in message_entries]
    assert texts == ["问题一", "第一轮回答", "第二轮回答"]


@pytest.mark.asyncio
async def test_tool_results_persisted():
    """带工具调用的消息也应正确写入账本。"""
    # 第一轮：assistant 请求调用工具；第二轮：assistant 给最终回答。
    provider_events = [
        [
            ProviderResponseStartEvent(model="test-model"),
            ProviderResponseEndEvent(
                message=AssistantMessage(
                    content=[
                        TextContent(text="让我查一下"),
                        ToolCall(id="call-1", name="read", arguments={"path": "README.md"}),
                    ]
                ),
                finish_reason="tool_calls",
            ),
        ],
        _plain_stream("文件内容是 README"),
    ]
    storage = InMemorySessionStorage()
    provider = FakeProvider(provider_events)
    session = CodingSession.load(_make_config(storage, provider))

    await _run_prompt(session, "读 README")

    entries = storage.read_all()
    message_entries = [e for e in entries if isinstance(e, MessageEntry)]
    types = [type(e.message).__name__ for e in message_entries]
    # user → assistant(带工具调用) → toolResult → assistant(最终)
    assert types == ["UserMessage", "AssistantMessage", "ToolResultMessage", "AssistantMessage"]
    # 带工具调用的 assistant 消息要保留 tool_calls。
    tool_msg = message_entries[1].message
    assert isinstance(tool_msg, AssistantMessage)
    assert len(tool_msg.tool_calls) == 1
    assert tool_msg.tool_calls[0].name == "read"
    # 工具结果消息要保留 tool_call_id。
    tool_result = message_entries[2].message
    assert isinstance(tool_result, ToolResultMessage)
    assert tool_result.tool_call_id == "call-1"


@pytest.mark.asyncio
async def test_prompt_writes_real_jsonl(tmp_path):
    """用真实 JsonlStorage 落盘：JSONL 文件存在且内容正确。"""
    storage = JsonlStorage(tmp_path / "session.jsonl")
    provider = FakeProvider([_plain_stream("落盘回答")])
    session = CodingSession.load(_make_config(storage, provider))

    await _run_prompt(session, "你好")

    # 文件真实存在，且每行都是合法 JSON。
    lines = (tmp_path / "session.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 5  # info + model + user + assistant + leaf
    import json

    for line in lines:
        json.loads(line)

    # 重新 load 能恢复出同样消息。
    provider2 = FakeProvider([_plain_stream("再来")])
    session2 = CodingSession.load(_make_config(storage, provider2))
    assert [m.text for m in session2.messages] == ["你好", "落盘回答"]
