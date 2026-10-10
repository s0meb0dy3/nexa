"""测试 session 持久化层的 5 个不变量。"""

from __future__ import annotations

from nexa_agent.messages import AssistantMessage, TextContent, ToolResultMessage, UserMessage
from nexa_agent.session.entries import LabelEntry, MessageEntry, ModelChangeEntry
from nexa_agent.session.jsonl import JsonlLineError, entry_from_line
from nexa_agent.session.memory import SessionState
from nexa_agent.session.storage import JsonlStorage
from nexa_agent.session.tree import TreeError, path_to_entry

# ── 辅助函数 ────────────────────────────────────────────────────────────────


def _user_entry(entry_id: str, parent_id: str | None, text: str) -> MessageEntry:
    """构造一条用户消息条目。"""
    return MessageEntry(id=entry_id, parent_id=parent_id, message=UserMessage(content=text))


# ── 测试用例 ────────────────────────────────────────────────────────────────


# 不变量 1：追加不变量 —— append 后可读回，且不覆盖已有内容
def test_append_does_not_overwrite(tmp_path) -> None:
    storage = JsonlStorage(tmp_path / "session.jsonl")

    storage.append(_user_entry("m1", None, "你好"))
    storage.append(_user_entry("m2", "m1", "你好吗？"))

    entries = storage.read_all()
    assert len(entries) == 2
    # 两条消息都在，说明追加没有覆盖掉第一条。
    assert [e.id for e in entries] == ["m1", "m2"]


# 不变量 2：顺序不变量 —— read_all 保持 append 顺序
def test_read_all_preserves_order(tmp_path) -> None:
    storage = JsonlStorage(tmp_path / "session.jsonl")

    for i in range(5):
        storage.append(_user_entry(f"m{i}", f"m{i - 1}" if i > 0 else None, f"消息{i}"))

    entries = storage.read_all()
    assert [e.id for e in entries] == ["m0", "m1", "m2", "m3", "m4"]


# 不变量 3：树不变量 —— path_to_entry 沿 parent_id 走回根
def test_path_to_entry_walks_back_to_root() -> None:
    entries = [
        _user_entry("root", None, "根"),
        _user_entry("a", "root", "子 a"),
        _user_entry("b", "a", "子 b"),
    ]

    path = path_to_entry(entries, "b")
    # 根 -> a -> b，包含叶子本身。
    assert [e.id for e in path] == ["root", "a", "b"]

    # 找不到父节点时抛错。
    entries_broken = [_user_entry("root", None, "根"), _user_entry("a", "missing", "悬空")]
    try:
        path_to_entry(entries_broken, "a")
    except TreeError:
        pass
    else:
        raise AssertionError("父节点悬空时应该抛 TreeError")


# 不变量 4：坏行容忍不变量 —— 未知 type / 坏格式行跳过，其余正常解析
def test_bad_lines_are_skipped(tmp_path) -> None:
    storage = JsonlStorage(tmp_path / "session.jsonl")

    # 正常条目。
    storage.append(_user_entry("m1", None, "你好"))
    # 手工追加一行未知 type 的 JSON。
    with storage.path.open("a", encoding="utf-8") as handle:
        handle.write('{"id": "x1", "type": "unknown_thing", "label": "看不懂"}\n')
    # 再追加一条正常条目。
    storage.append(_user_entry("m2", "m1", "你好吗？"))

    entries = storage.read_all()
    # 未知 type 的行被跳过，两条正常条目保留。
    assert [e.id for e in entries] == ["m1", "m2"]

    # entry_from_line 对未知 type 返回 None。
    assert entry_from_line('{"id": "x", "type": "bogus"}', line_no=1) is None
    # 对坏 JSON 抛 JsonlLineError 并带行号。
    try:
        entry_from_line("{不是 JSON", line_no=3)
    except JsonlLineError as error:
        assert "第 3 行" in str(error)
    else:
        raise AssertionError("坏 JSON 应该抛 JsonlLineError")


# 不变量 5：回放不变量 —— from_entries 回放 messages/model/label，且是 AgentMessage 实例
def test_from_entries_replays_state() -> None:
    entries = [
        _user_entry("root", None, "你好"),
        MessageEntry(
            id="a",
            parent_id="root",
            message=AssistantMessage(
                content=[TextContent(text="你好！"), TextContent(text="有什么可以帮你？")]
            ),
        ),
        MessageEntry(
            id="b",
            parent_id="a",
            message=ToolResultMessage(
                tool_call_id="c1", tool_name="read", content=[TextContent(text="文件内容")]
            ),
        ),
        ModelChangeEntry(id="c", parent_id="b", model="deepseek-flash", provider="deepseek"),
        LabelEntry(id="d", parent_id="c", label="重构 session 层"),
    ]

    state = SessionState.from_entries(entries)

    # 消息回放成真正的 AgentMessage 实例，且类型正确。
    assert isinstance(state.messages[0], UserMessage)
    assert state.messages[0].text == "你好"
    assert isinstance(state.messages[1], AssistantMessage)
    assert state.messages[1].text == "你好！有什么可以帮你？"
    assert isinstance(state.messages[2], ToolResultMessage)
    assert state.messages[2].tool_call_id == "c1"
    # model / label 被覆盖成最新值。
    assert state.model == "deepseek-flash"
    assert state.label == "重构 session 层"


# 不变量 5 补充：指定 leaf_id 时只回放根到该叶子的路径
def test_from_entries_with_leaf_replays_path_only() -> None:
    entries = [
        _user_entry("root", None, "你好"),
        _user_entry("a", "root", "第一条"),
        ModelChangeEntry(id="b", parent_id="a", model="deepseek-flash"),
        LabelEntry(id="c", parent_id="b", label="标签"),
        _user_entry("d", "c", "叶子之后的消息"),
    ]

    # 只回放到叶子 c：后面的消息 d 不应出现。
    state = SessionState.from_entries(entries, leaf_id="c")
    assert [m.text for m in state.messages] == ["你好", "第一条"]
    assert state.model == "deepseek-flash"
    assert state.label == "标签"
