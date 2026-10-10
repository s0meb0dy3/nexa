"""测试 TUI 的事件 → 状态翻译层（不开终端，不依赖 Textual）。"""

from __future__ import annotations

from nexa_agent.events import (
    AgentEndEvent,
    AgentStartEvent,
    MessageDeltaEvent,
    MessageEndEvent,
    MessageStartEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    TurnStartEvent,
)
from nexa_agent.messages import AssistantMessage, TextContent, ThinkingContent
from nexa_agent.tools import AgentToolResult
from nexa_coding.tui.adapter import TuiEventAdapter
from nexa_coding.tui.state import ChatItemKind, TuiState


def _make_adapter() -> tuple[TuiState, TuiEventAdapter]:
    """创建绑定的状态 + 适配器。"""
    state = TuiState()
    return state, TuiEventAdapter(state)


# ── 测试用例 ──────────────────────────────────────────────────────────────────


def test_empty_state_initial_values():
    """空状态初始值正确。"""
    state = TuiState()
    assert state.chat_items == []
    assert state.streaming_text == ""
    assert state.running is False
    assert state.error is None


def test_start_end_events_toggle_running():
    """AgentStart → running=True；AgentEnd → running=False（不反查消息）。"""
    state, adapter = _make_adapter()

    adapter.apply(AgentStartEvent())
    assert state.running is True

    # AgentEnd 只停 running，不往 chat_items 加东西。
    adapter.apply(AgentEndEvent(messages=[]))
    assert state.running is False
    assert state.chat_items == []


def test_assistant_message_pair():
    """MessageStart + MessageEnd 把整条消息合并进 chat_items。"""
    state, adapter = _make_adapter()
    message = AssistantMessage(content=[TextContent(text="你好！")])

    adapter.apply(MessageStartEvent(message=message))
    adapter.apply(MessageEndEvent(message=message))

    # 整条消息合并成一条 assistant 记录。
    assert len(state.chat_items) == 1
    assert state.chat_items[0].kind == ChatItemKind.assistant
    assert state.chat_items[0].text == "你好！"


def test_tool_events_produce_tool_items():
    """工具事件产生 tool item，失败带 error 标记。"""
    state, adapter = _make_adapter()

    adapter.apply(ToolExecutionStartEvent(tool_call_id="c1", tool_name="read", args={}))
    adapter.apply(
        ToolExecutionEndEvent(
            tool_call_id="c1",
            tool_name="read",
            result=AgentToolResult(content=[TextContent(text="文件内容")]),
            is_error=True,
        )
    )

    # 同一次工具调用更新同一个条目，保留完整结果。
    assert len(state.chat_items) == 1
    item = state.chat_items[0]
    assert item.kind == ChatItemKind.tool
    assert "read" in item.text
    assert item.status == "失败"
    assert item.error is True
    assert item.output == "文件内容"
    assert item.elapsed is not None


def test_user_message_added():
    """add_user 把用户输入作为 user 记录。"""
    state, adapter = _make_adapter()

    state.add_user("你好")
    assert len(state.chat_items) == 1
    assert state.chat_items[0].kind == ChatItemKind.user
    assert state.chat_items[0].text == "你好"


def test_ignored_and_unknown_events_do_not_crash():
    """被忽略的事件不崩；未知事件静默忽略，不污染状态。"""
    state, adapter = _make_adapter()

    # TurnStartEvent 应被忽略，不影响任何状态。
    adapter.apply(TurnStartEvent())
    assert state.error is None
    assert state.chat_items == []
    assert state.running is False

    # 一个不是已知事件类型的对象：应静默忽略，不抛错也不记 error。
    class UnknownEvent:
        pass

    adapter.apply(UnknownEvent())  # type: ignore[arg-type]
    assert state.error is None
    assert state.chat_items == []


def test_thinking_and_answer_become_two_items():
    """带思考的消息 → 先 thinking 记录，再 assistant 记录。"""
    state, adapter = _make_adapter()
    message = AssistantMessage(
        content=[ThinkingContent(text="让我想想"), TextContent(text="答案是 2")]
    )

    adapter.apply(MessageEndEvent(message=message))

    assert len(state.chat_items) == 2
    assert state.chat_items[0].kind == ChatItemKind.thinking
    assert state.chat_items[0].text == "让我想想"
    assert state.chat_items[1].kind == ChatItemKind.assistant
    assert state.chat_items[1].text == "答案是 2"


def test_thinking_only_message_still_shows():
    """只有思考、没有正文的消息也要产生 thinking 记录。"""
    state, adapter = _make_adapter()
    message = AssistantMessage(content=[ThinkingContent(text="只在思考")])

    adapter.apply(MessageEndEvent(message=message))

    assert len(state.chat_items) == 1
    assert state.chat_items[0].kind == ChatItemKind.thinking


def test_toggle_thinking_flips_flag():
    """toggle_thinking 在展开/折叠之间翻转。"""
    state = TuiState()

    assert state.show_thinking is False
    state.toggle_thinking()
    assert state.show_thinking is True
    state.toggle_thinking()
    assert state.show_thinking is False


def test_streaming_deltas_accumulate_then_commit():
    """流式：delta 累积到缓冲，MessageEnd 时以完整消息提交。"""
    state, adapter = _make_adapter()

    adapter.apply(MessageStartEvent(message=AssistantMessage()))
    adapter.apply(MessageDeltaEvent(kind="text", delta="你"))
    adapter.apply(MessageDeltaEvent(kind="text", delta="好"))

    # 流式中：内容还在缓冲，尚未成为正式记录。
    assert state.streaming_started is True
    assert state.streaming_text == "你好"
    assert state.chat_items == []

    adapter.apply(MessageEndEvent(message=AssistantMessage(content=[TextContent(text="你好")])))

    # 提交后：缓冲清空，正式记录产生。
    assert state.streaming_started is False
    assert state.streaming_text == ""
    assert len(state.chat_items) == 1
    assert state.chat_items[0].text == "你好"


def test_streaming_reasoning_delta_goes_to_thinking_buffer():
    """kind=reasoning 的 delta 进思考缓冲，与正文分开。"""
    state, adapter = _make_adapter()

    adapter.apply(MessageStartEvent(message=AssistantMessage()))
    adapter.apply(MessageDeltaEvent(kind="reasoning", delta="想"))
    adapter.apply(MessageDeltaEvent(kind="reasoning", delta="想"))
    adapter.apply(MessageDeltaEvent(kind="text", delta="答案"))

    assert state.streaming_thinking == "想想"
    assert state.streaming_text == "答案"


def test_message_start_with_empty_message_enters_streaming():
    """流式的 MessageStart 携带空消息，也应进入流式状态。"""
    state, adapter = _make_adapter()

    adapter.apply(MessageStartEvent(message=AssistantMessage()))

    assert state.streaming_started is True


def test_end_without_text_does_not_add_item():
    """MessageEnd 无正文（例如只调用工具）时只重置状态，不产生记录。"""
    state, adapter = _make_adapter()

    adapter.apply(MessageStartEvent(message=AssistantMessage()))
    adapter.apply(MessageEndEvent(message=AssistantMessage()))

    assert state.chat_items == []
    assert state.streaming_started is False
