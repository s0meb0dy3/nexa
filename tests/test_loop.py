"""测试 AgentLoop 主循环。"""

from __future__ import annotations

import pytest

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
from nexa_agent.loop import AgentLoop
from nexa_agent.messages import (
    AssistantMessage,
    TextContent,
    ToolCall,
    UserMessage,
)
from nexa_agent.provider_events import (
    ProviderDeltaEvent,
    ProviderErrorEvent,
    ProviderResponseEndEvent,
    ProviderResponseStartEvent,
)
from nexa_agent.tools import AgentTool, AgentToolResult
from nexa_ai.fake import FakeProvider

# ── 辅助函数 ──────────────────────────────────────────────────────────────────


def _make_tool(name: str = "fake_tool", result_text: str = "工具结果") -> AgentTool:
    """创建一个假的测试工具。"""

    async def execute(tool_call_id: str, arguments: dict) -> AgentToolResult:
        return AgentToolResult(content=[TextContent(text=result_text)])

    return AgentTool(
        name=name,
        description=f"假工具 {name}",
        parameters={"type": "object", "properties": {}},
        execute_fn=execute,
    )


async def _collect_events(agent: AgentLoop, **kwargs) -> list:
    """运行 Agent 并收集所有事件。"""

    events = []
    async for event in agent.run(
        model="test-model",
        system="你是助手",
        messages=[UserMessage(content="你好")],
        **kwargs,
    ):
        events.append(event)
    return events


# ── 测试用例 ──────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_loop_without_tools():
    """模型直接回复文字，不调用工具。"""

    # 准备：Provider 只返回一次文字回复
    provider = FakeProvider(
        [
            [
                ProviderResponseStartEvent(model="test"),
                ProviderResponseEndEvent(
                    message=AssistantMessage(content=[TextContent(text="你好！我是助手。")])
                ),
            ]
        ]
    )

    agent = AgentLoop(provider)
    events = await _collect_events(agent, tools=[])

    # 验证：有开始和结束事件
    assert isinstance(events[0], AgentStartEvent)
    assert isinstance(events[-1], AgentEndEvent)

    # 验证：只有 1 轮（没有工具调用，不需要多轮）
    turn_starts = [e for e in events if isinstance(e, TurnStartEvent)]
    assert len(turn_starts) == 1

    # 验证：没有工具执行事件
    tool_events = [e for e in events if isinstance(e, ToolExecutionStartEvent)]
    assert len(tool_events) == 0

    # 验证：最终消息包含助手的回复
    end_event = events[-1]
    assert isinstance(end_event, AgentEndEvent)
    assistant_msgs = [m for m in end_event.messages if isinstance(m, AssistantMessage)]
    assert assistant_msgs[-1].text == "你好！我是助手。"


@pytest.mark.asyncio
async def test_loop_forwards_deltas_between_start_and_end():
    """流式：MessageStart（空）→ 逐段 MessageDelta → MessageEnd（完整消息）。"""

    provider = FakeProvider(
        [
            [
                ProviderResponseStartEvent(model="test"),
                ProviderDeltaEvent(kind="text", delta="你"),
                ProviderDeltaEvent(kind="reasoning", delta="想"),
                ProviderDeltaEvent(kind="text", delta="好"),
                ProviderResponseEndEvent(
                    message=AssistantMessage(content=[TextContent(text="你好")])
                ),
            ]
        ]
    )

    agent = AgentLoop(provider)
    events = await _collect_events(agent, tools=[])

    # 找到消息生命周期这一段。
    starts = [i for i, e in enumerate(events) if isinstance(e, MessageStartEvent)]
    ends = [i for i, e in enumerate(events) if isinstance(e, MessageEndEvent)]
    assert len(starts) == 1 and len(ends) == 1
    assert starts[0] < ends[0]

    # MessageStart 携带空消息（只表示"开始"）。
    assert events[starts[0]].message.text == ""

    # 中间的 delta 按顺序、按 kind 转发。
    deltas = [e for e in events if isinstance(e, MessageDeltaEvent)]
    assert [(d.kind, d.delta) for d in deltas] == [
        ("text", "你"),
        ("reasoning", "想"),
        ("text", "好"),
    ]

    # MessageEnd 携带完整消息。
    assert events[ends[0]].message.text == "你好"


@pytest.mark.asyncio
async def test_loop_with_tool_call():
    """模型调用工具，然后给出最终回答。"""

    # 准备：一个假工具
    tool = _make_tool("test_tool", "工具执行成功")

    # 准备：Provider 返回两次
    # 第 1 次：请求调用工具
    # 第 2 次：给出最终回答
    provider = FakeProvider(
        [
            [
                ProviderResponseEndEvent(
                    message=AssistantMessage(
                        content=[ToolCall(id="call-1", name="test_tool", arguments={})]
                    )
                ),
            ],
            [
                ProviderResponseEndEvent(
                    message=AssistantMessage(content=[TextContent(text="工具已执行完毕！")])
                ),
            ],
        ]
    )

    agent = AgentLoop(provider)
    events = await _collect_events(agent, tools=[tool])

    # 验证：有 2 轮
    turn_starts = [e for e in events if isinstance(e, TurnStartEvent)]
    assert len(turn_starts) == 2

    # 验证：有工具执行事件
    tool_starts = [e for e in events if isinstance(e, ToolExecutionStartEvent)]
    assert len(tool_starts) == 1
    assert tool_starts[0].tool_name == "test_tool"

    tool_ends = [e for e in events if isinstance(e, ToolExecutionEndEvent)]
    assert len(tool_ends) == 1
    assert tool_ends[0].result.text == "工具执行成功"
    assert tool_ends[0].is_error is False


@pytest.mark.asyncio
async def test_loop_unknown_tool():
    """模型请求一个不存在的工具，应该返回错误信息。"""

    # 准备：Provider 请求一个不存在的工具，然后给出回答
    provider = FakeProvider(
        [
            [
                ProviderResponseEndEvent(
                    message=AssistantMessage(
                        content=[ToolCall(id="call-1", name="not_exist_tool", arguments={})]
                    )
                ),
            ],
            [
                ProviderResponseEndEvent(
                    message=AssistantMessage(content=[TextContent(text="抱歉，工具不存在。")])
                ),
            ],
        ]
    )

    agent = AgentLoop(provider)
    events = await _collect_events(agent, tools=[])

    # 验证：工具执行标记为错误
    tool_ends = [e for e in events if isinstance(e, ToolExecutionEndEvent)]
    assert len(tool_ends) == 1
    assert tool_ends[0].is_error is True
    assert "未知工具" in tool_ends[0].result.text


@pytest.mark.asyncio
async def test_loop_max_turns():
    """达到最大轮次时应该停止循环。"""

    # 准备：Provider 每次都请求调用工具（死循环场景）
    provider = FakeProvider(
        [
            [
                ProviderResponseEndEvent(
                    message=AssistantMessage(
                        content=[ToolCall(id=f"call-{i}", name="loop_tool", arguments={})]
                    )
                ),
            ]
            for i in range(5)  # 准备 5 轮，但 max_turns=3
        ]
    )

    tool = _make_tool("loop_tool")
    agent = AgentLoop(provider, max_turns=3)
    events = await _collect_events(agent, tools=[tool])

    # 验证：只执行了 3 轮
    turn_starts = [e for e in events if isinstance(e, TurnStartEvent)]
    assert len(turn_starts) == 3


@pytest.mark.asyncio
async def test_loop_provider_error():
    """Provider 返回错误时，应该把错误转成助手消息。"""

    # 准备：Provider 返回一个错误
    provider = FakeProvider(
        [
            [
                ProviderErrorEvent(message="网络连接失败"),
            ]
        ]
    )

    agent = AgentLoop(provider)
    events = await _collect_events(agent, tools=[])

    # 验证：只有 1 轮（错误后直接结束）
    turn_starts = [e for e in events if isinstance(e, TurnStartEvent)]
    assert len(turn_starts) == 1

    # 验证：最终消息中包含错误信息
    end_event = events[-1]
    assert isinstance(end_event, AgentEndEvent)
    assistant_msgs = [m for m in end_event.messages if isinstance(m, AssistantMessage)]
    assert "错误" in assistant_msgs[-1].text
    assert "网络连接失败" in assistant_msgs[-1].text


@pytest.mark.asyncio
async def test_loop_multiple_tool_calls():
    """模型一次请求多个工具调用。"""

    # 准备：两个假工具
    tool_a = _make_tool("tool_a", "A 的结果")
    tool_b = _make_tool("tool_b", "B 的结果")

    # 准备：Provider 第 1 次请求两个工具，第 2 次给出最终回答
    provider = FakeProvider(
        [
            [
                ProviderResponseEndEvent(
                    message=AssistantMessage(
                        content=[
                            ToolCall(id="call-1", name="tool_a", arguments={}),
                            ToolCall(id="call-2", name="tool_b", arguments={}),
                        ]
                    )
                ),
            ],
            [
                ProviderResponseEndEvent(
                    message=AssistantMessage(content=[TextContent(text="两个工具都执行完了！")])
                ),
            ],
        ]
    )

    agent = AgentLoop(provider)
    events = await _collect_events(agent, tools=[tool_a, tool_b])

    # 验证：有 2 次工具执行
    tool_starts = [e for e in events if isinstance(e, ToolExecutionStartEvent)]
    assert len(tool_starts) == 2
    assert tool_starts[0].tool_name == "tool_a"
    assert tool_starts[1].tool_name == "tool_b"

    tool_ends = [e for e in events if isinstance(e, ToolExecutionEndEvent)]
    assert len(tool_ends) == 2
    assert tool_ends[0].result.text == "A 的结果"
    assert tool_ends[1].result.text == "B 的结果"


@pytest.mark.asyncio
async def test_loop_preserves_external_messages():
    """AgentLoop 不应该修改外部传入的消息列表。"""

    provider = FakeProvider(
        [
            [
                ProviderResponseEndEvent(
                    message=AssistantMessage(content=[TextContent(text="回复")])
                ),
            ]
        ]
    )

    agent = AgentLoop(provider)
    external_messages = [UserMessage(content="你好")]
    original_len = len(external_messages)

    async for _ in agent.run(
        model="test",
        system="",
        messages=external_messages,
        tools=[],
    ):
        pass

    # 验证：外部消息列表没有被修改
    assert len(external_messages) == original_len
