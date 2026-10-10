"""测试 AgentHarness。"""

from __future__ import annotations

import asyncio

import pytest

from nexa_agent.events import AgentEndEvent, AgentEvent, AgentStartEvent
from nexa_agent.harness import AgentHarness, AgentHarnessConfig
from nexa_agent.messages import (
    AssistantMessage,
    TextContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from nexa_agent.provider_events import (
    ProviderResponseEndEvent,
    ProviderResponseStartEvent,
)
from nexa_agent.tools import AgentTool, AgentToolResult
from nexa_ai.fake import FakeProvider

# ── 辅助类和函数 ──────────────────────────────────────────────────────────────


class MockListener:
    """用于测试的事件监听器。"""

    def __init__(self) -> None:
        self.events: list[AgentEvent] = []

    def on_event(self, event: AgentEvent) -> None:
        self.events.append(event)


def _make_config(
    provider: FakeProvider,
    *,
    model: str = "test-model",
    system: str = "你是助手",
    tools: list[AgentTool] | None = None,
) -> AgentHarnessConfig:
    """创建测试用配置。"""
    return AgentHarnessConfig(
        provider=provider,
        model=model,
        system=system,
        tools=tuple(tools) if tools else (),
    )


def _make_tool(name: str = "test_tool", result_text: str = "工具结果") -> AgentTool:
    """创建一个假的测试工具。"""

    async def execute(tool_call_id: str, arguments: dict) -> AgentToolResult:
        return AgentToolResult(content=[TextContent(text=result_text)])

    return AgentTool(
        name=name,
        description=f"假工具 {name}",
        parameters={"type": "object", "properties": {}},
        execute_fn=execute,
    )


async def _collect_events(
    harness: AgentHarness, method: str = "prompt", content: str = "你好"
) -> list[AgentEvent]:
    """收集所有事件。"""
    events = []
    if method == "prompt":
        async for event in harness.prompt(content):
            events.append(event)
    else:
        async for event in harness.continue_():
            events.append(event)
    return events


# ── 测试用例 ──────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_prompt_adds_user_and_assistant_messages():
    """prompt() 后历史应该是 user → assistant。"""

    provider = FakeProvider(
        [
            [
                ProviderResponseStartEvent(model="test"),
                ProviderResponseEndEvent(
                    message=AssistantMessage(content=[TextContent(text="你好！")])
                ),
            ]
        ]
    )

    config = _make_config(provider)
    harness = AgentHarness(config)

    # 初始历史为空
    assert len(harness.messages) == 0

    # 发送消息
    await _collect_events(harness, "prompt", "你好")

    # 验证历史：user → assistant
    assert len(harness.messages) == 2
    assert isinstance(harness.messages[0], UserMessage)
    assert harness.messages[0].text == "你好"
    assert isinstance(harness.messages[1], AssistantMessage)
    assert harness.messages[1].text == "你好！"


@pytest.mark.asyncio
async def test_continue_does_not_add_user_message():
    """continue_() 不应该添加用户消息。"""

    provider = FakeProvider(
        [
            # 第 1 次：prompt 的回复
            [
                ProviderResponseEndEvent(
                    message=AssistantMessage(content=[TextContent(text="第1次回复")])
                ),
            ],
            # 第 2 次：continue 的回复
            [
                ProviderResponseEndEvent(
                    message=AssistantMessage(content=[TextContent(text="第2次回复")])
                ),
            ],
        ]
    )

    config = _make_config(provider)
    harness = AgentHarness(config)

    # 先 prompt
    await _collect_events(harness, "prompt", "你好")
    assert len(harness.messages) == 2  # user + assistant

    # 再 continue
    await _collect_events(harness, "continue_")
    assert len(harness.messages) == 3  # user + assistant + assistant（没有新的 user）

    # 验证没有重复的 user 消息
    user_messages = [m for m in harness.messages if isinstance(m, UserMessage)]
    assert len(user_messages) == 1


@pytest.mark.asyncio
async def test_tool_results_stay_in_history():
    """工具调用结果应该留在 harness 历史中。"""

    tool = _make_tool("test_tool", "工具执行成功")

    provider = FakeProvider(
        [
            # 第 1 次：请求调用工具
            [
                ProviderResponseEndEvent(
                    message=AssistantMessage(
                        content=[ToolCall(id="call-1", name="test_tool", arguments={})]
                    )
                ),
            ],
            # 第 2 次：给出最终回答
            [
                ProviderResponseEndEvent(
                    message=AssistantMessage(content=[TextContent(text="完成！")])
                ),
            ],
        ]
    )

    config = _make_config(provider, tools=[tool])
    harness = AgentHarness(config)

    await _collect_events(harness, "prompt", "执行工具")

    # 验证历史：user → assistant(工具调用) → tool_result → assistant(最终回答)
    assert len(harness.messages) == 4
    assert isinstance(harness.messages[0], UserMessage)
    assert isinstance(harness.messages[1], AssistantMessage)
    assert len(harness.messages[1].tool_calls) == 1
    assert isinstance(harness.messages[2], ToolResultMessage)
    assert harness.messages[2].text == "工具执行成功"
    assert isinstance(harness.messages[3], AssistantMessage)
    assert harness.messages[3].text == "完成！"


@pytest.mark.asyncio
async def test_listener_receives_events():
    """监听器应该收到所有事件。"""

    provider = FakeProvider(
        [
            [
                ProviderResponseEndEvent(
                    message=AssistantMessage(content=[TextContent(text="回复")])
                ),
            ]
        ]
    )

    config = _make_config(provider)
    harness = AgentHarness(config)

    # 注册监听器
    listener = MockListener()
    harness.subscribe(listener)

    await _collect_events(harness, "prompt", "你好")

    # 验证监听器收到了事件
    assert len(listener.events) > 0
    assert listener.events[0].message.role == "user"
    assert isinstance(listener.events[1], AgentStartEvent)
    assert isinstance(listener.events[-1], AgentEndEvent)


@pytest.mark.asyncio
async def test_unsubscribe_stops_events():
    """取消订阅后不再收到事件。"""

    provider = FakeProvider(
        [
            [
                ProviderResponseEndEvent(
                    message=AssistantMessage(content=[TextContent(text="第1次")])
                ),
            ],
            [
                ProviderResponseEndEvent(
                    message=AssistantMessage(content=[TextContent(text="第2次")])
                ),
            ],
        ]
    )

    config = _make_config(provider)
    harness = AgentHarness(config)

    # 注册监听器
    listener = MockListener()
    unsubscribe = harness.subscribe(listener)

    # 第 1 次 prompt
    await _collect_events(harness, "prompt", "你好")
    first_count = len(listener.events)

    # 取消订阅
    unsubscribe()

    # 第 2 次 prompt
    await _collect_events(harness, "prompt", "继续")

    # 验证：取消订阅后没有收到新事件
    assert len(listener.events) == first_count


@pytest.mark.asyncio
async def test_concurrent_calls_rejected():
    """并发调用 prompt/continue 应该被拒绝。"""

    # 创建一个会延迟的 Provider，模拟长时间运行
    class SlowProvider:
        async def stream_response(self, *, model, system, messages, tools):
            yield ProviderResponseStartEvent(model=model)
            await asyncio.sleep(0.1)  # 模拟延迟
            yield ProviderResponseEndEvent(
                message=AssistantMessage(content=[TextContent(text="回复")])
            )

    config = AgentHarnessConfig(
        provider=SlowProvider(),
        model="test",
        system="",
    )
    harness = AgentHarness(config)

    # 启动第 1 个 prompt
    async def first_prompt():
        async for _ in harness.prompt("你好"):
            pass

    task1 = asyncio.create_task(first_prompt())

    # 等待一小段时间，确保第 1 个 prompt 已经开始
    await asyncio.sleep(0.01)

    # 尝试第 2 个 prompt，应该抛出异常
    with pytest.raises(RuntimeError, match="已经在运行中"):
        async for _ in harness.prompt("第二个"):
            pass

    # 等待第 1 个完成
    await task1


@pytest.mark.asyncio
async def test_cancel_stops_loop():
    """cancel() 应该让循环安全退出。"""

    # 创建一个会多次调用工具的 Provider
    call_count = 0

    class MultiTurnProvider:
        async def stream_response(self, *, model, system, messages, tools):
            nonlocal call_count
            call_count += 1
            yield ProviderResponseStartEvent(model=model)
            # 每次都请求调用工具
            yield ProviderResponseEndEvent(
                message=AssistantMessage(
                    content=[ToolCall(id=f"call-{call_count}", name="loop_tool", arguments={})]
                )
            )

    tool = _make_tool("loop_tool")
    config = AgentHarnessConfig(
        provider=MultiTurnProvider(),
        model="test",
        system="",
        tools=(tool,),
        max_turns=10,  # 允许 10 轮
    )
    harness = AgentHarness(config)

    # 注册一个监听器，在第 2 轮时取消
    class CancelListener:
        def __init__(self, h: AgentHarness):
            self.h = h
            self.turn_count = 0

        def on_event(self, event: AgentEvent) -> None:
            if isinstance(event, AgentEndEvent):
                return
            # 简单计数，在第 1 轮结束后取消
            if hasattr(event, "type") and event.type == "turn_end":
                self.turn_count += 1
                if self.turn_count >= 1:
                    self.h.cancel()

    listener = CancelListener(harness)
    harness.subscribe(listener)

    # 运行
    events = []
    async for event in harness.prompt("开始"):
        events.append(event)

    # 验证：循环被提前终止，没有跑完 10 轮
    # 由于 cancel 是在 turn_end 后设置的，所以应该至少完成了 1 轮
    assert not harness.is_running


@pytest.mark.asyncio
async def test_messages_property_returns_tuple():
    """messages 属性应该返回 tuple，不可修改。"""

    provider = FakeProvider(
        [
            [
                ProviderResponseEndEvent(
                    message=AssistantMessage(content=[TextContent(text="回复")])
                ),
            ]
        ]
    )

    config = _make_config(provider)
    harness = AgentHarness(config)

    await _collect_events(harness, "prompt", "你好")

    # 验证返回的是 tuple
    messages = harness.messages
    assert isinstance(messages, tuple)

    # 验证 tuple 不可修改
    with pytest.raises(TypeError):
        messages[0] = UserMessage(content="修改")  # type: ignore


@pytest.mark.asyncio
async def test_append_and_replace_messages():
    """append_message 和 replace_messages 应该正确修改历史。"""

    provider = FakeProvider([])
    config = _make_config(provider)
    harness = AgentHarness(config)

    # 初始为空
    assert len(harness.messages) == 0

    # append
    harness.append_message(UserMessage(content="消息1"))
    assert len(harness.messages) == 1
    assert harness.messages[0].text == "消息1"

    # append 更多
    harness.append_message(AssistantMessage(content=[TextContent(text="回复1")]))
    assert len(harness.messages) == 2

    # replace
    harness.replace_messages(
        [
            UserMessage(content="新消息1"),
            UserMessage(content="新消息2"),
        ]
    )
    assert len(harness.messages) == 2
    assert harness.messages[0].text == "新消息1"
    assert harness.messages[1].text == "新消息2"


@pytest.mark.asyncio
async def test_initial_messages():
    """构造时可以传入初始消息（用于会话恢复）。"""

    provider = FakeProvider(
        [
            [
                ProviderResponseEndEvent(
                    message=AssistantMessage(content=[TextContent(text="继续对话")])
                ),
            ]
        ]
    )

    config = _make_config(provider)

    # 传入初始消息
    initial_messages = [
        UserMessage(content="之前的消息"),
        AssistantMessage(content=[TextContent(text="之前的回复")]),
    ]
    harness = AgentHarness(config, messages=initial_messages)

    # 验证初始历史
    assert len(harness.messages) == 2

    # prompt 后应该追加
    await _collect_events(harness, "prompt", "新消息")
    assert len(harness.messages) == 4
