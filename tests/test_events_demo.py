"""用可运行的示例理解 NEXA 的事件机制。

这个文件不是传统意义上的"测试"，而是一组可以运行的教学示例：
每个示例演示事件机制的一个知识点，跑完后打印出真实的事件序列。

运行方式：
    uv run pytest tests/test_events_demo.py -v -s
    （-s 让 print 输出直接显示出来）

知识点对应关系：
    示例 1：事件是什么 —— 每个 AgentEvent 都是 Pydantic 模型，靠 type 字段区分
    示例 2：事件流怎么来 —— yield 产出事件，async for 消费事件
    示例 3：一次完整运行的事件顺序 —— agent_start → turn → ... → agent_end
    示例 4：工具调用会产生哪些事件 —— start/end 一一对应
    示例 5：两套事件体系 —— ProviderEvent（模型语言）被翻译成 AgentEvent（生命周期）
    示例 6：discriminator —— Pydantic 根据 type 字段自动选择解析成哪个类
    示例 7：事件的消费方式 —— async for（拉）和 subscribe（推）
    示例 8：取消机制 —— 事件检查点也是循环的安全退出点
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from nexa_agent.events import (
    AgentEndEvent,
    AgentEvent,
    AgentStartEvent,
    MessageEndEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    TurnStartEvent,
)
from nexa_agent.harness import AgentHarness, AgentHarnessConfig
from nexa_agent.loop import AgentLoop
from nexa_agent.messages import AssistantMessage, TextContent, ToolCall, UserMessage
from nexa_agent.provider_events import (
    ProviderDeltaEvent,
    ProviderResponseEndEvent,
    ProviderResponseStartEvent,
)
from nexa_agent.tools import AgentTool, AgentToolResult
from nexa_ai.fake import FakeProvider

# ── 示例公共部分：准备一个假 Provider 和一个假工具 ─────────────────────────────


def _make_tool(name: str = "echo", result_text: str = "工具执行完成") -> AgentTool:
    """创建一个假工具：不管收到什么参数，都返回固定的文字。"""

    async def execute(tool_call_id: str, arguments: dict) -> AgentToolResult:
        return AgentToolResult(content=[TextContent(text=result_text)])

    return AgentTool(
        name=name,
        description="假工具，用于演示",
        parameters={"type": "object", "properties": {}},
        execute_fn=execute,
    )


def _provider_replying_text(text: str) -> FakeProvider:
    """创建一个假 Provider：模拟模型直接回复一段文字（不调用工具）。"""

    return FakeProvider(
        [
            [
                ProviderResponseStartEvent(model="test-model"),
                ProviderDeltaEvent(delta="你好"),
                ProviderDeltaEvent(delta="，"),
                ProviderDeltaEvent(delta=text),
                ProviderResponseEndEvent(
                    message=AssistantMessage(content=[TextContent(text=f"你好，{text}")])
                ),
            ]
        ]
    )


def _provider_calling_tool(result_text: str = "42") -> FakeProvider:
    """创建一个假 Provider：第一轮请求调用工具，第二轮请求给出最终回答。"""

    return FakeProvider(
        [
            # 第 1 轮：模型说"我要调用工具 echo"
            [
                ProviderResponseStartEvent(model="test-model"),
                ProviderResponseEndEvent(
                    message=AssistantMessage(
                        content=[
                            ToolCall(
                                id="call_001",
                                name="echo",
                                arguments={"question": "生命的意义"},
                            )
                        ]
                    )
                ),
            ],
            # 第 2 轮：模型拿到工具结果后，给出最终回答
            [
                ProviderResponseStartEvent(model="test-model"),
                ProviderResponseEndEvent(
                    message=AssistantMessage(content=[TextContent(text=f"答案是 {result_text}")])
                ),
            ],
        ]
    )


async def _collect(events_aiter: AsyncIterator[AgentEvent]) -> list[AgentEvent]:
    """把事件流收集成列表，方便检查。"""
    return [event async for event in events_aiter]


def _type_names(events: list[AgentEvent]) -> list[str]:
    """把事件列表转成 type 字符串列表，方便打印和对比。"""
    return [event.type for event in events]


# ── 示例 1：事件是什么 ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_01_what_is_an_event():
    """事件 = 一个带 type 字段的 Pydantic 模型。

    每种事件是一个类，类里所有实例的 type 都相同（Literal 锁死）。
    所以判断"这是什么事件"有两种等价方式：
        isinstance(event, TurnStartEvent)   # 按类判断
        event.type == "turn_start"          # 按字段判断
    """

    start = AgentStartEvent()
    end = AgentEndEvent(messages=[])

    # type 字段是自动填好的，不用手动传
    assert start.type == "agent_start"
    assert end.type == "agent_end"

    # 事件是 Pydantic 模型，可以直接转成 JSON（持久化/网络传输时有用）
    assert AgentStartEvent().model_dump_json() == '{"type":"agent_start"}'

    # 事件也可以携带数据，比如 AgentEndEvent 带着全部对话历史
    end_with_history = AgentEndEvent(
        messages=[UserMessage(content="你好"), AssistantMessage(content="你好！")]
    )
    assert len(end_with_history.messages) == 2

    print("事件示例：", start, end_with_history)


# ── 示例 2：事件流怎么来 ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_02_event_stream_basics():
    """AgentLoop.run() 返回的是异步生成器（异步事件流）。

    调用方用 async for 逐个取事件 —— 事件产生一个就能处理一个，
    不用等整个 agent 运行结束（这就是"流"的意义）。
    """

    provider = _provider_replying_text("我是助手")
    agent = AgentLoop(provider)

    # run() 不是普通函数：直接调用它得到的是一个事件流，不是结果
    stream = agent.run(
        model="test-model",
        system="你是助手",
        messages=[UserMessage(content="你好")],
        tools=[],
    )

    # 逐个消费事件，边收边数
    count = 0
    types: list[str] = []
    async for event in stream:
        count += 1
        types.append(event.type)

    # 模型直接回复、没有工具调用。流式下每个增量都会转发：
    # agent_start → turn_start → message_start → delta×3 → message_end
    # → turn_end → agent_end
    print("收到的事件序列：", types)
    assert types == [
        "agent_start",
        "turn_start",
        "message_start",
        "message_delta",
        "message_delta",
        "message_delta",
        "message_end",
        "turn_end",
        "agent_end",
    ]
    assert count == 9


# ── 示例 3：一次完整运行的事件顺序 ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_03_full_event_order_no_tools():
    """没有工具调用时的事件顺序（把流程从头到尾看一遍）。

    AgentLoop 的工作流程映射到事件上就是：
        agent 开始 → 每轮开始 → 模型消息开始/结束 → 轮结束 → agent 结束
    """

    provider = _provider_replying_text("1 加 1 等于 2")
    agent = AgentLoop(provider)
    events = await _collect(
        agent.run(
            model="test-model",
            system="你是数学老师",
            messages=[UserMessage(content="1+1=?")],
            tools=[],
        )
    )

    # 事件的首尾是固定的：AgentStartEvent 开场，AgentEndEvent 收尾
    assert isinstance(events[0], AgentStartEvent)
    assert isinstance(events[-1], AgentEndEvent)

    # AgentEndEvent.messages 携带完整对话历史：用户消息 + 助手回复
    history = events[-1].messages
    assert [m.role for m in history] == ["user", "assistant"]

    # 从 MessageEndEvent 里能拿到完整的助手回复
    # （假 Provider 的回复固定是 "你好，" + 指定的文字）
    message_end = next(e for e in events if isinstance(e, MessageEndEvent))
    assert isinstance(message_end.message, AssistantMessage)
    assert message_end.message.text == "你好，1 加 1 等于 2"

    print("事件顺序：", _type_names(events))


# ── 示例 4：工具调用会产生哪些事件 ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_04_tool_call_events():
    """模型要调用工具时，一轮会变成：两轮模型交互 + 工具执行事件。

    完整顺序：
        agent_start
        turn_start                       ← 第 1 轮：模型请求调用工具
        message_start / message_end
        tool_execution_start             ← 开始执行工具
        tool_execution_end               ← 工具执行完毕
        turn_end
        turn_start                       ← 第 2 轮：模型看到结果，给出回答
        message_start / message_end
        turn_end
        agent_end
    """

    provider = _provider_calling_tool()
    agent = AgentLoop(provider)
    events = await _collect(
        agent.run(
            model="test-model",
            system="你是助手",
            messages=[UserMessage(content="生命的意义是什么？")],
            tools=[_make_tool()],
        )
    )

    # 有几轮，就有几个 turn_start
    turn_starts = [e for e in events if isinstance(e, TurnStartEvent)]
    assert len(turn_starts) == 2

    # 工具执行事件成对出现，并且位于第 1 轮里
    tool_start = next(e for e in events if isinstance(e, ToolExecutionStartEvent))
    tool_end = next(e for e in events if isinstance(e, ToolExecutionEndEvent))
    assert tool_start.tool_name == "echo"
    assert tool_start.args == {"question": "生命的意义"}
    assert tool_end.tool_call_id == tool_start.tool_call_id == "call_001"
    assert tool_end.result.text == "工具执行完成"

    # ToolExecutionStartEvent 的位置：紧跟在 message_end 之后（模型请求 → 立刻执行）
    order = _type_names(events)
    assert (
        order.index("message_end") < order.index("tool_execution_start") < order.index("turn_end")
    )

    print("工具调用的事件顺序：", order)


# ── 示例 5：两套事件体系 —— Provider 事件被"翻译"成 Agent 事件 ─────────────────


@pytest.mark.asyncio
async def test_05_two_event_systems():
    """项目里有两套事件，各管一段：

    ProviderEvent（nexa_agent/provider_events.py，5 种）—— 描述"模型正在说话"：
        response_start / delta / tool_call / response_end / error
    AgentEvent（nexa_agent/events.py，10 种）—— 描述"agent 整体在干什么"

    AgentLoop 是翻译层：消费 Provider 事件，产出 Agent 事件。
    流式：每个 delta 被原样转发成 MessageDeltaEvent；完整消息在 message_end。
    """

    provider = _provider_replying_text("我是助手")

    # ── 第一视角：直接看 Provider 层的事件 ──
    provider_stream = provider.stream_response(
        model="test-model",
        system="你是助手",
        messages=[UserMessage(content="你好")],
        tools=[],
    )
    provider_types = []
    async for event in provider_stream:
        provider_types.append(event.type)
    # 模型语言的流：开始 → 3 个增量 → 结束
    assert provider_types == [
        "response_start",
        "delta",
        "delta",
        "delta",
        "response_end",
    ]

    # ── 第二视角：经过 AgentLoop 翻译后的 Agent 事件 ──
    # （换一个新的假 Provider：上一个的预设流已经被第一视角消费完了）
    agent = AgentLoop(_provider_replying_text("我是助手"))
    agent_types = []
    async for event in agent.run(
        model="test-model",
        system="你是助手",
        messages=[UserMessage(content="你好")],
        tools=[],
    ):
        agent_types.append(event.type)
    # 生命周期语言的流：message_start 表示开始，3 个 delta 实时转发，
    # message_end 携带完整消息收尾。
    assert agent_types == [
        "agent_start",
        "turn_start",
        "message_start",
        "message_delta",
        "message_delta",
        "message_delta",
        "message_end",
        "turn_end",
        "agent_end",
    ]

    print("Provider 视角：", provider_types)
    print("Agent 视角：  ", agent_types)


# ── 示例 6：discriminator —— 按 type 字段自动选择事件类 ────────────────────────


@pytest.mark.asyncio
async def test_06_discriminator_parsing():
    """AgentEvent 是"可辨识联合"类型：

        type AgentEvent = Annotated[... | ..., Field(discriminator="type")]

    discriminator="type" 的意思：解析 JSON 时，看 type 字段的值，
    自动决定实例化哪个事件类。这在"从文件/网络恢复事件"时很有用。
    """

    from pydantic import TypeAdapter

    # TypeAdapter 可以对"一个类型注解"做解析和校验，这里解析 AgentEvent 联合类型
    adapter = TypeAdapter(AgentEvent)

    # 一段 JSON，只有 type 字段没有类名 —— Pydantic 靠 type 分辨它是谁
    json_str = '{"type": "turn_start"}'
    event = adapter.validate_json(json_str)

    # 解析出来的已经是具体的类，可以直接 isinstance 判断
    assert isinstance(event, TurnStartEvent)
    assert event.type == "turn_start"

    # 换一个 type，就解析成另一个类
    event2 = adapter.validate_json('{"type": "agent_start"}')
    assert isinstance(event2, AgentStartEvent)

    # type 值不认识时，直接报错 —— 拼写错误藏不住
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        adapter.validate_json('{"type": "turn_startt"}')  # 手滑多打了一个 t


# ── 示例 7：事件的两种消费方式 —— async for（拉）和 subscribe（推） ─────────────


@pytest.mark.asyncio
async def test_07_pull_vs_push():
    """同一个事件流，有两条消费通道：

    拉（async for）：调用方主动逐个取事件，适合写主流程逻辑。
    推（subscribe）：注册监听器，agent 每产生一个事件就推给它，
                     适合日志、持久化这类"旁路"功能。
                     监听器抛异常会停止主流程，避免保存失败却继续运行。
    """

    provider = _provider_replying_text("我是助手")
    config = AgentHarnessConfig(
        provider=provider,
        model="test-model",
        system="你是助手",
    )
    harness = AgentHarness(config)

    # ── 推通道：注册一个监听器，收集推给它的所有事件 ──
    pushed: list[str] = []

    class RecordingListener:
        def on_event(self, event: AgentEvent) -> None:
            pushed.append(event.type)

    unsubscribe = harness.subscribe(RecordingListener())

    # ── 拉通道：正常用 async for 消费 ──
    pulled = await _collect(harness.prompt("你好"))

    # 两条通道收到的是同一批事件
    assert pushed == _type_names(pulled)
    assert "agent_start" in pushed and "agent_end" in pushed

    # 取消订阅后，下一个监听器不再收到事件
    unsubscribe()
    assert len(harness._listeners) == 0

    # ── 监听器失败必须传给调用方 ──
    class BrokenListener:
        def on_event(self, event: AgentEvent) -> None:
            raise RuntimeError("监听器坏了")

    harness2 = AgentHarness(config)
    harness2.subscribe(BrokenListener())  # 注册一个会炸的监听器
    with pytest.raises(RuntimeError, match="监听器坏了"):
        await _collect(harness2.prompt("你好"))
    assert not harness2.is_running


# ── 附：Harness 的取消机制也建立在事件检查点上 ────────────────────────────────


@pytest.mark.asyncio
async def test_08_cancel_at_event_checkpoint():
    """Harness 每转发一个事件前都检查取消标志（harness.py 的 _run_loop）。

    所以事件流不只是"通知"，也是循环的安全退出点：cancel() 设置标志后，
    下一个事件会在检查点被拦下、不再转发——已经送达的事件不受影响，
    正在执行的工作也不会被暴力中断。
    """

    provider = _provider_replying_text("我会被取消")
    config = AgentHarnessConfig(provider=provider, model="test-model", system="你是助手")
    harness = AgentHarness(config)

    seen = 0

    async for event in harness.prompt("你好"):
        seen += 1
        if event.type == "agent_start":
            # 收到第一个事件（agent_start）就请求取消：
            # 它本身已经送达，下一个事件 turn_start 会在检查点被拦下
            harness.cancel()

    # 用户消息先完成，随后 agent_start；后面的循环事件被拦下。
    assert seen == 2
    assert harness.is_running is False

    print(f"取消后只处理了 {seen} 个事件")
