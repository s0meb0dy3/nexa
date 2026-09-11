"""最小 Agent 演示：接入 DeepSeek，使用工具，展示完整循环过程。"""

from __future__ import annotations

import asyncio
import datetime

from nexa_agent.events import (
    AgentEndEvent,
    AgentStartEvent,
    MessageStartEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    TurnEndEvent,
    TurnStartEvent,
)
from nexa_agent.loop import AgentLoop
from nexa_agent.messages import AgentMessage, UserMessage
from nexa_agent.tools import AgentTool, AgentToolResult
from nexa_coding.cli import resolve_provider

# ── 工具定义 ──────────────────────────────────────────────────────────────────


async def get_current_time(tool_call_id: str, arguments: dict) -> AgentToolResult:
    """获取当前时间。"""

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return AgentToolResult(content=[{"type": "text", "text": f"当前时间是: {now}"}])


async def calculate_string_length(tool_call_id: str, arguments: dict) -> AgentToolResult:
    """计算字符串长度。"""

    text = arguments.get("text", "")
    length = len(text)
    return AgentToolResult(content=[{"type": "text", "text": f'字符串 "{text}" 的长度是 {length}'}])


# ── 主流程 ────────────────────────────────────────────────────────────────────


async def main() -> None:
    # 从 ~/.nexa/config.toml 解析供应商档案（默认档案）。
    provider, profile = resolve_provider()

    # 创建 Agent 循环，最多允许 5 轮。
    agent = AgentLoop(provider, max_turns=5)

    # 定义工具列表。
    tools = [
        AgentTool(
            name="get_current_time",
            description="获取当前的日期和时间",
            parameters={"type": "object", "properties": {}, "required": []},
            execute_fn=get_current_time,
        ),
        AgentTool(
            name="calculate_string_length",
            description="计算给定字符串的长度",
            parameters={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "要计算长度的字符串",
                    }
                },
                "required": ["text"],
            },
            execute_fn=calculate_string_length,
        ),
    ]

    # 用户消息。
    messages: list[AgentMessage] = [
        UserMessage(content="请告诉我现在几点了，然后帮我算一下 'Hello, 世界！' 这个字符串有多长。")
    ]

    print("=" * 60)
    print(f"🚀 最小 Agent 演示 (Provider: {profile.name})")
    print("=" * 60)

    # 运行 Agent 循环，逐个接收事件。
    async for event in agent.run(
        model=profile.model,
        system="你是一个有用的助手，可以查看时间和计算字符串长度。请用中文回答。",
        messages=messages,
        tools=tools,
    ):
        if isinstance(event, AgentStartEvent):
            print("\n📍 Agent 开始运行")

        elif isinstance(event, TurnStartEvent):
            print("\n🔄 ── 新一轮开始 ──")

        elif isinstance(event, MessageStartEvent):
            # 助手消息到达。
            text = event.message.text
            if text:
                print(f"\n🤖 助手回复: {text}")
            tool_calls = event.message.tool_calls
            if tool_calls:
                for tc in tool_calls:
                    print(f"🔧 请求工具: {tc.name}({tc.arguments})")

        elif isinstance(event, ToolExecutionStartEvent):
            print(f"  ⏳ 执行工具: {event.tool_name}({event.args})")

        elif isinstance(event, ToolExecutionEndEvent):
            print(f"  ✅ 工具结果: {event.result.text}")

        elif isinstance(event, TurnEndEvent):
            print("🔄 ── 新一轮结束 ──")

        elif isinstance(event, AgentEndEvent):
            print("\n📍 Agent 运行结束")
            # 打印最终回复。
            for msg in reversed(event.messages):
                if hasattr(msg, "text") and msg.text and msg.role == "assistant":
                    print(f"\n💬 最终回复: {msg.text}")
                    break

    print("\n" + "=" * 60)
    print("✅ 演示完成")


if __name__ == "__main__":
    asyncio.run(main())
