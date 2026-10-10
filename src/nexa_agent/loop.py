"""最小的 Agent 循环：调用模型 → 执行工具 → 再调用模型，直到模型不再请求工具。"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import aclosing

from nexa_agent.events import (
    AgentEndEvent,
    AgentEvent,
    AgentStartEvent,
    MessageDeltaEvent,
    MessageEndEvent,
    MessageStartEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    TurnEndEvent,
    TurnStartEvent,
)
from nexa_agent.messages import (
    AgentMessage,
    AssistantMessage,
    TextContent,
    ToolCall,
    ToolResultMessage,
)
from nexa_agent.provider import ModelProvider
from nexa_agent.tools import AgentTool, AgentToolResult


class AgentLoop:
    """最基础的 Agent 循环。

    工作流程：
    1. 把消息发给模型，等待回复。
    2. 如果模型回复中包含工具调用，就执行工具，把结果追加到消息列表。
    3. 带着更新后的消息列表，再次请求模型。
    4. 重复以上步骤，直到模型不再请求工具，或者达到最大轮次。
    """

    def __init__(self, provider: ModelProvider, *, max_turns: int = 50) -> None:
        # provider 是具体的模型后端，例如 DeepSeek、OpenAI 等。
        self.provider = provider
        # max_turns 防止模型反复调用工具，陷入死循环。
        self.max_turns = max_turns

    async def run(
        self,
        *,
        model: str,
        system: str,
        messages: list[AgentMessage],
        tools: list[AgentTool],
    ) -> AsyncGenerator[AgentEvent, None]:
        """主循环：管理轮次，协调模型调用和工具执行。"""

        # 把用户传入的消息复制一份，避免修改外部列表。
        history: list[AgentMessage] = list(messages)

        # 通知外部：Agent 开始运行了。
        yield AgentStartEvent()

        for _ in range(self.max_turns):
            # 通知外部：新一轮开始。
            yield TurnStartEvent()

            # ── 1. 调用模型，翻译事件 ──────────────────────────────────────
            message: AssistantMessage | None = None

            async with aclosing(
                self._assistant_events(model=model, system=system, history=history, tools=tools)
            ) as stream:
                async for event in stream:
                    yield event
                    if isinstance(event, MessageEndEvent) and isinstance(
                        event.message, AssistantMessage
                    ):
                        message = event.message

            assert isinstance(message, AssistantMessage)

            # 把助手的回复加入历史，后续轮次能看到它。
            history.append(message)

            # ── 2. 如果没有工具调用，结束循环 ──────────────────────────────
            if not message.tool_calls:
                yield TurnEndEvent(message=message)
                break

            # ── 3. 执行工具 ────────────────────────────────────────────────
            tool_results: list[ToolResultMessage] = []

            for tool_call in message.tool_calls:
                tool_result_msg: ToolResultMessage | None = None

                async with aclosing(self._execute_tool_call(tool_call, tools)) as stream:
                    async for event in stream:
                        yield event
                        if isinstance(event, MessageEndEvent) and isinstance(
                            event.message, ToolResultMessage
                        ):
                            tool_result_msg = event.message

                assert isinstance(tool_result_msg, ToolResultMessage)
                tool_results.append(tool_result_msg)
                history.append(tool_result_msg)

            # 通知外部：这一轮结束，并附带本轮的消息和工具结果。
            yield TurnEndEvent(message=message, tool_results=tool_results)

        # 通知外部：Agent 整体运行结束，并附上全部消息。
        yield AgentEndEvent(messages=history)

    async def _assistant_events(
        self,
        *,
        model: str,
        system: str,
        history: list[AgentMessage],
        tools: list[AgentTool],
    ) -> AsyncGenerator[AgentEvent, None]:
        """翻译层：把 Provider 的"模型语言"翻译成 Agent 的"消息生命周期语言"。

        Provider 事件（response_start, delta, response_end 等）
        → Agent 事件（MessageStartEvent, MessageDeltaEvent, MessageEndEvent）

        流式：先发一个空消息的 MessageStartEvent 表示"开始输出"，随后每个
        Provider delta 转发成一个 MessageDeltaEvent，最后用 MessageEndEvent
        携带完整消息收尾。完整消息仍是唯一"事实来源"，落账本与最终答案都靠它。
        """

        # 调用 Provider 的流式接口，拿到一组事件。
        stream = self.provider.stream_response(
            model=model,
            system=system,
            messages=history,
            tools=tools,
        )

        # 开始输出：先发空消息，让下游可以准备流式缓冲。
        yield MessageStartEvent(message=AssistantMessage())

        # 遍历 Provider 事件流：增量实时转发，完整消息与错误留到最后。
        final_message: AssistantMessage | None = None

        # 提前关闭上层事件流时，也立即关闭底层 HTTP 流。
        async with aclosing(stream):
            async for event in stream:
                if event.type == "delta":
                    # 正文 / 思考的增量片段，原样转成 Agent 层增量事件。
                    yield MessageDeltaEvent(kind=event.kind, delta=event.delta)
                elif event.type == "response_end":
                    # 模型响应结束，携带完整的助手消息。
                    final_message = event.message
                elif event.type == "error":
                    # 模型报错时，构造一个包含错误信息的助手消息。
                    final_message = AssistantMessage(
                        content=[
                            TextContent(text=f"错误: {event.message or 'Provider 未提供错误详情'}")
                        ]
                    )

        if final_message is None:
            raise RuntimeError("模型响应没有完整的结束消息")

        # 产出完整消息，收尾。
        yield MessageEndEvent(message=final_message)

    async def _execute_tool_call(
        self,
        tool_call: ToolCall,
        tools: list[AgentTool],
    ) -> AsyncGenerator[AgentEvent, None]:
        """执行单个工具调用，产出工具相关事件。"""

        # 通知外部：开始执行某个工具。
        yield ToolExecutionStartEvent(
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
            args=dict(tool_call.arguments),
        )

        # 在工具列表中查找对应的工具。
        tool = self._find_tool(tools, tool_call.name)

        if tool is None:
            # 找不到工具时，返回一个错误结果，让模型知道情况。
            result = AgentToolResult(
                content=[TextContent(text=f"未知工具: {tool_call.name}")], is_error=True
            )
        else:
            # 调用工具的真正执行函数。
            try:
                result = await tool.execute(tool_call.id, tool_call.arguments)
            except Exception as error:
                # CancelledError 不属于 Exception，取消仍然向上传播。
                result = AgentToolResult(
                    content=[TextContent(text=f"工具执行失败: {type(error).__name__}: {error}")],
                    is_error=True,
                )

        # 工具结果先成为完整消息，Harness 保存后才向界面报告执行结束。
        yield MessageEndEvent(
            message=ToolResultMessage(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                content=result.content,
                is_error=result.is_error,
                details=result.details,
            )
        )

        # 通知外部：工具执行完毕。
        yield ToolExecutionEndEvent(
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
            result=result,
            is_error=result.is_error,
        )

    @staticmethod
    def _find_tool(tools: list[AgentTool], name: str) -> AgentTool | None:
        """在工具列表中按名字查找工具。"""

        for tool in tools:
            if tool.name == name:
                return tool
        return None


__all__ = ["AgentLoop"]
