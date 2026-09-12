"""模型 Provider 输出的最小、统一事件。

这些事件是"核心契约"，由可移植的 agent 层（nexa_agent）拥有；
具体 Provider 实现（nexa_ai）负责产出它们。放在核心层是为了让
依赖方向单向：nexa_ai → nexa_agent，而不是反过来。
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

from nexa_agent.messages import AssistantMessage, ToolCall, WireModel
from nexa_agent.types import DeltaKind


class ProviderResponseStartEvent(WireModel):
    """模型开始生成一次响应。"""

    type: Literal["response_start"] = "response_start"
    model: str


class ProviderDeltaEvent(WireModel):
    """模型流式输出的一小段：正文或思考。

    kind="text" 表示正文片段，kind="reasoning" 表示思考片段。
    两类共用同一个事件，消费端靠 kind 区分，不需要各自一套事件。
    """

    type: Literal["delta"] = "delta"
    # 片段类别：正文 or 思考。
    kind: DeltaKind = "text"
    delta: str


class ProviderToolCallEvent(WireModel):
    """模型请求调用工具；Provider 不在这里执行工具。"""

    type: Literal["tool_call"] = "tool_call"
    tool_call: ToolCall


class ProviderResponseEndEvent(WireModel):
    """模型响应结束，并携带累积后的助手消息。"""

    type: Literal["response_end"] = "response_end"
    message: AssistantMessage
    finish_reason: str | None = None


class ProviderErrorEvent(WireModel):
    """Provider 或模型 API 返回错误。"""

    type: Literal["error"] = "error"
    message: str


# 根据 type 字段，Pydantic 可以把字典解析成对应的事件类。
type ProviderEvent = Annotated[
    ProviderResponseStartEvent
    | ProviderDeltaEvent
    | ProviderToolCallEvent
    | ProviderResponseEndEvent
    | ProviderErrorEvent,
    Field(discriminator="type"),
]


__all__ = [
    "ProviderDeltaEvent",
    "ProviderErrorEvent",
    "ProviderEvent",
    "ProviderResponseEndEvent",
    "ProviderResponseStartEvent",
    "ProviderToolCallEvent",
]
