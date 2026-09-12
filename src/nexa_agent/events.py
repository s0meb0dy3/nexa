"""定义 Agent 运行过程中发出的各种事件（进度通知）。"""

from __future__ import annotations

# Annotated 可以给类型附加额外说明，Literal 表示只能取指定的固定值。
from typing import Annotated, Literal

# Field 用来设置字段的默认值，以及告诉 Pydantic 如何区分不同事件。
from pydantic import Field

# AgentMessage 表示用户、助手或工具结果消息。
from nexa_agent.messages import AgentMessage, ToolResultMessage, WireModel

# AgentToolResult 表示工具执行后产生的结果。
from nexa_agent.tools import AgentToolResult

# JSONValue 表示任意合法的 JSON 值，例如字符串、数字、列表或字典。
from nexa_agent.types import DeltaKind, JSONValue

# 可以把事件理解成 Agent 发出的“我现在进行到哪一步了”的通知。
# 大致流程是：Agent 开始 -> 一轮对话开始 -> 消息/工具执行 -> 一轮结束 -> Agent 结束。


class AgentStartEvent(WireModel):
    """Agent 整个运行过程开始时发出的事件。"""

    # type 固定为 "agent_start"，用来识别事件类型。
    type: Literal["agent_start"] = "agent_start"


class AgentEndEvent(WireModel):
    """Agent 整个运行过程结束时发出的事件。"""

    # type 固定为 "agent_end"，用来识别事件类型。
    type: Literal["agent_end"] = "agent_end"
    # 保存这次运行过程中产生的全部核心消息；没有消息时是空列表。
    messages: list[AgentMessage] = Field(default_factory=list)


class TurnStartEvent(WireModel):
    """Agent 开始新一轮处理时发出的事件。"""

    # 一轮（turn）通常表示 Agent 处理一次输入并请求一次模型结果。
    type: Literal["turn_start"] = "turn_start"


class TurnEndEvent(WireModel):
    """Agent 完成一轮处理时发出的事件。"""

    # type 固定为 "turn_end"，用来识别事件类型。
    type: Literal["turn_end"] = "turn_end"
    # 这一轮产生的消息。
    message: AgentMessage
    # 这一轮中执行过的工具结果；没有工具调用时是空列表。
    tool_results: list[ToolResultMessage] = Field(default_factory=list)


class MessageStartEvent(WireModel):
    """开始处理一条消息时发出的事件。

    流式场景下，它在模型开始吐字时就发出（携带一条空消息表示"开始了"），
    随后由 MessageDeltaEvent 逐个送来片段；最终完整内容在 MessageEndEvent 里。
    """

    # type 固定为 "message_start"，用来识别事件类型。
    type: Literal["message_start"] = "message_start"
    # 正在开始处理的具体消息；流式时通常是空消息。
    message: AgentMessage


class MessageDeltaEvent(WireModel):
    """助手消息流式输出的一小段：正文或思考。"""

    # type 固定为 "message_delta"，用来识别事件类型。
    type: Literal["message_delta"] = "message_delta"
    # 片段类别：正文（text）或思考（reasoning）。
    kind: DeltaKind = "text"
    # 这一小段的文字。
    delta: str


class MessageEndEvent(WireModel):
    """一条消息处理完成时发出的事件。"""

    # type 固定为 "message_end"，用来识别事件类型。
    type: Literal["message_end"] = "message_end"
    # 已经处理完成的具体消息。
    message: AgentMessage


class ToolExecutionStartEvent(WireModel):
    """开始执行工具时发出的事件。"""

    # type 固定为 "tool_execution_start"，用来识别事件类型。
    type: Literal["tool_execution_start"] = "tool_execution_start"
    # 这次工具调用的唯一编号。
    tool_call_id: str
    # 要执行的工具名称。
    tool_name: str
    # 传给工具的参数；没有参数时是空字典。
    args: dict[str, JSONValue] = Field(default_factory=dict)


class ToolExecutionUpdateEvent(WireModel):
    """工具执行过程中产生阶段性结果时发出的事件。"""

    # type 固定为 "tool_execution_update"，用来识别事件类型。
    type: Literal["tool_execution_update"] = "tool_execution_update"
    # 这次工具调用的唯一编号。
    tool_call_id: str
    # 正在执行的工具名称。
    tool_name: str
    # 传给工具的参数。
    args: dict[str, JSONValue] = Field(default_factory=dict)
    # 到目前为止得到的部分结果，不一定是最终结果。
    partial_result: AgentToolResult


class ToolExecutionEndEvent(WireModel):
    """工具执行完成时发出的事件。"""

    # type 固定为 "tool_execution_end"，用来识别事件类型。
    type: Literal["tool_execution_end"] = "tool_execution_end"
    # 这次工具调用的唯一编号。
    tool_call_id: str
    # 执行完成的工具名称。
    tool_name: str
    # 工具最终返回的结果。
    result: AgentToolResult
    # 是否执行出错；True 表示出错，False 表示成功。
    is_error: bool


# AgentEvent 表示所有可能的事件类型。
# 这里的 | 表示"或者"：一个 AgentEvent 可以是下面十种事件中的一种。
# discriminator="type" 告诉 Pydantic 根据 type 字段选择具体的事件类。
# 例如 type="agent_start" 就解析成 AgentStartEvent。
type AgentEvent = Annotated[
    AgentStartEvent
    | AgentEndEvent
    | TurnStartEvent
    | TurnEndEvent
    | MessageStartEvent
    | MessageDeltaEvent
    | MessageEndEvent
    | ToolExecutionStartEvent
    | ToolExecutionUpdateEvent
    | ToolExecutionEndEvent,
    Field(discriminator="type"),
]
