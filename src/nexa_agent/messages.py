"""为最小 Agent 定义与具体模型供应商无关的消息格式。"""

from __future__ import annotations

# Literal 表示“只能是指定的字面量”，例如 Literal["user"] 只能是字符串 "user"。
from typing import Annotated, Literal

# BaseModel 是 Pydantic 的数据模型基类；其他几个工具用来配置和校验模型。
from pydantic import BaseModel, ConfigDict, Field, model_validator

# JSONValue 表示一个可以放进 JSON 的值，例如字符串、数字、列表或字典。
from nexa_agent.types import JSONValue


def _to_camel(name: str) -> str:
    """把 Python 常见的下划线命名转换成 JSON 常见的驼峰命名。

    例如：tool_call_id -> toolCallId。
    """

    # 先按下划线拆开："tool_call_id" -> ["tool", "call", "id"]。
    parts = name.split("_")

    # 第一部分保持原样，后面的部分首字母大写，再拼接起来。
    return parts[0] + "".join(part.title() for part in parts[1:])


class WireModel(BaseModel):
    """所有消息模型共同继承的基类。

    它统一规定：Python 代码使用 snake_case，传输给外部的 JSON 使用 camelCase。
    """

    model_config = ConfigDict(
        # 不允许输入代码中没有定义的额外字段，避免拼写错误悄悄混过去。
        extra="forbid",
        # 读取数据时，同时接受 Python 字段名和 JSON 别名。
        validate_by_name=True,
        validate_by_alias=True,
        # 输出数据时，使用 JSON 别名（camelCase）。
        serialize_by_alias=True,
        # 自动把 tool_call_id 转成 toolCallId。
        alias_generator=_to_camel,
    )


class TextContent(WireModel):
    """一段普通的文字内容。"""

    # type 固定为 "text"，用来告诉程序：这是一个文本块。
    type: Literal["text"] = "text"
    # 真正要显示的文字。
    text: str


class ToolCall(WireModel):
    """助手请求执行某个工具的消息。"""

    # type 固定为 "toolCall"，用来告诉程序：这是一个工具调用块。
    type: Literal["toolCall"] = "toolCall"
    # 这次工具调用的唯一编号，之后工具结果会用它来对应这次调用。
    id: str
    # 要调用的工具名称。
    name: str
    # 传给工具的参数，例如 {"path": "README.md"}。
    # 如果没有参数，就使用一个新的空字典。
    arguments: dict[str, JSONValue] = Field(default_factory=dict)


class ThinkingContent(WireModel):
    """模型的思考（推理）内容块。

    和 TextContent、ToolCall 平级，一起按顺序存在 AssistantMessage.content 里，
    因此会自动跟随消息走完整条链路：事件流 → 会话账本 → 回放 → 展示。

    不同的 Provider 会把它编码成不同字段（DeepSeek 的 reasoning_content、
    Ollama 的 reasoning 等），但核心层只把它当"一段文字"看待。

    Attributes:
        text: 思考正文。
        signature: 供应商拥有的不透明状态（例如 Anthropic 的签名），原样存取、
            不解析。DeepSeek 这类不带签名的 Provider 保持 None。
    """

    type: Literal["thinking"] = "thinking"
    # 思考的正文文字。
    text: str
    # 不透明的供应商状态：只负责原样保存和还原，核心层不解释它的含义。
    signature: str | None = None


class UserMessage(WireModel):
    """用户发送给 Agent 的消息。"""

    # role 固定为 "user"，表示消息来自用户。
    role: Literal["user"] = "user"
    # 内容可以直接是字符串，也可以是多个文本块组成的列表。
    content: str | list[TextContent]

    @property
    def text(self) -> str:
        """获取消息中的纯文字，调用时写 message.text，不需要括号。"""

        # 如果 content 本身就是字符串，直接返回它。
        if isinstance(self.content, str):
            return self.content

        # 如果 content 是文本块列表，就把每个文本块拼成一个字符串。
        return "".join(block.text for block in self.content)


class AssistantMessage(WireModel):
    """Agent 返回的消息，可以按顺序包含文字和工具调用。"""

    # role 固定为 "assistant"，表示消息来自助手。
    role: Literal["assistant"] = "assistant"
    # 内容按顺序保存；每一项要么是文字，要么是思考，要么是工具调用。
    # 没有内容时，使用一个新的空列表。
    content: list[TextContent | ThinkingContent | ToolCall] = Field(default_factory=list)

    # 在 Pydantic 正式检查字段之前，先把输入格式整理一下。
    @model_validator(mode="before")
    @classmethod
    def _normalize_string_content(cls, value: object) -> object:
        """把方便输入的字符串转换成统一的文本块列表。"""

        # 例如允许用户写 content="你好"，而不必手动写 TextContent。
        # 这样写起来更方便，但模型内部仍统一保存为文本块列表。
        if isinstance(value, dict):
            # 复制一份输入字典，避免直接修改调用者传进来的原字典。
            data = dict(value)
            content = data.get("content")
            if isinstance(content, str):
                # 非空字符串变成一个文本块；空字符串变成空列表。
                data["content"] = [TextContent(text=content)] if content else []
            return data

        # 如果输入不是字典，交给 Pydantic 后续处理和报错。
        return value

    @property
    def text(self) -> str:
        """只取出助手消息里的文字，忽略思考和工具调用。"""

        # 只保留 TextContent，再把它们的文字拼接起来。
        return "".join(block.text for block in self.content if isinstance(block, TextContent))

    @property
    def thinking(self) -> str:
        """只取出助手消息里的思考文字，忽略正文和工具调用。"""

        # 只保留 ThinkingContent，再把它们的思考文字拼接起来。
        return "".join(block.text for block in self.content if isinstance(block, ThinkingContent))

    @property
    def tool_calls(self) -> tuple[ToolCall, ...]:
        """获取助手消息中所有的工具调用。"""

        # 只保留 ToolCall，并转成 tuple，表示这是一组固定结果。
        return tuple(block for block in self.content if isinstance(block, ToolCall))


class ToolResultMessage(WireModel):
    """工具执行完后返回给 Agent 的结果消息。"""

    # role 固定为 "toolResult"，表示这是工具执行结果。
    role: Literal["toolResult"] = "toolResult"
    # 对应 ToolCall.id，用来知道这个结果属于哪次工具调用。
    tool_call_id: str
    # 实际执行的工具名称。
    tool_name: str
    # 工具返回的文字内容。
    content: list[TextContent] = Field(default_factory=list)
    # 工具是否执行失败；默认 False，表示没有错误。
    is_error: bool = False
    # 额外的结构化信息，可以是任意合法 JSON 值；默认没有额外信息。
    details: JSONValue = None

    # 和 AssistantMessage 一样，允许 content 直接传字符串。
    @model_validator(mode="before")
    @classmethod
    def _normalize_string_content(cls, value: object) -> object:
        if isinstance(value, dict):
            # 复制输入字典，避免修改外部传入的数据。
            data = dict(value)
            content = data.get("content")
            if isinstance(content, str):
                # 把字符串统一转换成文本块列表。
                data["content"] = [TextContent(text=content)] if content else []
            return data
        return value

    @property
    def text(self) -> str:
        """获取工具结果中的纯文字。"""

        return "".join(block.text for block in self.content)


# AgentMessage 表示 Agent 可能收到或产生的三种核心消息之一。
# discriminator="role" 告诉 Pydantic：根据 role 的值判断具体是哪一种消息。
# 例如 role="user" 就按 UserMessage 解析，role="assistant" 就按 AssistantMessage 解析。
type AgentMessage = Annotated[
    UserMessage | AssistantMessage | ToolResultMessage,
    Field(discriminator="role"),
]

# 这是 AgentMessage 的简短名字，两个名字指向完全相同的类型。
Message = AgentMessage


def message_text(message: AgentMessage) -> str:
    """统一获取任意核心消息中的可见文字。"""

    return message.text


# 使用 from messages import * 时，只导出下面这些名称。
__all__ = [
    "AgentMessage",
    "AssistantMessage",
    "Message",
    "TextContent",
    "ThinkingContent",
    "ToolCall",
    "ToolResultMessage",
    "UserMessage",
    "WireModel",
    "message_text",
]
