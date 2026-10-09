"""最小 Agent 的工具描述和执行结果。"""

from __future__ import annotations

# 这些类型用来描述“工具执行函数应该长什么样”。
# Awaitable 表示可以等待的结果，Callable 表示一个可以调用的函数，
# Mapping 表示类似字典、可以按键读取的数据。
from collections.abc import Awaitable, Callable, Mapping

# dataclass 可以自动生成初始化方法；这里用它保存一个工具的基本信息。
from dataclasses import dataclass

# Field 用来设置字段默认值，model_validator 用来在 Pydantic 校验前整理输入。
from pydantic import Field, model_validator

# TextContent 是文字块，ToolCall 是工具调用块，WireModel 是统一的消息基类。
# ToolCall 也会在文件末尾重新导出，方便调用者从这里一起导入。
from nexa_agent.messages import TextContent, ToolCall, WireModel

# JSONValue 表示任意合法的 JSON 值，例如字符串、数字、列表或字典。
from nexa_agent.types import JSONValue


class AgentToolResult(WireModel):
    """工具执行后返回给 Agent 的结果。"""

    # 工具返回的文字，统一保存成一个个文本块。
    # 没有文字时，默认使用一个新的空列表。
    content: list[TextContent] = Field(default_factory=list)
    # 除文字以外的额外信息，可以是任意合法的 JSON 值。
    details: JSONValue = None
    # 工具报告的失败状态，由 Loop 传递到事件和历史。
    is_error: bool = False

    # 在 Pydantic 正式检查字段之前，先把输入格式统一整理好。
    @model_validator(mode="before")
    @classmethod
    def _normalize_string_content(cls, value: object) -> object:
        """允许调用者直接传字符串，同时把它保存成文本块列表。"""

        if isinstance(value, dict):
            # 复制输入字典，避免修改调用者原本的数据。
            data = dict(value)
            content = data.get("content")
            if isinstance(content, str):
                # 非空字符串变成一个文本块；空字符串变成空列表。
                data["content"] = [TextContent(text=content)] if content else []
            return data

        # 其他格式交给 Pydantic 后续校验并报告错误。
        return value

    @property
    def text(self) -> str:
        """获取结果里的纯文字，调用时写 result.text。"""

        # 把所有文本块的内容拼接成一个普通字符串。
        return "".join(block.text for block in self.content)


# 规定一个工具执行函数的统一格式：
# 1. 接收工具调用 ID 和参数；
# 2. 这是一个异步函数，所以调用后需要 await；
# 3. await 后得到 AgentToolResult。
ToolExecutor = Callable[
    [str, Mapping[str, JSONValue]],
    Awaitable[AgentToolResult],
]


# dataclass 会自动生成 __init__ 等常用方法，让 AgentTool 可以直接创建。
# frozen=True 表示创建后不能修改字段；slots=True 可以减少对象额外开销。
@dataclass(frozen=True, slots=True)
class AgentTool:
    """一个可被 Agent 调用的工具。"""

    # 工具的名字，模型会根据这个名字决定调用哪个工具。
    name: str
    # 给模型看的工具用途说明，帮助模型选择工具。
    description: str
    # 工具参数的描述，通常是一个 JSON Schema 风格的字典。
    parameters: Mapping[str, JSONValue]
    # 真正执行工作的异步函数。
    execute_fn: ToolExecutor

    @property
    def input_schema(self) -> Mapping[str, JSONValue]:
        """提供给模型的工具参数描述。"""

        # 这里不复制或修改参数，只是用更容易理解的名字返回它。
        return self.parameters

    async def execute(
        self,
        tool_call_id: str,
        arguments: Mapping[str, JSONValue],
    ) -> AgentToolResult:
        """执行工具，并把调用 ID 和参数交给真正的执行函数。"""

        # execute_fn 是创建 AgentTool 时传进来的具体工具函数。
        # await 会等待这个异步函数执行完成，并拿到它返回的结果。
        return await self.execute_fn(tool_call_id, arguments)


# 使用 from nexa_agent.tools import * 时，只导出这些名称。
__all__ = [
    "AgentTool",
    "AgentToolResult",
    "ToolCall",
    "ToolExecutor",
]
