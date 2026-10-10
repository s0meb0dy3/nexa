"""DeepSeek 的思考参数和工具调用历史适配。"""

from copy import copy
from typing import Self

from nexa_agent.messages import AgentMessage, AssistantMessage
from nexa_agent.provider import ThinkingLevel
from nexa_agent.tools import AgentTool
from nexa_ai.openai_compatible import OpenAICompatibleProvider


class DeepSeekProvider(OpenAICompatibleProvider):
    """复用父类的 HTTP 请求和流式解析，补充 DeepSeek 的思考规则。

    Session 调用 thinking_options / with_thinking 来查看和修改设置；
    父类发送请求时调用 _thinking_parameters / _request_messages 来构造请求。
    这些方法负责准备配置和数据，真正的 API 调用仍在父类 _stream 中。
    """

    # 控制模型如何思考；与 TUI 的 Ctrl+T 展开状态无关。
    default_thinking: ThinkingLevel = "high"
    thinking: ThinkingLevel = "high"

    def thinking_options(self, model: str) -> tuple[ThinkingLevel, ...]:
        """根据模型名，返回用户可以选择的思考设置。

        Session 用它检查设置是否可用，TUI 用它显示 /thinking 的可选项。
        当前只适配 Flash 和 Pro，两者使用相同选项；其他模型名直接报错。
        off 表示关闭；low / high / max 表示开启思考并指定强度。
        默认使用 high，不单独提供 default 或 on。
        """
        if model not in ("deepseek-flash", "deepseek-v4-pro"):
            raise ValueError(
                f"不支持 DeepSeek 模型 {model}；请使用 deepseek-flash 或 deepseek-v4-pro"
            )
        return ("off", "low", "high", "max")

    def with_thinking(self, model: str, level: ThinkingLevel) -> Self:
        """返回采用新思考设置的 Provider 副本，不修改当前实例。

        Session 修改设置或切换模型时调用。先由父类校验 level 是否可用，
        再浅拷贝当前 Provider：地址和密钥等配置保留，只改变 thinking。
        这样 Session 可以先保存设置，保存成功后再启用副本；失败时旧实例不变。
        Self 表示返回值仍是当前这个 Provider 类型。
        """
        super().with_thinking(model, level)
        candidate = copy(self)
        candidate.thinking = level
        return candidate

    def _thinking_parameters(self, model: str) -> dict:
        """把当前思考设置转换成 DeepSeek 请求需要的参数字典。

        父类 _stream 每次请求都会调用它，并把返回值合并进请求体。
        off 发送 disabled；其他选项发送 enabled，并添加 reasoning_effort。
        model 参数与父类接口保持一致；两种已适配模型共用这套转换规则。
        """
        parameters: dict = {
            "thinking": {"type": "disabled" if self.thinking == "off" else "enabled"}
        }
        if self.thinking in ("low", "high", "max"):
            parameters["reasoning_effort"] = self.thinking
        return parameters

    def _request_messages(
        self, system: str, messages: list[AgentMessage], tools: list[AgentTool]
    ) -> list[dict]:
        """把系统提示词和历史消息转换成请求格式，并补上思考历史。

        父类 _stream 发请求前调用：system 是系统提示词，messages 是历史，
        tools 是本次提供给模型的工具。父类先转换正文和工具调用；
        如果请求包含工具，再为每条助手消息补上 reasoning_content，
        让 DeepSeek 在工具调用后的请求中收到之前的思考内容。
        返回值是可直接放进请求体 messages 字段的字典列表。
        """
        converted = super()._request_messages(system, messages, tools)
        if tools:
            # 系统消息没有对应的 AgentMessage，其余消息一一对应。
            history = converted[1:] if system else converted
            for message, api_message in zip(messages, history, strict=True):
                if isinstance(message, AssistantMessage):
                    api_message["reasoning_content"] = message.thinking
        return converted
