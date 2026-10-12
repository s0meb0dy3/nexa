"""MiniMax M3.1 的思考设置、上下文上限和思考历史回传。"""

from copy import copy
from typing import Self

from nexa_agent.messages import AgentMessage, AssistantMessage
from nexa_agent.provider import ThinkingLevel
from nexa_agent.tools import AgentTool
from nexa_ai.openai_compatible import OpenAICompatibleProvider

MODEL = "MiniMax-M3.1-Flash-Preview"


class MiniMaxProvider(OpenAICompatibleProvider):
    """复用通用 HTTP 和流式解析，仅适配当前使用的 M3.1 模型。"""

    default_thinking: ThinkingLevel = "max"
    thinking: ThinkingLevel = "max"

    def context_window(self, model: str) -> int | None:
        """返回官方上下文容量，供 /usage 和底栏显示百分比。"""
        return 1_000_000 if model == MODEL else None

    def thinking_options(self, model: str) -> tuple[ThinkingLevel, ...]:
        """供 Session 校验设置；M3.1 必须思考，不提供 off。"""
        if model != MODEL:
            raise ValueError(f"不支持 MiniMax 模型 {model}；请使用 {MODEL}")
        return ("low", "medium", "high", "xhigh", "max")

    def with_thinking(self, model: str, level: ThinkingLevel) -> Self:
        """先校验，再返回副本，让 Session 保存成功后才切换设置。"""
        super().with_thinking(model, level)
        candidate = copy(self)
        candidate.thinking = level
        return candidate

    def _thinking_parameters(self, model: str) -> dict:
        """父类发请求时合并这些参数，思考与正文使用独立字段。"""
        return {"reasoning_effort": self.thinking, "reasoning_split": True}

    def _request_messages(
        self, system: str, messages: list[AgentMessage], tools: list[AgentTool]
    ) -> list[dict]:
        """工具调用后的下一次请求必须带回助手思考，保持推理链完整。"""
        converted = super()._request_messages(system, messages, tools)
        history = converted[1:] if system else converted
        for message, api_message in zip(messages, history, strict=True):
            if isinstance(message, AssistantMessage):
                api_message["reasoning_content"] = message.thinking
        return converted
