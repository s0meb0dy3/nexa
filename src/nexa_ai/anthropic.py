"""Anthropic Messages 协议：消息转换、流式内容块和工具调用。"""

import json
from collections.abc import AsyncGenerator
from copy import copy
from typing import Self

import httpx

from nexa_agent.messages import (
    AgentMessage,
    AssistantMessage,
    TextContent,
    ThinkingContent,
    TokenUsage,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from nexa_agent.provider import ModelProvider, ThinkingLevel
from nexa_agent.provider_events import (
    ProviderDeltaEvent,
    ProviderErrorEvent,
    ProviderEvent,
    ProviderResponseEndEvent,
    ProviderResponseStartEvent,
)
from nexa_agent.tools import AgentTool


class AnthropicProvider:
    """独立实现 Messages 协议；model_settings 复用供应商已有的模型规则。"""

    def __init__(
        self,
        *,
        name: str,
        api_key: str,
        base_url: str,
        model_settings: ModelProvider | None = None,
        max_tokens: int = 8192,
    ) -> None:
        self.name = name
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model_settings = model_settings
        self.max_tokens = max_tokens
        self.default_thinking: ThinkingLevel = (
            model_settings.default_thinking if model_settings else "default"
        )
        self.thinking = self.default_thinking

    def context_window(self, model: str) -> int | None:
        return self.model_settings.context_window(model) if self.model_settings else None

    def thinking_options(self, model: str) -> tuple[ThinkingLevel, ...]:
        return self.model_settings.thinking_options(model) if self.model_settings else ("default",)

    def with_thinking(self, model: str, level: ThinkingLevel) -> Self:
        """只改变候选副本；Session 保存设置成功后再启用它。"""
        if level not in self.thinking_options(model):
            raise ValueError(f"模型 {model} 不支持思考设置 {level}")
        candidate = copy(self)
        candidate.thinking = level
        return candidate

    def _thinking_parameters(self) -> dict:
        """DeepSeek/MiniMax 的等级用 output_config.effort，而非 reasoning_effort。"""
        if self.thinking == "default":
            return {}
        if self.thinking == "off":
            return {"thinking": {"type": "disabled"}}
        return {"thinking": {"type": "adaptive"}, "output_config": {"effort": self.thinking}}

    @staticmethod
    def _convert_messages(messages: list[AgentMessage]) -> list[dict]:
        """工具结果属于 user 内容块；连续结果合并，保留思考与调用的原始顺序。"""
        converted: list[dict] = []
        for message in messages:
            content: list[dict] = []
            if isinstance(message, UserMessage):
                role = "user"
                content.append({"type": "text", "text": message.text})
            elif isinstance(message, ToolResultMessage):
                role = "user"
                content.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": message.tool_call_id,
                        "content": message.text,
                        "is_error": message.is_error,
                    }
                )
            else:
                role = "assistant"
                for block in message.content:
                    if isinstance(block, TextContent):
                        content.append({"type": "text", "text": block.text})
                    elif isinstance(block, ThinkingContent):
                        item = {"type": "thinking", "thinking": block.text}
                        if block.signature is not None:
                            item["signature"] = block.signature
                        content.append(item)
                    else:
                        content.append(
                            {
                                "type": "tool_use",
                                "id": block.id,
                                "name": block.name,
                                "input": block.arguments,
                            }
                        )
            if converted and converted[-1]["role"] == role:
                converted[-1]["content"].extend(content)
            else:
                converted.append({"role": role, "content": content})
        return converted

    async def stream_response(
        self,
        *,
        model: str,
        system: str,
        messages: list[AgentMessage],
        tools: list[AgentTool],
    ) -> AsyncGenerator[ProviderEvent, None]:
        """按 index 累积内容块；只有 message_stop 才产出完整助手消息。"""
        payload = {
            "model": model,
            "max_tokens": self.max_tokens,
            "stream": True,
            "messages": self._convert_messages(messages),
            **self._thinking_parameters(),
        }
        if system:
            payload["system"] = system
        if tools:
            payload["tools"] = [
                {"name": t.name, "description": t.description, "input_schema": t.parameters}
                for t in tools
            ]
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Accept": "text/event-stream",
        }
        # base_url 是 SDK 风格根地址，路径固定追加 /v1/messages。
        url = self.base_url + "/v1/messages"
        blocks: dict[int, dict] = {}
        arguments: dict[int, str] = {}
        usage: dict = {}
        finish_reason = None
        try:
            async with (
                httpx.AsyncClient(timeout=60) as client,
                client.stream("POST", url, headers=headers, json=payload) as response,
            ):
                if response.status_code != 200:
                    body = (await response.aread()).decode()
                    yield ProviderErrorEvent(message=f"HTTP {response.status_code}: {body}")
                    return
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    event = json.loads(line[5:].strip())
                    kind = event["type"]
                    if kind == "message_start":
                        usage.update(event["message"]["usage"])
                        yield ProviderResponseStartEvent(model=model)
                    elif kind == "content_block_start":
                        index = event["index"]
                        block = event["content_block"]
                        if block["type"] not in ("text", "thinking", "tool_use"):
                            raise ValueError(f"暂不支持 Anthropic 内容块 {block['type']}")
                        blocks[index] = block
                    elif kind == "content_block_delta":
                        index = event["index"]
                        delta = event["delta"]
                        block = blocks[index]
                        if delta["type"] == "text_delta":
                            block["text"] += delta["text"]
                            yield ProviderDeltaEvent(kind="text", delta=delta["text"])
                        elif delta["type"] == "thinking_delta":
                            block["thinking"] += delta["thinking"]
                            yield ProviderDeltaEvent(kind="reasoning", delta=delta["thinking"])
                        elif delta["type"] == "signature_delta":
                            block["signature"] = block.get("signature", "") + delta["signature"]
                        elif delta["type"] == "input_json_delta":
                            arguments[index] = arguments.get(index, "") + delta["partial_json"]
                    elif kind == "content_block_stop":
                        index = event["index"]
                        if index in arguments:
                            blocks[index]["input"] = json.loads(arguments[index])
                    elif kind == "message_delta":
                        usage.update(event.get("usage", {}))
                        finish_reason = event["delta"].get("stop_reason")
                    elif kind == "error":
                        yield ProviderErrorEvent(message=event["error"]["message"])
                        return
                    elif kind == "message_stop":
                        content: list[TextContent | ThinkingContent | ToolCall] = []
                        for index in sorted(blocks):
                            block = blocks[index]
                            if block["type"] == "text":
                                content.append(TextContent(text=block["text"]))
                            elif block["type"] == "thinking":
                                content.append(
                                    ThinkingContent(
                                        text=block["thinking"], signature=block.get("signature")
                                    )
                                )
                            else:
                                content.append(
                                    ToolCall(
                                        id=block["id"],
                                        name=block["name"],
                                        arguments=block["input"],
                                    )
                                )
                        # Anthropic 将缓存输入单列，必须加回才能表示完整输入上下文。
                        input_tokens = sum(
                            usage.get(k, 0)
                            for k in (
                                "input_tokens",
                                "cache_creation_input_tokens",
                                "cache_read_input_tokens",
                            )
                        )
                        output_tokens = usage["output_tokens"]
                        tokens = TokenUsage(
                            model=model,
                            input_tokens=input_tokens,
                            output_tokens=output_tokens,
                            total_tokens=input_tokens + output_tokens,
                        )
                        yield ProviderResponseEndEvent(
                            message=AssistantMessage(content=content, usage=tokens),
                            finish_reason=finish_reason,
                        )
                        return
                raise ValueError("Anthropic 流未收到 message_stop，响应不完整")
        except (httpx.HTTPError, ValueError) as error:
            yield ProviderErrorEvent(message=str(error) or type(error).__name__)
