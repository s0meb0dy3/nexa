"""OpenAI 兼容 API 的 Provider 实现。"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator

import httpx

from nexa_agent.messages import (
    AgentMessage,
    AssistantMessage,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from nexa_agent.provider_events import (
    ProviderDeltaEvent,
    ProviderErrorEvent,
    ProviderEvent,
    ProviderResponseEndEvent,
    ProviderResponseStartEvent,
)
from nexa_agent.tools import AgentTool


class OpenAICompatibleProvider:
    """通过 OpenAI 兼容 API 与模型对话。

    支持任何兼容 OpenAI API 格式的服务，例如：
    - OpenAI 官方 API
    - DeepSeek
    - Azure OpenAI
    - 本地部署的 OpenAI 兼容模型（如 Ollama、vLLM）
    """

    def __init__(
        self,
        *,
        name: str,
        api_key: str,
        base_url: str = "https://api.deepseek.com",
    ) -> None:
        """初始化 Provider。

        Args:
            name: 提供商别名，用于日志和调试（如 "deepseek"、"openai"、"ollama"）
            api_key: API 密钥
            base_url: API 基础地址
        """
        self.name = name
        self.api_key = api_key
        # base_url 是 API 的基础地址，不带末尾斜杠。
        self.base_url = base_url.rstrip("/")

    def stream_response(
        self,
        *,
        model: str,
        system: str,
        messages: list[AgentMessage],
        tools: list[AgentTool],
    ) -> AsyncIterator[ProviderEvent]:
        """向 OpenAI 兼容 API 发起一次流式请求，返回 ProviderEvent 异步迭代器。"""

        return self._stream(model=model, system=system, messages=messages, tools=tools)

    async def _stream(
        self,
        *,
        model: str,
        system: str,
        messages: list[AgentMessage],
        tools: list[AgentTool],
    ) -> AsyncIterator[ProviderEvent]:
        """内部方法：真正执行 HTTP 流式请求。"""

        # 构造请求头，带上 API Key 和流式标识。
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }

        # 把内部消息格式转换成 OpenAI 兼容 API 需要的格式。
        api_messages = self._convert_messages(system, messages)
        payload: dict = {
            "model": model,
            "messages": api_messages,
            "stream": True,
        }

        # 如果有工具定义，也一起发送。
        if tools:
            payload["tools"] = [self._tool_to_api(t) for t in tools]

        # 使用 httpx 异步客户端发起流式请求。
        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                ) as response:
                    # 如果 HTTP 状态码不是 2xx，读取错误信息并报错。
                    if response.status_code != 200:
                        body = await response.aread()
                        yield ProviderErrorEvent(
                            message=f"HTTP {response.status_code}: {body.decode()}"
                        )
                        return

                    # 通知外部：模型开始生成响应。
                    yield ProviderResponseStartEvent(model=model)

                    # 逐步累积助手消息的内容。
                    content_blocks: list[TextContent | ThinkingContent | ToolCall] = []
                    text_buffer = ""
                    # 推理内容单独累积（DeepSeek 的 reasoning_content / Ollama 的 reasoning）。
                    reasoning_buffer = ""
                    # 用字典暂存工具调用的各个部分，key 是工具在列表中的索引。
                    tool_buffers: dict[int, dict] = {}

                    # 逐行读取 SSE（Server-Sent Events）流。
                    async for line in response.aiter_lines():
                        event = self._parse_sse_line(line)
                        if event is None:
                            continue

                        # [DONE] 表示流式传输结束。
                        if event == "[DONE]":
                            break

                        choices = event.get("choices")
                        if not choices:
                            continue
                        delta = choices[0].get("delta")
                        if not delta:
                            continue

                        # 处理推理内容增量：推理模型（DeepSeek reasoner、qwen3 等）
                        # 会把思考吐在独立字段里，作为 kind="reasoning" 的片段发出。
                        reasoning_delta = self._reasoning_delta(delta)
                        if reasoning_delta:
                            reasoning_buffer += reasoning_delta
                            yield ProviderDeltaEvent(kind="reasoning", delta=reasoning_delta)

                        # 处理文本增量：模型每次吐出一小段文字。
                        text_delta = delta.get("content")
                        if text_delta:
                            text_buffer += text_delta
                            yield ProviderDeltaEvent(kind="text", delta=text_delta)

                        # 处理工具调用增量：模型可能分多个 chunk 传完一个工具调用。
                        tool_calls_delta = delta.get("tool_calls")
                        if tool_calls_delta:
                            for tc_delta in tool_calls_delta:
                                idx = tc_delta.get("index", 0)

                                if idx not in tool_buffers:
                                    # 第一次看到这个工具调用，初始化暂存区。
                                    tool_buffers[idx] = {
                                        "id": tc_delta.get("id", ""),
                                        "name": tc_delta.get("function", {}).get("name", ""),
                                        "arguments": "",
                                    }

                                # 追加参数片段（模型可能分多次传完整个 JSON 参数）。
                                args_chunk = tc_delta.get("function", {}).get("arguments", "")
                                if args_chunk:
                                    tool_buffers[idx]["arguments"] += args_chunk

                    # 流结束后，先放思考块（推理发生在正文之前），再放正文。
                    if reasoning_buffer:
                        content_blocks.append(ThinkingContent(text=reasoning_buffer))

                    # 把累积的文本转成 TextContent。
                    if text_buffer:
                        content_blocks.append(TextContent(text=text_buffer))

                    # 把暂存的工具调用转成 ToolCall 对象。
                    for idx in sorted(tool_buffers):
                        buf = tool_buffers[idx]
                        # 尝试解析 JSON 参数；如果解析失败，使用空字典。
                        try:
                            arguments = json.loads(buf["arguments"]) if buf["arguments"] else {}
                        except json.JSONDecodeError:
                            arguments = {}
                        content_blocks.append(
                            ToolCall(
                                id=buf["id"] or str(uuid.uuid4()),
                                name=buf["name"],
                                arguments=arguments,
                            )
                        )

                    # 构造完整的助手消息，通知外部响应结束。
                    message = AssistantMessage(content=content_blocks)
                    finish_reason = None
                    if choices:
                        finish_reason = choices[0].get("finish_reason")
                    yield ProviderResponseEndEvent(
                        message=message,
                        finish_reason=finish_reason,
                    )

            except httpx.HTTPError as e:
                # 网络层面的错误（超时、连接失败等）。
                # 部分 httpx 异常（例如连接被对端提前关闭）没有错误文本。
                message = str(e) or f"{type(e).__name__}（未提供详情）"
                yield ProviderErrorEvent(message=message)

    @staticmethod
    def _reasoning_delta(delta: dict) -> str:
        """从 SSE 的 delta 里取出这一段思考文字。

        不同后端的字段名不一样：
        - DeepSeek：reasoning_content
        - Ollama 等：reasoning

        都没有（或为空）时返回空字符串。
        """

        for key in ("reasoning_content", "reasoning"):
            value = delta.get(key)
            if isinstance(value, str) and value:
                return value
        return ""

    def _parse_sse_line(self, line: str) -> dict | str | None:
        """解析一行 SSE 数据。

        返回：
        - dict: 正常的 JSON 数据块
        - "[DONE]": 流结束标记
        - None: 空行或无关行，应该跳过
        """

        stripped = line.strip()
        # 跳过空行和注释行。
        if not stripped or not stripped.startswith("data: "):
            return None

        data_str = stripped[6:]  # 去掉 "data: " 前缀。
        if data_str == "[DONE]":
            return "[DONE]"

        try:
            return json.loads(data_str)
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _convert_messages(system: str, messages: list[AgentMessage]) -> list[dict]:
        """把内部消息格式转换成 OpenAI 兼容 API 需要的字典列表。"""

        api_messages: list[dict] = []

        # 系统提示词放在最前面。
        if system:
            api_messages.append({"role": "system", "content": system})

        for msg in messages:
            if isinstance(msg, UserMessage):
                # 用户消息：直接取文字内容。
                api_messages.append({"role": "user", "content": msg.text})

            elif isinstance(msg, AssistantMessage):
                # 助手消息：可能同时包含文字和工具调用。
                # 注意：思考块（ThinkingContent）有意不回传——只取 msg.text（正文）
                # 和 msg.tool_calls。DeepSeek 这类接口不接受把推理内容塞回请求；
                # 将来接 Anthropic 时，带签名的思考需要按其要求回传，届时应新增适配器处理。
                entry: dict = {"role": "assistant"}
                text = msg.text
                tool_calls = msg.tool_calls

                if text:
                    entry["content"] = text
                else:
                    # 某些 API 要求 content 不能为 None，设为空字符串。
                    entry["content"] = ""

                if tool_calls:
                    entry["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments),
                            },
                        }
                        for tc in tool_calls
                    ]
                api_messages.append(entry)

            elif isinstance(msg, ToolResultMessage):
                # 工具结果消息：告诉模型某个工具执行后返回了什么。
                api_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": msg.tool_call_id,
                        "content": msg.text,
                    }
                )

        return api_messages

    @staticmethod
    def _tool_to_api(tool: AgentTool) -> dict:
        """把内部工具定义转换成 OpenAI 兼容 API 需要的格式。"""

        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema,
            },
        }


__all__ = ["OpenAICompatibleProvider"]
