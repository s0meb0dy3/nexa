"""测试思考块（ThinkingContent）：模型、序列化、Provider 解析、链路穿透。"""

from __future__ import annotations

import json

import pytest

from nexa_agent.messages import AssistantMessage, TextContent, ThinkingContent
from nexa_agent.provider_events import ProviderResponseEndEvent, ProviderResponseStartEvent
from nexa_agent.session.entries import MessageEntry
from nexa_agent.session.jsonl import entry_from_line, entry_to_line
from nexa_ai.fake import FakeProvider
from nexa_ai.openai_compatible import OpenAICompatibleProvider

# ── 消息模型 ──────────────────────────────────────────────────────────────────


def test_assistant_text_ignores_thinking():
    """AssistantMessage.text 只取正文，思考被忽略。"""
    message = AssistantMessage(
        content=[ThinkingContent(text="先算一下"), TextContent(text="答案是 2")]
    )

    assert message.text == "答案是 2"
    assert message.thinking == "先算一下"


def test_assistant_thinking_only_has_empty_text():
    """只有思考、没有正文时，text 为空而 thinking 有内容。"""
    message = AssistantMessage(content=[ThinkingContent(text="纯粹在想")])

    assert message.text == ""
    assert message.thinking == "纯粹在想"
    assert message.tool_calls == ()


def test_thinking_content_type_literal():
    """思考块的 type 判别字段是 "thinking"。"""
    assert ThinkingContent(text="x").type == "thinking"


# ── 序列化 / 账本往返 ─────────────────────────────────────────────────────────


def test_thinking_survives_jsonl_round_trip():
    """思考块带着签名经 JSONL 往返后保真。"""
    message = AssistantMessage(
        content=[
            ThinkingContent(text="推理过程", signature="sig-abc"),
            TextContent(text="答案"),
        ]
    )
    entry = MessageEntry(id="m1", parent_id=None, message=message)

    restored = entry_from_line(entry_to_line(entry), 1)

    assert restored is not None
    restored_message = restored.message
    assert isinstance(restored_message, AssistantMessage)
    assert restored_message.thinking == "推理过程"
    assert restored_message.text == "答案"
    # 签名作为不透明状态原样还原。
    signature = next(
        block for block in restored_message.content if isinstance(block, ThinkingContent)
    ).signature
    assert signature == "sig-abc"


def test_thinking_without_signature_omits_it():
    """signature=None 时序列化省略该字段（exclude_none），读回仍是 None。"""
    message = AssistantMessage(content=[ThinkingContent(text="无签名")])
    entry = MessageEntry(id="m1", parent_id=None, message=message)

    line = entry_to_line(entry)
    assert "signature" not in json.loads(line)["message"]["content"][0]

    restored = entry_from_line(line, 1)
    assert restored is not None
    block = restored.message.content[0]
    assert isinstance(block, ThinkingContent)
    assert block.signature is None


# ── Provider 解析 ─────────────────────────────────────────────────────────────


def test_reasoning_delta_reads_deepseek_field():
    """DeepSeek 的 reasoning_content 被识别。"""
    assert OpenAICompatibleProvider._reasoning_delta({"reasoning_content": "思考"}) == "思考"


def test_reasoning_delta_reads_ollama_field():
    """Ollama 等后端的 reasoning 字段被识别。"""
    assert OpenAICompatibleProvider._reasoning_delta({"reasoning": "thinking"}) == "thinking"


@pytest.mark.parametrize(
    "delta",
    [
        {},  # 没有字段
        {"reasoning_content": ""},  # 空字符串
        {"reasoning_content": None},  # None
        {"content": "正文"},  # 只有正文
    ],
)
def test_reasoning_delta_returns_empty(delta):
    """没有思考内容时返回空字符串。"""
    assert OpenAICompatibleProvider._reasoning_delta(delta) == ""


def test_convert_messages_drops_thinking():
    """回传请求时思考块被有意丢弃：只发正文与工具调用。"""
    message = AssistantMessage(
        content=[
            ThinkingContent(text="不该回传的推理", signature="sig"),
            TextContent(text="正文"),
        ]
    )

    api_messages = OpenAICompatibleProvider._convert_messages("", [message])

    assistant_entry = next(m for m in api_messages if m["role"] == "assistant")
    assert assistant_entry["content"] == "正文"
    assert "不该回传的推理" not in json.dumps(api_messages)


# ── 链路穿透：思考块能穿过 loop 不丢 ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_thinking_message_passes_through_loop():
    """FakeProvider 产出的带思考消息，经 AgentLoop 后思考仍在。"""
    from nexa_agent.loop import AgentLoop

    message = AssistantMessage(
        content=[ThinkingContent(text="内部推理"), TextContent(text="最终回答")]
    )
    provider = FakeProvider(
        [
            [
                ProviderResponseStartEvent(model="test-model"),
                ProviderResponseEndEvent(message=message),
            ]
        ]
    )
    loop = AgentLoop(provider)

    events = [
        event async for event in loop.run(model="test-model", system="", messages=[], tools=[])
    ]

    # 最后一个事件是 AgentEndEvent，其 messages 里应保留思考。
    final = events[-1]
    assert final.type == "agent_end"
    assistant = next(m for m in final.messages if m.role == "assistant")
    assert assistant.thinking == "内部推理"
    assert assistant.text == "最终回答"
