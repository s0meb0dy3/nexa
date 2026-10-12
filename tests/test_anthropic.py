"""Messages 协议的流式解析、签名回传、缓存统计和错误边界。"""

import json
from contextlib import aclosing

import httpx
import pytest

from nexa_agent.messages import UserMessage
from nexa_agent.session.storage import JsonlStorage
from nexa_ai.anthropic import AnthropicProvider
from nexa_coding.config import ProviderProfile
from nexa_coding.providers import build_provider
from nexa_coding.session import CodingSession, CodingSessionConfig


def sse(events):
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)


def stream_events(tool_calls=False):
    events = [
        {
            "type": "message_start",
            "message": {
                "usage": {
                    "input_tokens": 20,
                    "cache_creation_input_tokens": 30,
                    "cache_read_input_tokens": 40,
                    "output_tokens": 1,
                }
            },
        },
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "thinking", "thinking": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "读取两个文件"},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "signature_delta", "signature": "opaque-"},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "signature_delta", "signature": "signature"},
        },
        {"type": "content_block_stop", "index": 0},
    ]
    if tool_calls:
        for index, path in enumerate(("one.txt", "two.txt"), 1):
            events.extend(
                [
                    {
                        "type": "content_block_start",
                        "index": index,
                        "content_block": {
                            "type": "tool_use",
                            "id": f"call{index}",
                            "name": "read",
                            "input": {},
                        },
                    },
                    {
                        "type": "content_block_delta",
                        "index": index,
                        "delta": {"type": "input_json_delta", "partial_json": '{"path":'},
                    },
                    {
                        "type": "content_block_delta",
                        "index": index,
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": json.dumps(path) + "}",
                        },
                    },
                    {"type": "content_block_stop", "index": index},
                ]
            )
    else:
        events.extend(
            [
                {
                    "type": "content_block_start",
                    "index": 1,
                    "content_block": {"type": "text", "text": ""},
                },
                {
                    "type": "content_block_delta",
                    "index": 1,
                    "delta": {"type": "text_delta", "text": "完成"},
                },
                {"type": "content_block_stop", "index": 1},
            ]
        )
    events.extend(
        [
            {
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use" if tool_calls else "end_turn"},
                "usage": {"output_tokens": 10},
            },
            {"type": "message_stop"},
        ]
    )
    return events


def mock_client(monkeypatch, handler):
    client = httpx.AsyncClient
    monkeypatch.setattr(
        "nexa_ai.anthropic.httpx.AsyncClient",
        lambda **kwargs: client(transport=httpx.MockTransport(handler), **kwargs),
    )


@pytest.mark.parametrize(
    "host,model,default,level",
    [
        ("api.deepseek.com", "deepseek-flash", "high", "low"),
        ("api.minimax.cn", "MiniMax-M3.1-Flash-Preview", "max", "xhigh"),
    ],
)
def test_vendor_settings(host, model, default, level):
    provider = build_provider(
        ProviderProfile("custom", f"https://{host}/anthropic", model, "unused", "anthropic")
    )
    assert isinstance(provider, AnthropicProvider)
    assert provider.default_thinking == default
    assert provider.with_thinking(model, level)._thinking_parameters()["output_config"] == {
        "effort": level
    }
    assert provider.thinking == default
    if host == "api.deepseek.com":
        assert provider.with_thinking(model, "off")._thinking_parameters() == {
            "thinking": {"type": "disabled"}
        }
    else:
        with pytest.raises(ValueError):
            provider.with_thinking(model, "off")


async def test_tool_loop_signature_usage_and_restore(tmp_path, monkeypatch):
    requests = []
    for path in ("one.txt", "two.txt"):
        (tmp_path / path).write_text(path)

    def handler(request):
        assert str(request.url) == "https://api.deepseek.com/anthropic/v1/messages"
        assert request.headers["x-api-key"] == "unused"
        assert request.headers["anthropic-version"] == "2023-06-01"
        requests.append(json.loads(request.content))
        return httpx.Response(200, text=sse(stream_events(tool_calls=len(requests) == 1)))

    mock_client(monkeypatch, handler)
    provider = build_provider(
        ProviderProfile(
            "deepseek",
            "https://api.deepseek.com/anthropic",
            "deepseek-flash",
            "unused",
            "anthropic",
        )
    )
    config = CodingSessionConfig(
        provider=provider,
        model="deepseek-flash",
        cwd=tmp_path,
        storage=JsonlStorage(tmp_path / "session.jsonl"),
        system="test",
    )
    session = CodingSession.load(config)
    session.set_thinking("low")
    events = [e async for e in session.prompt("读取两个文件")]
    assert any(e.type == "message_delta" and e.kind == "reasoning" for e in events)
    assert len(requests) == 2
    assert requests[0]["system"] == "test"
    assert "input_schema" in requests[0]["tools"][0]
    assert requests[0]["output_config"] == {"effort": "low"}
    assistant = requests[1]["messages"][1]["content"]
    assert assistant[0] == {
        "type": "thinking",
        "thinking": "读取两个文件",
        "signature": "opaque-signature",
    }
    assert [b["id"] for b in assistant[1:]] == ["call1", "call2"]
    results = requests[1]["messages"][2]["content"]
    assert [b["tool_use_id"] for b in results] == ["call1", "call2"]
    assert all(not b["is_error"] for b in results)
    restored = CodingSession.load(config)
    assert restored.thinking == "low"
    assert restored.messages[-1].usage.input_tokens == 90
    assert restored.messages[-1].usage.total_tokens == 100
    async for _ in restored.prompt("继续"):
        pass
    assert requests[-1]["messages"][1]["content"][0]["signature"] == "opaque-signature"


@pytest.mark.parametrize(
    "status,body",
    [
        (401, "bad key"),
        (200, sse([{"type": "error", "error": {"message": "overloaded"}}])),
        (200, sse(stream_events()[:-1])),
    ],
)
async def test_failed_or_incomplete_stream_has_no_final_message(monkeypatch, status, body):
    mock_client(monkeypatch, lambda request: httpx.Response(status, text=body))
    provider = AnthropicProvider(name="test", api_key="unused", base_url="https://example.test")
    events = [
        e
        async for e in provider.stream_response(
            model="test", system="", messages=[UserMessage(content="hi")], tools=[]
        )
    ]
    assert events[-1].type == "error"
    assert not any(e.type == "response_end" for e in events)


async def test_closing_generator_closes_http_stream(monkeypatch):
    class Stream(httpx.AsyncByteStream):
        closed = False

        async def __aiter__(self):
            yield sse(stream_events()).encode()

        async def aclose(self):
            self.closed = True

    stream = Stream()
    mock_client(monkeypatch, lambda request: httpx.Response(200, stream=stream))
    provider = AnthropicProvider(name="test", api_key="unused", base_url="https://example.test")
    async with aclosing(
        provider.stream_response(
            model="test", system="", messages=[UserMessage(content="hi")], tools=[]
        )
    ) as events:
        await anext(events)
    assert stream.closed
