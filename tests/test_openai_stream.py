"""完整模型流才产生最终消息；损坏的流和工具参数明确失败。"""

import json

import httpx
import pytest

from nexa_ai.openai_compatible import OpenAICompatibleProvider


@pytest.mark.parametrize(
    "case",
    ["complete", "no_done", "no_finish", "bad_json", "bad_args", "empty_args", "array_args"],
)
async def test_stream_completion_and_tool_arguments(case, monkeypatch):
    delta = {"content": "回答"}
    if case.endswith("args"):
        arguments = {"bad_args": '{"path":', "empty_args": "", "array_args": "[]"}[case]
        delta = {
            "tool_calls": [
                {"index": 0, "id": "read-1", "function": {"name": "read", "arguments": arguments}}
            ]
        }
    chunk = {
        "choices": [{"delta": delta, "finish_reason": None if case == "no_finish" else "stop"}]
    }
    body = "data: " + ("{broken" if case == "bad_json" else json.dumps(chunk)) + "\n\n"
    if case != "no_done":
        body += "data: [DONE]\n\n"
    client = httpx.AsyncClient
    monkeypatch.setattr(
        "nexa_ai.openai_compatible.httpx.AsyncClient",
        lambda **kwargs: client(
            transport=httpx.MockTransport(lambda r: httpx.Response(200, text=body)), **kwargs
        ),
    )
    provider = OpenAICompatibleProvider(name="test", api_key="unused")
    events = [
        event
        async for event in provider.stream_response(model="test", system="", messages=[], tools=[])
    ]
    if case == "complete":
        assert events[-1].type == "response_end"
        assert events[-1].message.text == "回答"
    else:
        assert events[-1].type == "error"
        assert not any(event.type == "response_end" for event in events)
