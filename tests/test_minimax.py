"""MiniMax 的参数、工具思考回传和会话恢复。"""

import json

import httpx
import pytest

from nexa_agent.session.storage import JsonlStorage
from nexa_ai.minimax import MODEL, MiniMaxProvider
from nexa_coding.config import ProviderProfile
from nexa_coding.providers import build_provider
from nexa_coding.session import CodingSession, CodingSessionConfig


@pytest.mark.parametrize("host", ["api.minimax.cn", "api.minimax.io"])
def test_minimax_detection_and_settings(host):
    provider = build_provider(ProviderProfile("custom", f"https://{host}/v1", MODEL, "unused"))
    assert isinstance(provider, MiniMaxProvider)
    assert provider.context_window(MODEL) == 1_000_000
    assert provider.default_thinking == "max"
    for level in provider.thinking_options(MODEL):
        candidate = provider.with_thinking(MODEL, level)
        assert candidate._thinking_parameters(MODEL)["reasoning_effort"] == level
        assert provider.thinking == "max"
    with pytest.raises(ValueError, match="不支持"):
        provider.with_thinking(MODEL, "off")


async def test_minimax_tool_loop_and_restore(tmp_path, monkeypatch):
    requests = []
    (tmp_path / "example.txt").write_text("hello")

    def response(request):
        payload = json.loads(request.content)
        requests.append(payload)
        delta = {"reasoning_content": "先读取文件"}
        if len(requests) == 1:
            delta["tool_calls"] = [
                {
                    "index": 0,
                    "id": "call1",
                    "type": "function",
                    "function": {"name": "read", "arguments": '{"path":"example.txt"}'},
                }
            ]
        else:
            delta["content"] = "完成"
        chunks = [
            {"choices": [{"delta": delta, "finish_reason": "stop"}]},
            {
                "choices": [],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 20,
                    "total_tokens": 120,
                },
            },
        ]
        return httpx.Response(
            200, text="".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
        )

    client = httpx.AsyncClient
    monkeypatch.setattr(
        "nexa_ai.openai_compatible.httpx.AsyncClient",
        lambda **kwargs: client(transport=httpx.MockTransport(response), **kwargs),
    )
    config = CodingSessionConfig(
        provider=MiniMaxProvider(name="minimax", api_key="unused"),
        model=MODEL,
        cwd=tmp_path,
        storage=JsonlStorage(tmp_path / "session.jsonl"),
        system="test",
    )
    session = CodingSession.load(config)
    assert session.thinking == "max"
    session.set_thinking("xhigh")
    async for _ in session.prompt("读取文件"):
        pass
    assert len(requests) == 2
    assert all(p["reasoning_effort"] == "xhigh" and p["reasoning_split"] for p in requests)
    history = requests[1]["messages"]
    assistant = next(m for m in history if m["role"] == "assistant")
    assert assistant["reasoning_content"] == "先读取文件"
    assert assistant["tool_calls"][0]["id"] == "call1"
    assert "hello" in next(m for m in history if m["role"] == "tool")["content"]
    restored = CodingSession.load(config)
    assert restored.thinking == "xhigh"
    assert restored.messages[-1].usage.total_tokens == 120
    async for _ in restored.prompt("继续"):
        pass
    assert all(m["reasoning_content"] for m in requests[-1]["messages"] if m["role"] == "assistant")
