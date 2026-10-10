"""API 实测统计穿过工具循环、账本和 TUI；不累加请求或伪造缺失值。"""

import json

import httpx
import pytest
from rich.text import Text
from textual.widgets import Static

from nexa_agent.events import MessageEndEvent
from nexa_agent.messages import AssistantMessage, TokenUsage
from nexa_agent.session.storage import JsonlStorage
from nexa_ai.deepseek import DeepSeekProvider
from nexa_ai.fake import FakeProvider
from nexa_coding.paths import NexaPaths
from nexa_coding.session import CodingSession, CodingSessionConfig
from nexa_coding.tui.adapter import TuiEventAdapter
from nexa_coding.tui.app import NexaTuiApp, PromptInput
from nexa_coding.tui.state import TuiState


@pytest.mark.asyncio
@pytest.mark.parametrize("usage_only", [True, False])
async def test_stream_usage_through_tool_loop_and_restore(tmp_path, monkeypatch, usage_only):
    requests = []
    (tmp_path / "file.txt").write_text("hello")

    def response(request):
        payload = json.loads(request.content)
        assert payload["stream_options"] == {"include_usage": True}
        requests.append(payload)
        first = len(requests) == 1
        delta = (
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call1",
                        "function": {
                            "name": "read",
                            "arguments": '{"path":"file.txt"}',
                        },
                    }
                ]
            }
            if first
            else {"content": "完成"}
        )
        usage = {
            "prompt_tokens": 100 if first else 200,
            "completion_tokens": 20,
            "total_tokens": 120 if first else 220,
        }
        chunks = [
            {
                "choices": [{"delta": delta, "finish_reason": "tool_calls" if first else "stop"}],
                "usage": None,
            },
            {
                "choices": [] if usage_only else [{"delta": {}, "finish_reason": "stop"}],
                "usage": usage,
            },
        ]
        if len(requests) == 3:
            chunks = [{"choices": [{"delta": {"content": "无统计"}, "finish_reason": "stop"}]}]
        return httpx.Response(
            200, text="".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
        )

    client = httpx.AsyncClient
    monkeypatch.setattr(
        "nexa_ai.openai_compatible.httpx.AsyncClient",
        lambda **kwargs: client(transport=httpx.MockTransport(response), **kwargs),
    )
    config = CodingSessionConfig(
        provider=DeepSeekProvider(name="deepseek", api_key="unused"),
        cwd=tmp_path,
        storage=JsonlStorage(tmp_path / "session.jsonl"),
    )
    session = CodingSession.load(config)
    state = TuiState()
    adapter = TuiEventAdapter(state)
    ends = []
    async for event in session.prompt("读取 file.txt"):
        adapter.apply(event)
        if isinstance(event, MessageEndEvent) and isinstance(event.message, AssistantMessage):
            ends.append(event)
    assert len(requests) == 2
    assert [event.message.usage.total_tokens for event in ends] == [120, 220]
    assert session.latest_usage.total_tokens == 220  # 最新实测，而非累加成 340。
    assert state.usage == session.latest_usage
    restored = CodingSession.load(config)
    assert restored.latest_usage == session.latest_usage
    async for event in restored.prompt("继续"):
        adapter.apply(event)
    assert restored.messages[-1].usage is None
    assert state.usage.total_tokens == 220
    assert restored.latest_usage.total_tokens == 220
    restored.select_model(config.provider, "deepseek-v4-pro", "deepseek")
    assert restored.latest_usage is None  # 不把另一个模型的实测值当成当前模型的数据。


@pytest.mark.asyncio
async def test_context_ui_and_session_restore(tmp_path):
    paths = NexaPaths(home=tmp_path / "home")
    provider = DeepSeekProvider(name="deepseek", api_key="unused")
    app = NexaTuiApp(provider, model="deepseek-flash", cwd=tmp_path, paths=paths)
    config = CodingSessionConfig(
        provider=provider, cwd=tmp_path, storage=JsonlStorage(app._session_path)
    )
    CodingSession.load(config)
    usage = TokenUsage(
        model="deepseek-flash", input_tokens=40_000, output_tokens=4_000, total_tokens=44_000
    )
    # 模拟已有账本中的完整模型响应，随后让 TUI 正常恢复。
    from nexa_agent.session.entries import MessageEntry

    config.storage.append(
        MessageEntry(
            id="message-1", parent_id="model", message=AssistantMessage(content="完成", usage=usage)
        )
    )
    async with app.run_test(size=(180, 32)) as pilot:
        await pilot.pause()
        assert app._state.usage == usage
        text = str(app.query_one("#context-usage", Static).render())
        assert text == "上下文 4.2% · 44k/1M"
        await pilot.resize_terminal(100, 32)
        await pilot.pause()
        assert not app.query_one("#input-help").display
        assert app.query_one("#context-usage").size.width >= Text(text).cell_len
        app.query_one(PromptInput).load_text("/usage")
        await pilot.press("enter")
        await pilot.pause()
        assert "不累加" in app._state.chat_items[-1].text
        assert "44,000 / 1,048,576 · 4.2%" in app._state.chat_items[-1].text
        app.query_one(PromptInput).load_text("/new")
        await pilot.press("enter")
        await pilot.pause()
        assert app._state.usage is None
        assert app._context_status() == "上下文 —"
        app._provider = FakeProvider([])
        app._state.usage = usage
        assert "上限未知" in app._context_status()
