"""思考设置经过 TUI、账本和实际 HTTP 请求构造的验证。"""

import json

import httpx
import pytest
from textual.widgets import Static

from nexa_agent.messages import AssistantMessage, ThinkingContent, ToolCall, ToolResultMessage
from nexa_agent.session.entries import ThinkingChangeEntry
from nexa_agent.session.storage import JsonlStorage
from nexa_ai.deepseek import DeepSeekProvider
from nexa_ai.openai_compatible import OpenAICompatibleProvider
from nexa_coding.paths import NexaPaths
from nexa_coding.session import CodingSession, CodingSessionConfig
from nexa_coding.tui.app import NexaTuiApp, PromptInput


def deepseek():
    return DeepSeekProvider(name="deepseek", api_key="unused")


@pytest.mark.asyncio
async def test_thinking_tool_loop_and_restore(tmp_path, monkeypatch):
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
        chunk = {"choices": [{"delta": delta, "finish_reason": "stop"}]}
        return httpx.Response(200, text=f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n")

    client_type = httpx.AsyncClient
    monkeypatch.setattr(
        "nexa_ai.openai_compatible.httpx.AsyncClient",
        lambda **kwargs: client_type(transport=httpx.MockTransport(response), **kwargs),
    )
    config = CodingSessionConfig(
        provider=deepseek(),
        model="deepseek-flash",
        cwd=tmp_path,
        storage=JsonlStorage(tmp_path / "session.jsonl"),
        system="test",
    )
    session = CodingSession.load(config)
    assert session.thinking == "high"
    with pytest.raises(ValueError, match="不支持"):
        session.set_thinking("medium")
    async for _ in session.prompt("读取文件"):
        pass
    assert len(requests) == 2
    for payload in requests:
        assert payload["thinking"] == {"type": "enabled"}
        assert payload["reasoning_effort"] == "high"
    history = requests[1]["messages"]
    assistant = next(m for m in history if m["role"] == "assistant")
    assert assistant["reasoning_content"] == "先读取文件"
    assert assistant["tool_calls"][0]["id"] == "call1"
    result = next(m for m in history if m["role"] == "tool")
    assert "hello" in result["content"]

    restored = CodingSession.load(config)
    assert restored.thinking == "high"
    async for _ in restored.prompt("继续"):
        pass
    assert requests[-1]["reasoning_effort"] == "high"
    assert all(m["reasoning_content"] for m in requests[-1]["messages"] if m["role"] == "assistant")
    restored.set_thinking("off")
    async for _ in restored.prompt("关闭思考"):
        pass
    assert requests[-1]["thinking"] == {"type": "disabled"}
    assert "reasoning_effort" not in requests[-1]
    for removed in ("default", "on"):
        with pytest.raises(ValueError, match="不支持"):
            restored.set_thinking(removed)
    assert restored.thinking == "off"
    for level in ("low", "max"):
        restored.set_thinking(level)
        async for _ in restored.prompt("继续"):
            pass
        assert requests[-1]["thinking"] == {"type": "enabled"}
        assert requests[-1]["reasoning_effort"] == level
    for old_setting in ("default", "on"):
        config.storage.append(ThinkingChangeEntry(id=f"old-{old_setting}", thinking=old_setting))
        migrated = CodingSession.load(config)
        assert migrated.thinking == "high"
        assert migrated.provider.thinking == "high"


def test_settings_failure_and_model_switch(tmp_path, monkeypatch):
    storage = JsonlStorage(tmp_path / "session.jsonl")
    config = CodingSessionConfig(
        provider=deepseek(),
        model="deepseek-flash",
        cwd=tmp_path,
        storage=storage,
    )
    session = CodingSession.load(config)
    previous = session.provider
    with monkeypatch.context() as patch:

        def fail(entry):
            raise OSError("disk failure")

        patch.setattr(storage, "append", fail)
        with pytest.raises(OSError):
            session.set_thinking("low")
    assert session.provider is previous
    assert session.thinking == "high"
    session.set_thinking("high")
    session._harness._is_running = True
    with pytest.raises(RuntimeError, match="运行中"):
        session.set_thinking("off")
    session._harness._is_running = False
    assert not session.select_model(deepseek(), "deepseek-v4-pro", "deepseek")
    assert session.thinking == "high"
    assert session.select_model(
        OpenAICompatibleProvider(name="other", api_key="unused"),
        "other",
        "other",
    )
    assert session.thinking == "default"
    config.provider = session.provider
    assert CodingSession.load(config).thinking == "default"

    provider = deepseek()
    for obsolete in ("deepseek-chat", "deepseek-reasoner", "deepseek-v4-flash"):
        with pytest.raises(ValueError, match="不支持 DeepSeek 模型"):
            provider.thinking_options(obsolete)
    message = AssistantMessage(
        content=[ThinkingContent(text="思考"), ToolCall(id="t", name="read", arguments={})]
    )
    messages = [message, ToolResultMessage(tool_call_id="t", tool_name="read", content="结果")]
    assert "reasoning_content" not in provider._request_messages("", messages, [])[0]


@pytest.mark.asyncio
async def test_tui_thinking_new_resume_and_display(tmp_path):
    paths = NexaPaths(home=tmp_path / "home")
    app = NexaTuiApp(deepseek(), model="deepseek-flash", cwd=tmp_path, paths=paths)

    async def submit(pilot, text):
        app.query_one(PromptInput).load_text(text)
        await pilot.press("enter")
        await pilot.pause()

    async with app.run_test(size=(120, 32)) as pilot:
        await submit(pilot, "/thinking")
        assert "当前：high；可用：off, low, high, max" in app._state.chat_items[-1].text
        await submit(pilot, "/thinking high")
        assert app._session.thinking == "high"
        assert "思考 high" in str(app.query_one("#environment", Static).render())
        before = app._state.show_thinking
        await pilot.press("ctrl+t")
        assert app._state.show_thinking != before
        assert app._session.thinking == "high"
        await submit(pilot, "/new")
        assert app._session.thinking == "high"
        await submit(pilot, "/thinking off")
        await submit(pilot, "/resume default")
        assert app._session.thinking == "high"
        assert app._provider.thinking == "high"
        app._state.set_running(True)
        await submit(pilot, "/thinking off")
        assert app._session.thinking == "high"
        assert "等待" in app._state.chat_items[-1].text
        app._state.set_running(False)


@pytest.mark.asyncio
async def test_model_command_recovers_obsolete_session(tmp_path):
    from nexa_agent.messages import UserMessage
    from nexa_agent.session.entries import MessageEntry, ModelChangeEntry
    from nexa_coding.session_manager import SessionManager

    paths = NexaPaths(home=tmp_path / "home")
    paths.home.mkdir()
    paths.config_file.write_text("""
[providers.deepseek]
base_url = "https://api.deepseek.com"
model = "deepseek-flash"
api_key = "unused"
""")
    manager = SessionManager(paths.project_session_dir(tmp_path))
    path = manager.current_path()
    storage = JsonlStorage(path)
    storage.append(ModelChangeEntry(id="model", model="deepseek-chat", provider="deepseek"))
    storage.append(
        MessageEntry(id="message-1", parent_id="model", message=UserMessage(content="旧会话"))
    )
    app = NexaTuiApp(deepseek(), model="deepseek-flash", cwd=tmp_path, paths=paths)
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app._session is None
        assert "恢复失败" in app._state.chat_items[-1].text
        app.query_one(PromptInput).load_text("/model deepseek")
        await pilot.press("enter")
        await pilot.pause()
        assert app._session.model == "deepseek-flash"
        assert app._session.thinking == "high"
        assert app._session.messages[0].text == "旧会话"
        assert any("旧会话" in item.text for item in app._state.chat_items)
        assert app._state.error is None
        assert path == app._session_path
    restored = CodingSession.load(
        CodingSessionConfig(provider=deepseek(), cwd=tmp_path, storage=storage)
    )
    assert restored.model == "deepseek-flash"
    assert restored.messages[0].text == "旧会话"
