"""测试事件渲染器（Phase 11）：text / json / transcript 三种输出形态。"""

from __future__ import annotations

import json
from io import StringIO

from nexa_agent.events import (
    AgentEndEvent,
    AgentStartEvent,
    MessageEndEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
)
from nexa_agent.messages import AssistantMessage, TextContent, ThinkingContent
from nexa_agent.tools import AgentToolResult
from nexa_coding.rendering import PrintOutputMode, create_event_renderer

# ── 辅助函数 ──────────────────────────────────────────────────────────────────


def _success_events() -> list:
    """一组"成功运行"的事件流：开始 → 工具执行 → 结束（有最终答案）。"""
    return [
        AgentStartEvent(),
        ToolExecutionStartEvent(tool_call_id="c1", tool_name="read", args={}),
        ToolExecutionEndEvent(
            tool_call_id="c1",
            tool_name="read",
            result=AgentToolResult(content=[TextContent(text="文件内容")]),
            is_error=False,
        ),
        AgentEndEvent(messages=[AssistantMessage(content=[TextContent(text="总结完成")])]),
    ]


def _error_answer_events() -> list:
    """一组"最终答案是错误"的事件流。"""
    return [
        AgentEndEvent(messages=[AssistantMessage(content=[TextContent(text="错误: 连接失败")])])
    ]


def _render(renderer, events: list) -> bool:
    """把事件流喂给渲染器并返回 finish() 结果。"""
    for event in events:
        renderer.render(event)
    return renderer.finish()


# ── 测试用例 ──────────────────────────────────────────────────────────────────


# 测试 1：text 模式只输出最终答案，工具事件不进输出
def test_text_mode_only_final_answer():
    stdout, stderr = StringIO(), StringIO()
    renderer = create_event_renderer(PrintOutputMode.text, stdout=stdout, stderr=stderr)

    ok = _render(renderer, _success_events())

    assert ok is True
    # 只有最终答案，工具过程不出现在输出里。
    assert stdout.getvalue() == "总结完成\n"
    assert stderr.getvalue() == ""


# 测试 2：json 模式每个事件一行可解析 JSON
def test_json_mode_each_event_a_line():
    stdout, stderr = StringIO(), StringIO()
    renderer = create_event_renderer(PrintOutputMode.json, stdout=stdout, stderr=stderr)

    ok = _render(renderer, _success_events())

    assert ok is True
    lines = stdout.getvalue().strip().splitlines()
    # 4 个事件 → 4 行 JSON，每行都可解析。
    assert len(lines) == 4
    for line in lines:
        obj = json.loads(line)
        assert "type" in obj
    # stderr 应为空。
    assert stderr.getvalue() == ""


# 测试 3：transcript 模式助手进 stdout、工具进 stderr
def test_transcript_mode_splits_streams():
    stdout, stderr = StringIO(), StringIO()
    renderer = create_event_renderer(PrintOutputMode.transcript, stdout=stdout, stderr=stderr)

    ok = _render(renderer, _success_events())

    assert ok is True
    # 最终答案进 stdout。
    assert "总结完成" in stdout.getvalue()
    # 工具过程进 stderr。
    assert "工具开始：read" in stderr.getvalue()
    assert "工具结束：read" in stderr.getvalue()


# 测试 4：非可恢复错误 → finish() 返回 False
def test_text_mode_error_answer_returns_false():
    stdout, stderr = StringIO(), StringIO()
    renderer = create_event_renderer(PrintOutputMode.text, stdout=stdout, stderr=stderr)

    ok = _render(renderer, _error_answer_events())

    assert ok is False
    # 错误详情进 stderr，stdout 保持干净。
    assert stdout.getvalue() == ""
    assert "连接失败" in stderr.getvalue()


# ── 思考块展示 ────────────────────────────────────────────────────────────────


def _thinking_events() -> list:
    """一条带思考的助手消息：MessageEnd（带思考）+ AgentEnd（含最终答案）。"""
    message = AssistantMessage(
        content=[ThinkingContent(text="先分析一下"), TextContent(text="最终答案")]
    )
    return [MessageEndEvent(message=message), AgentEndEvent(messages=[message])]


def test_text_mode_hides_thinking():
    """text 模式的 stdout 只出答案，思考不出现（管道保持干净）。"""
    stdout, stderr = StringIO(), StringIO()
    renderer = create_event_renderer(PrintOutputMode.text, stdout=stdout, stderr=stderr)

    ok = _render(renderer, _thinking_events())

    assert ok is True
    assert stdout.getvalue() == "最终答案\n"
    assert "先分析一下" not in stdout.getvalue()
    assert stderr.getvalue() == ""


def test_transcript_mode_sends_thinking_to_stderr():
    """transcript 模式：思考进 stderr，答案进 stdout。"""
    stdout, stderr = StringIO(), StringIO()
    renderer = create_event_renderer(PrintOutputMode.transcript, stdout=stdout, stderr=stderr)

    ok = _render(renderer, _thinking_events())

    assert ok is True
    assert "最终答案" in stdout.getvalue()
    assert "先分析一下" not in stdout.getvalue()
    assert "先分析一下" in stderr.getvalue()


def test_json_mode_includes_thinking():
    """json 模式序列化的事件里包含思考块。"""
    stdout, stderr = StringIO(), StringIO()
    renderer = create_event_renderer(PrintOutputMode.json, stdout=stdout, stderr=stderr)

    _render(renderer, _thinking_events())

    output = stdout.getvalue()
    assert '"type":"thinking"' in output
    assert "先分析一下" in output
