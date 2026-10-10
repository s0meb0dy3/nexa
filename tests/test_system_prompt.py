"""测试系统提示词构建（Phase 10）。"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from nexa_agent.messages import AssistantMessage, TextContent
from nexa_agent.provider_events import ProviderResponseEndEvent, ProviderResponseStartEvent
from nexa_agent.session.entries import Entry
from nexa_agent.tools import AgentTool, AgentToolResult
from nexa_ai.fake import FakeProvider
from nexa_coding.session import CodingSession, CodingSessionConfig
from nexa_coding.skills import Skill
from nexa_coding.system_prompt import (
    BuildSystemPromptOptions,
    ProjectContextFile,
    build_system_prompt,
    format_guidelines,
)

# ── 辅助函数 ──────────────────────────────────────────────────────────────────


def _make_tool(name: str, description: str) -> AgentTool:
    """构造一个最小工具。"""

    async def execute(tool_call_id: str, arguments: dict) -> AgentToolResult:
        return AgentToolResult()

    return AgentTool(name=name, description=description, parameters={}, execute_fn=execute)


class InMemorySessionStorage:
    """把条目存在内存列表里，避免测试碰磁盘。"""

    def __init__(self) -> None:
        self._entries: list[Entry] = []

    def append(self, entry: Entry) -> None:
        self._entries.append(entry)

    def read_all(self) -> list[Entry]:
        return list(self._entries)


# ── 测试用例 ──────────────────────────────────────────────────────────────────


# 测试 1：默认 prompt 含身份 + 工具列表 + guidelines
def test_default_prompt_has_identity_tools_guidelines():
    tools = [_make_tool("read", "读取文件"), _make_tool("bash", "执行命令")]
    prompt = build_system_prompt(
        BuildSystemPromptOptions(cwd=Path("/work"), tools=tools, current_date=date(2026, 9, 10))
    )

    # 身份
    assert "编码 agent" in prompt
    # 工具列表
    assert "可用工具：" in prompt
    assert "- read: 读取文件" in prompt
    # 指南
    assert "使用指南：" in prompt
    assert "- read: 使用 read 工具时，读取文件" in prompt
    # 工作目录 + 日期
    assert "工作目录：/work" in prompt
    assert "当前日期：2026-09-10" in prompt


# 测试 2：guidelines 去重且顺序确定
def test_guidelines_dedup_and_ordered():
    tools = [_make_tool("read", "读取文件"), _make_tool("bash", "执行命令")]
    extra = ["- bash: 使用 bash 工具时，执行命令", "自定义指南"]
    prompt1 = format_guidelines(tools, extra)
    prompt2 = format_guidelines(tools, extra)

    # 确定性：两次输出一致。
    assert prompt1 == prompt2
    # extra 里重复的 bash 指南被去重（工具推导已有一条 bash）。
    assert prompt1.count("- bash: 使用 bash 工具时，执行命令") == 1
    # 顺序：工具指南在前，extra 在后，首次出现顺序保留。
    lines = prompt1.splitlines()
    assert lines.index("- read: 使用 read 工具时，读取文件") < lines.index("自定义指南")


# 测试 3：自定义 prompt 替换主体，但保留日期 + cwd
def test_custom_prompt_replaces_body_keeps_env():
    tools = [_make_tool("read", "读取文件")]
    custom = "你是专门做代码审查的专家。"
    prompt = build_system_prompt(
        BuildSystemPromptOptions(
            cwd=Path("/proj"), tools=tools, custom_prompt=custom, current_date=date(2026, 9, 10)
        )
    )

    # 自定义主体替换默认身份。
    assert "代码审查的专家" in prompt
    assert "编码 agent" not in prompt
    # 环境上下文仍保留。
    assert "工作目录：/proj" in prompt
    assert "当前日期：2026-09-10" in prompt


# 测试 4：技能索引在有技能时插入，且是"索引"（有 location 无全文）
def test_skills_inserted_when_present():
    skill = Skill(
        name="python-testing", path=Path("/s/skill.md"), description="写测试", content="正文"
    )
    # 有技能就插入，工具组合不影响（read 工具在 coding agent 里几乎必然存在）。
    prompt = build_system_prompt(
        BuildSystemPromptOptions(cwd=Path("/w"), tools=[_make_tool("bash", "执行")], skills=[skill])
    )
    assert "<available_skills>" in prompt
    assert "python-testing" in prompt
    assert "正文" not in prompt  # 只放索引，不放全文

    # 无技能时不插技能段。
    prompt_no_skills = build_system_prompt(
        BuildSystemPromptOptions(cwd=Path("/w"), tools=[_make_tool("read", "读取文件")])
    )
    assert "<available_skills>" not in prompt_no_skills


# 测试 5：项目上下文 XML 格式正确
def test_project_context_xml():
    files = [ProjectContextFile(path=Path("AGENTS.md"), content="规则：不要删除文件")]
    prompt = build_system_prompt(
        BuildSystemPromptOptions(
            cwd=Path("/w"), tools=[_make_tool("read", "读取文件")], context_files=files
        )
    )

    assert '<project_instructions file="AGENTS.md">' in prompt
    assert "规则：不要删除文件" in prompt
    assert "</project_instructions>" in prompt


# 测试 6：CodingSession.load() 集成——不给 system 时自动构建
@pytest.mark.asyncio
async def test_load_auto_builds_system():
    storage = InMemorySessionStorage()
    provider = FakeProvider(
        [
            [
                ProviderResponseStartEvent(model="test-model"),
                ProviderResponseEndEvent(
                    message=AssistantMessage(content=[TextContent(text="好的")]),
                    finish_reason="stop",
                ),
            ]
        ]
    )
    config = CodingSessionConfig(provider=provider, model="test-model", storage=storage, cwd=".")
    session = CodingSession.load(config)

    # 没给 system，自动构建的提示词含工具列表。
    assert "可用工具：" in session._harness._config.system  # 内部字段，仅测试用
