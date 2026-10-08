"""系统提示词的构建：把身份、工具、技能、项目上下文拼成一段提示词。

这是 Phase 10 的核心纯函数层：build_system_prompt 没有 IO、没有随机，
同一组输入两次调用输出完全一致，因此可单测、可复用；
CodingSession 加载会话时用它构建提示词。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from nexa_agent.tools import AgentTool
from nexa_coding.skills import Skill

# 默认身份描述。
DEFAULT_IDENTITY = (
    "你是一个在终端里工作的编码 agent。你帮助用户读写文件、执行命令、"
    "修改代码。回答要简洁，需要时使用工具。"
)


@dataclass(frozen=True)
class ProjectContextFile:
    """一个项目上下文文件（例如 AGENTS.md）。

    这里只负责格式化，不负责发现文件——发现是 Phase 19 的事。
    """

    path: Path
    content: str


@dataclass
class BuildSystemPromptOptions:
    """build_system_prompt 的输入。

    Attributes:
        cwd: 工作目录。
        tools: 可用工具列表。
        skills: 技能列表（只放索引进 prompt）。
        custom_prompt: 自定义主体；给了就替换默认身份段。
        context_files: 项目上下文文件。
        extra_guidelines: 额外使用指南。
        current_date: 当前日期，用于提示词里的时间戳。
    """

    cwd: Path
    tools: Sequence[AgentTool] = ()
    skills: Sequence[Skill] = ()
    custom_prompt: str | None = None
    context_files: Sequence[ProjectContextFile] = ()
    extra_guidelines: Sequence[str] = ()
    current_date: date | None = None


def build_system_prompt(options: BuildSystemPromptOptions) -> str:
    """把各段拼成完整系统提示词。

    结构（每段存在才出现）：
        身份（或自定义主体）→ 工作目录 → 日期 → 工具 → 指南 → 技能 → 项目上下文
    """
    sections: list[str] = []

    # 主体：自定义 prompt 替换默认身份，但环境上下文始终追加。
    sections.append(options.custom_prompt or DEFAULT_IDENTITY)

    sections.append(f"工作目录：{options.cwd}")

    if options.current_date is not None:
        sections.append(f"当前日期：{options.current_date.isoformat()}")

    # 工具列表始终在（模型需要知道有哪些工具）。
    sections.append(format_available_tools(options.tools))

    # 指南：工具指南 + 额外指南，去重、定序。
    sections.append(format_guidelines(options.tools, options.extra_guidelines))

    # 技能索引：有技能就插——read 工具在 coding agent 里几乎必然存在，
    # 不需要额外检查。
    if options.skills:
        sections.append(format_skills_for_prompt(options.skills))

    # 项目上下文。
    if options.context_files:
        sections.append(format_project_context(options.context_files))

    return "\n\n".join(section for section in sections if section)


def format_available_tools(tools: Sequence[AgentTool]) -> str:
    """生成可用工具列表，如 "- read: Read file contents"。"""

    if not tools:
        return ""
    lines = ["可用工具："]
    for tool in tools:
        lines.append(f"- {tool.name}: {tool.description}")
    return "\n".join(lines)


def format_guidelines(tools: Sequence[AgentTool], extra: Sequence[str]) -> str:
    """收集使用指南（从工具推导 + 额外），去重并保持确定顺序。"""

    # 每个工具推导一条"如何使用"的指南。
    derived = [f"- {tool.name}: 使用 {tool.name} 工具时，{tool.description}" for tool in tools]

    # dict.fromkeys 保持首次出现顺序并去重。
    combined = list(dict.fromkeys([*derived, *extra]))
    if not combined:
        return ""
    lines = ["使用指南："]
    lines.extend(combined)
    return "\n".join(lines)


def format_project_context(files: Sequence[ProjectContextFile]) -> str:
    """把项目上下文文件格式化成 <project_instructions> XML。"""

    blocks = []
    for file in files:
        blocks.append(
            f'<project_instructions file="{file.path}">\n{file.content}\n</project_instructions>'
        )
    return "\n".join(blocks)


def format_skills_for_prompt(skills: Sequence[Skill]) -> str:
    """把技能列表格式化成 <available_skills> XML（只放索引，不放全文）。

    每个技能列出 name / description / location，模型要用时自己 read 技能文件。
    """

    lines = ["<available_skills>"]
    for skill in skills:
        lines.append(
            f'<skill name="{skill.name}" location="{skill.path}">{skill.description}</skill>'
        )
    lines.append("</available_skills>")
    return "\n".join(lines)


__all__ = [
    "BuildSystemPromptOptions",
    "ProjectContextFile",
    "build_system_prompt",
    "format_available_tools",
    "format_guidelines",
    "format_project_context",
    "format_skills_for_prompt",
]
