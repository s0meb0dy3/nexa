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

    这里只定义数据，文件发现和读取由 context.py 负责。
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

    principles = [
        "- 修改前读取相关代码和适用的项目指令。",
        "- 保持修改聚焦，遵循项目架构与风格，不覆盖或丢弃用户已有的无关修改。",
        "- 使用项目规定的命令和包管理器，修改后运行相关测试及格式、lint、类型检查。",
        "- 如实报告修改与检查结果，未运行的检查不能声称通过。",
        "- 破坏性操作或重大设计歧义尚未得到明确授权时，先向用户确认。",
        "- 回答简洁，说明关键文件路径和未完成事项。",
    ]
    # dict.fromkeys 保持首次出现顺序并去重。
    combined = list(dict.fromkeys([*derived, *principles, *extra]))
    lines = ["使用指南："]
    lines.extend(combined)
    return "\n".join(lines)


def format_project_context(files: Sequence[ProjectContextFile]) -> str:
    """把项目上下文文件格式化成 <project_instructions> XML。"""

    blocks = [
        "用户级项目指令适用于所有项目。目录内 AGENTS.md 的规则适用于该目录及子目录；"
        ".nexa/AGENTS.md 和 .agents/AGENTS.md 适用于它们所在的启动目录。"
        "同一范围内的规则依次叠加；发生冲突时，用户当前明确要求优先，"
        "其后在适用范围内遵循更具体目录的规则。"
    ]
    for file in files:
        blocks.append(
            f'<project_instructions file="{file.path}">\n{file.content}\n</project_instructions>'
        )
    return "\n".join(blocks)


def format_skills_for_prompt(skills: Sequence[Skill]) -> str:
    """把技能列表格式化成 <available_skills> XML（只放索引，不放全文）。

    每个技能列出 name / description / location，模型要用时自己 read 技能文件。
    """

    lines = [
        "任务符合技能描述时，先用 read 读取完整 SKILL.md，再按说明执行。",
        "技能引用的相对路径以 SKILL.md 所在目录为基准。",
        "技能不会自动执行脚本，需要时通过已有工具执行。",
        "<available_skills>",
    ]
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
