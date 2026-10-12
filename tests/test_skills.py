"""测试技能（skills）加载、展开、索引，以及 CodingSession.prompt 里的展开。"""

from __future__ import annotations

from pathlib import Path

import pytest

from nexa_agent.messages import AssistantMessage, TextContent, ToolCall
from nexa_agent.provider_events import ProviderResponseEndEvent, ProviderResponseStartEvent
from nexa_agent.session.entries import Entry
from nexa_ai.fake import FakeProvider
from nexa_coding.paths import NexaPaths
from nexa_coding.resources import NexaResourcePaths, ResourceError, parse_markdown_resource
from nexa_coding.session import CodingSession, CodingSessionConfig
from nexa_coding.skills import Skill, build_skill_index, expand_skill_command, load_skills


async def test_session_can_read_global_skill_and_reference(tmp_path):
    cwd = tmp_path / "project"
    cwd.mkdir()
    paths = NexaPaths(home=tmp_path / "home", agents_home=tmp_path / "agents")
    directory = paths.user_skills_dir / "review"
    directory.mkdir(parents=True)
    (directory / "SKILL.md").write_text("---\ndescription: 检查代码\n---\n读取 checklist.md")
    (directory / "checklist.md").write_text("必须检查测试")
    provider = FakeProvider(
        [
            [
                ProviderResponseEndEvent(
                    message=AssistantMessage(
                        content=[
                            ToolCall(
                                id="skill",
                                name="read",
                                arguments={"path": str(directory / "SKILL.md")},
                            ),
                            ToolCall(
                                id="ref",
                                name="read",
                                arguments={"path": str(directory / "checklist.md")},
                            ),
                        ]
                    )
                )
            ],
            [ProviderResponseEndEvent(message=AssistantMessage(content="完成"))],
        ]
    )
    session = CodingSession.load(
        CodingSessionConfig(
            provider=provider,
            model="test",
            cwd=cwd,
            storage=InMemorySessionStorage(),
            resource_paths=NexaResourcePaths(cwd=cwd, paths=paths),
        )
    )
    async for _ in session.prompt("检查代码"):
        pass
    _, system, messages, _ = provider.calls[1]
    assert str(directory / "SKILL.md") in system
    assert "读取 checklist.md" in messages[-2].text
    assert "必须检查测试" in messages[-1].text
    assert not messages[-1].is_error


# ── 辅助函数 ──────────────────────────────────────────────────────────────────


def _write_skill_dir(directory: Path, rel_path: str, text: str) -> None:
    """在某个目录下写一个技能文件。"""
    file = directory / rel_path
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(text, encoding="utf-8")


class InMemorySessionStorage:
    """把条目存在内存列表里，避免测试碰磁盘。"""

    def __init__(self) -> None:
        self._entries: list[Entry] = []

    def append(self, entry: Entry) -> None:
        self._entries.append(entry)

    def read_all(self) -> list[Entry]:
        return list(self._entries)


# ── 测试用例 ──────────────────────────────────────────────────────────────────


# 测试 1：frontmatter 解析（有/无）
def test_parse_markdown_resource_with_and_without_frontmatter():
    """有 frontmatter 时解析出元数据，没有时返回空元数据。"""
    with_frontmatter = """---
name: python-testing
description: 写 pytest 测试
---
正文第一行
正文第二行"""
    metadata, body = parse_markdown_resource(with_frontmatter)
    assert metadata == {"name": "python-testing", "description": "写 pytest 测试"}
    assert body == "正文第一行\n正文第二行"

    # 没有 frontmatter：原样返回。
    no_frontmatter = "直接就是正文"
    metadata2, body2 = parse_markdown_resource(no_frontmatter)
    assert metadata2 == {}
    assert body2 == "直接就是正文"


# 测试 2：skill 加载（两种布局，用户级目录）
def test_load_skills_two_layouts(tmp_path):
    """目录布局和文件布局都能加载。"""
    # agents_home 也指向 tmp，避免扫到真实 ~/.agents 里的技能。
    paths = NexaPaths(home=tmp_path, agents_home=tmp_path / "agents")
    skills_dir = paths.user_skills_dir
    _write_skill_dir(
        skills_dir, "python-testing/SKILL.md", "---\ndescription: 写 pytest 测试\n---\n技能正文 1"
    )
    _write_skill_dir(skills_dir, "bash-helper.md", "---\ndescription: bash 小助手\n---\n技能正文 2")

    skills = load_skills(NexaResourcePaths(paths=paths))
    # 按名字排序：bash-helper, python-testing。
    assert [s.name for s in skills] == ["bash-helper", "python-testing"]
    assert skills[0].description == "bash 小助手"
    assert skills[0].content == "技能正文 2"
    assert skills[1].description == "写 pytest 测试"


# 测试 3：同目录重名抛 ResourceError
def test_load_skills_duplicate_name_raises(tmp_path):
    """同一目录内同名技能（目录布局 + 文件布局）是真冲突，抛 ResourceError。"""
    paths = NexaPaths(home=tmp_path, agents_home=tmp_path / "agents")
    skills_dir = paths.user_skills_dir
    _write_skill_dir(skills_dir, "dup/SKILL.md", "---\ndescription: 目录版\n---\n正文")
    _write_skill_dir(skills_dir, "dup.md", "---\ndescription: 文件版\n---\n正文")

    with pytest.raises(ResourceError, match="重名"):
        load_skills(NexaResourcePaths(paths=paths))


# 测试 4：/skill:name 展开产物正确（带 <skill> 包裹）
def test_expand_skill_command():
    """命中 /skill:name 时返回 <skill> 包裹的展开文本。"""
    skill = Skill(
        name="python-testing",
        path=__file__,
        description="写 pytest 测试",
        content="写一个 pytest 测试文件",
    )
    expanded = expand_skill_command("/skill:python-testing add tests", [skill])
    assert expanded is not None
    assert '<skill name="python-testing">' in expanded
    assert "写一个 pytest 测试文件" in expanded
    assert "<arguments>" in expanded and "add tests" in expanded

    # 未命中返回 None。
    assert expand_skill_command("普通问题", [skill]) is None
    with pytest.raises(ResourceError, match="未知技能"):
        expand_skill_command("/skill:nonexistent", [skill])
    with pytest.raises(ResourceError, match="指定技能名称"):
        expand_skill_command("/skill:", [skill])
    assert f"相对路径以 {Path(skill.path).resolve().parent}" in expanded
    assert "add tests" in expand_skill_command("/skill:python-testing\tadd tests", [skill])


# 测试 5：build_skill_index 输出
def test_build_skill_index():
    """输出 '- name: description' 列表。"""
    skills = [
        Skill(name="python-testing", path=__file__, description="写 pytest 测试", content="正文"),
        Skill(name="bash-helper", path=__file__, description="bash 小助手", content="正文"),
    ]
    index = build_skill_index(skills)
    assert "可用技能：" in index
    assert "- python-testing: 写 pytest 测试" in index
    assert "- bash-helper: bash 小助手" in index

    # 空列表返回空串。
    assert build_skill_index([]) == ""


# 测试 6：CodingSession.prompt() 里 skill 展开生效（fake provider）
@pytest.mark.asyncio
async def test_prompt_expands_skill_command():
    """prompt('/skill:...') 时，harness 收到的消息应含展开后的技能文本。"""
    skill = Skill(
        name="python-testing",
        path=__file__,
        description="写 pytest 测试",
        content="写一个 pytest 测试文件",
    )
    storage = InMemorySessionStorage()
    provider = FakeProvider(
        [
            [
                ProviderResponseStartEvent(model="test-model"),
                ProviderResponseEndEvent(
                    message=AssistantMessage(content=[TextContent(text="好的，我来写测试")]),
                    finish_reason="stop",
                ),
            ]
        ]
    )
    config = CodingSessionConfig(
        provider=provider,
        model="test-model",
        system="你是助手",
        storage=storage,
        cwd=".",
        skills=[skill],
    )
    session = CodingSession.load(config)

    async for _ in session.prompt("/skill:python-testing add tests"):
        pass

    # FakeProvider 记录下了发给模型的完整消息列表。
    assert provider.calls
    _, _, messages, _ = provider.calls[0]
    user_text = messages[0].text
    assert '<skill name="python-testing">' in user_text
    assert "写一个 pytest 测试文件" in user_text
    assert "add tests" in user_text


# ── Phase 13：多来源发现 ─────────────────────────────────────────────────────


# 新测试 1：skills_dirs 四级优先级顺序
def test_skills_dirs_priority_order():
    """cwd 给定时 4 级来源按优先级递增；cwd 为 None 时只有用户级 2 级。"""
    paths = NexaPaths(home=Path("/h"), agents_home=Path("/a"))
    cwd = Path("/work")

    dirs = NexaResourcePaths(cwd=cwd, paths=paths).skills_dirs
    assert dirs == (
        Path("/h/skills"),
        Path("/a/skills"),
        cwd / ".nexa" / "skills",
        cwd / ".agents" / "skills",
    )

    # 无 cwd：只有用户级。
    dirs_user = NexaResourcePaths(paths=paths).skills_dirs
    assert dirs_user == (Path("/h/skills"), Path("/a/skills"))


# 新测试 2：项目资源覆盖用户资源（同名技能，项目版赢）
def test_project_skill_overrides_user_skill(tmp_path):
    """同名技能：项目 .agents/skills 里的版本覆盖用户 ~/.nexa/skills 的。"""
    home = tmp_path / "home"
    agents = tmp_path / "agents"
    cwd = tmp_path / "proj"
    paths = NexaPaths(home=home, agents_home=agents)

    # 用户级和项目级各放一个同名技能，内容不同。
    _write_skill_dir(
        paths.user_skills_dir, "review.md", "---\ndescription: 用户版\n---\n用户版正文"
    )
    _write_skill_dir(
        paths.project_agents_skills_dir(cwd),
        "review.md",
        "---\ndescription: 项目版\n---\n项目版正文",
    )

    skills = load_skills(NexaResourcePaths(cwd=cwd, paths=paths))
    # 只有一个 review，且是项目版。
    assert [s.name for s in skills] == ["review"]
    assert skills[0].description == "项目版"
    assert skills[0].content == "项目版正文"


# 新测试 3：.agents 根目录不被当技能目录扫
def test_agents_root_not_treated_as_skill_dir(tmp_path):
    """.agents 下散落的文件不是技能；只有 .agents/skills 子目录被扫描。"""
    home = tmp_path / "home"
    agents = tmp_path / "agents"
    paths = NexaPaths(home=home, agents_home=agents)

    # .agents 根目录放 AGENTS.md 和一个伪技能文件——都不该被当成技能。
    agents.mkdir(parents=True)
    (agents / "AGENTS.md").write_text("这是指令文件，不是技能", encoding="utf-8")
    (agents / "loose.md").write_text("散落在 .agents 根目录，不是技能", encoding="utf-8")
    # 真正的技能放 .agents/skills 子目录。
    _write_skill_dir(paths.user_agents_skills_dir, "real.md", "---\ndescription: 真技能\n---\n正文")

    skills = load_skills(NexaResourcePaths(paths=paths))
    # 只有子目录里的 real.md 被加载。
    assert [s.name for s in skills] == ["real"]


# 新测试 4：CodingSession.load() 自动看到项目资源
@pytest.mark.asyncio
async def test_load_sees_project_skills(tmp_path):
    """load() 用 config.cwd 构造资源发现，项目技能自动可用。"""
    home = tmp_path / "home"
    agents = tmp_path / "agents"
    cwd = tmp_path / "proj"
    paths = NexaPaths(home=home, agents_home=agents)
    _write_skill_dir(
        paths.project_agents_skills_dir(cwd),
        "proj-only.md",
        "---\ndescription: 项目专属\n---\n项目技能正文",
    )

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
    config = CodingSessionConfig(
        provider=provider,
        model="test-model",
        system="你是助手",
        storage=InMemorySessionStorage(),
        cwd=cwd,
        resource_paths=NexaResourcePaths(cwd=cwd, paths=paths),
    )
    session = CodingSession.load(config)

    # 项目技能被自动加载，可以触发。
    names = [s.name for s in session.skills]
    assert names == ["proj-only"]
    expanded = expand_skill_command("/skill:proj-only", session.skills)
    assert expanded is not None and "项目技能正文" in expanded


# 新测试 5：prompt_templates 多目录覆盖
def test_prompt_templates_project_overrides_user(tmp_path):
    """同名模板：项目级覆盖用户级。"""
    from nexa_coding.prompt_templates import load_prompt_templates

    home = tmp_path / "home"
    agents = tmp_path / "agents"
    cwd = tmp_path / "proj"
    paths = NexaPaths(home=home, agents_home=agents)

    (paths.user_prompts_dir).mkdir(parents=True)
    (paths.user_prompts_dir / "review.md").write_text("用户版 {{what}}", encoding="utf-8")
    project_dir = paths.project_agents_prompts_dir(cwd)
    project_dir.mkdir(parents=True)
    (project_dir / "review.md").write_text("项目版 {{what}}", encoding="utf-8")

    templates = load_prompt_templates(NexaResourcePaths(cwd=cwd, paths=paths))
    assert [t.name for t in templates] == ["review"]
    assert templates[0].template == "项目版 {{what}}"
