"""测试 NexaPaths：用户数据位置的唯一真相源。"""

from __future__ import annotations

from pathlib import Path

from nexa_coding.paths import NexaPaths, _slugify_path


def test_canonical_user_paths():
    """用户级各路径都基于 home / agents_home。"""
    home = Path("/fake-home/.nexa")
    agents = Path("/fake-agents")
    paths = NexaPaths(home=home, agents_home=agents)

    assert paths.sessions_dir == home / "sessions"
    assert paths.user_skills_dir == home / "skills"
    assert paths.user_prompts_dir == home / "prompts"
    assert paths.config_file == home / "config.toml"
    assert paths.user_agents_skills_dir == agents / "skills"
    assert paths.user_agents_prompts_dir == agents / "prompts"


def test_project_paths_follow_cwd():
    """项目级路径基于 cwd。"""
    paths = NexaPaths()
    cwd = Path("/work/demo")

    assert paths.project_skills_dir(cwd) == cwd / ".nexa" / "skills"
    assert paths.project_agents_skills_dir(cwd) == cwd / ".agents" / "skills"
    assert paths.project_prompts_dir(cwd) == cwd / ".nexa" / "prompts"
    assert paths.project_agents_prompts_dir(cwd) == cwd / ".agents" / "prompts"


def test_project_session_dir_isolated():
    """两个不同项目 → 不同会话目录（可读名 + 短哈希，绝不重名）。"""
    paths = NexaPaths()
    dir_a = paths.project_session_dir(Path("/work/demo"))
    dir_b = paths.project_session_dir(Path("/work/other"))

    assert dir_a != dir_b
    # 都在集中的 sessions 目录下。
    assert dir_a.parent == paths.sessions_dir
    assert dir_b.parent == paths.sessions_dir
    # 同一项目两次计算结果一致（稳定）。
    assert dir_a == paths.project_session_dir(Path("/work/demo"))


def test_default_session_path():
    """default_session_path = project_session_dir/default.jsonl。"""
    paths = NexaPaths()
    cwd = Path("/work/demo")

    assert paths.default_session_path(cwd) == paths.project_session_dir(cwd) / "default.jsonl"


def test_slugify_path():
    """路径清洗：非法字符转 -，压缩连续 -，去首尾；中文保留（isalnum）。"""
    assert _slugify_path(Path("/work/my proj")) == "work-my-proj"
    assert _slugify_path(Path("/a//b")) == "a-b"
    assert _slugify_path(Path("/中文/项目")) == "中文-项目"  # 中文是字母类字符，保留
    # 即使可读名不同也要稳定；短哈希保证清洗后同名的路径仍隔离。
    paths = NexaPaths()
    assert paths.project_session_dir(Path("/a b")) != paths.project_session_dir(Path("/a-b"))
