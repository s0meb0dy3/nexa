"""项目指令的来源、边界与实际请求；不访问用户真实配置或模型 API。"""

import pytest

from nexa_agent.messages import AssistantMessage
from nexa_agent.provider_events import ProviderResponseEndEvent
from nexa_agent.session.storage import JsonlStorage
from nexa_ai.fake import FakeProvider
from nexa_coding.context import PROJECT_MARKERS, discover_project_context
from nexa_coding.paths import NexaPaths
from nexa_coding.resources import NexaResourcePaths, ResourceError
from nexa_coding.session import CodingSession, CodingSessionConfig


def write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path.resolve()


def resources(tmp_path, cwd):
    return NexaResourcePaths(
        cwd=cwd, paths=NexaPaths(home=tmp_path / "home", agents_home=tmp_path / "agents")
    )


def test_context_order_scope_and_deduplication(tmp_path):
    root = tmp_path / "repo"
    cwd = root / "src" / "service"
    cwd.mkdir(parents=True)
    (root / ".git").write_text("gitdir: elsewhere")
    paths = resources(tmp_path, cwd)
    expected = [
        write(paths.paths.home / "AGENTS.md", "全局一"),
        write(paths.paths.agents_home / "AGENTS.md", "全局二"),
        write(root / "AGENTS.md", "项目规则"),
        write(root / "src" / "AGENTS.md", "src 规则"),
        write(cwd / "AGENTS.md", "service 规则"),
        write(cwd / ".nexa" / "AGENTS.md", "nexa 规则"),
        write(cwd / ".agents" / "AGENTS.md", "agents 规则"),
    ]
    write(tmp_path / "AGENTS.md", "项目外，不加载")
    write(cwd / "child" / "AGENTS.md", "子目录，不扫描")
    files = discover_project_context(paths)
    assert [file.path for file in files] == expected
    assert [file.content for file in files] == [
        "全局一",
        "全局二",
        "项目规则",
        "src 规则",
        "service 规则",
        "nexa 规则",
        "agents 规则",
    ]
    expected[-1].unlink()
    (cwd / ".agents" / "AGENTS.md").symlink_to(expected[0])
    assert [file.path for file in discover_project_context(paths)] == expected[:-1]


@pytest.mark.parametrize("marker", PROJECT_MARKERS)
def test_nearest_project_root_wins(tmp_path, marker):
    outer = tmp_path / "outer"
    inner = outer / "inner"
    cwd = inner / "src"
    cwd.mkdir(parents=True)
    (outer / ".git").mkdir()
    (inner / marker).touch()
    write(outer / "AGENTS.md", "外层规则")
    expected = write(inner / "AGENTS.md", "内层规则")
    assert [file.path for file in discover_project_context(resources(tmp_path, cwd))] == [expected]


def test_no_marker_only_loads_cwd_and_global_only_mode(tmp_path):
    cwd = tmp_path / "plain" / "child"
    write(cwd.parent / "AGENTS.md", "不自动加载")
    expected = write(cwd / "AGENTS.md", "本目录")
    paths = resources(tmp_path, cwd)
    assert [file.path for file in discover_project_context(paths)] == [expected]
    assert discover_project_context(NexaResourcePaths(paths=paths.paths)) == ()


async def test_session_loads_rules_once_then_refreshes_on_restore(tmp_path):
    cwd = tmp_path / "repo"
    rule = write(cwd / "AGENTS.md", "规则版本一")
    provider = FakeProvider(
        [[ProviderResponseEndEvent(message=AssistantMessage(content="完成"))] for _ in range(3)]
    )
    cfg = CodingSessionConfig(
        provider=provider,
        model="test",
        cwd=cwd,
        resource_paths=resources(tmp_path, cwd),
        storage=JsonlStorage(tmp_path / "session.jsonl"),
    )
    session = CodingSession.load(cfg)
    async for _ in session.prompt("开始"):
        pass
    system = provider.calls[0][1]
    assert str(rule) in system
    assert "规则版本一" in system
    assert "不覆盖或丢弃" in system
    assert "未运行的检查不能声称通过" in system
    assert "用户当前明确要求优先" in system
    assert "当前日期" not in system
    rule.write_text("规则版本二")
    async for _ in session.prompt("继续"):
        pass
    assert provider.calls[1][1] == system
    restored = CodingSession.load(cfg)
    async for _ in restored.prompt("再继续"):
        pass
    assert "规则版本二" in provider.calls[2][1]
    assert "规则版本一" not in provider.calls[2][1]


def test_invalid_instruction_fails_clearly_but_explicit_system_bypasses_discovery(tmp_path):
    (tmp_path / "AGENTS.md").write_bytes(b"\xff")
    paths = resources(tmp_path, tmp_path)
    with pytest.raises(ResourceError, match="无法读取项目指令"):
        discover_project_context(paths)
    session = CodingSession.load(
        CodingSessionConfig(
            provider=FakeProvider([]),
            model="test",
            cwd=tmp_path,
            system="explicit",
            resource_paths=paths,
            storage=JsonlStorage(tmp_path / "session.jsonl"),
        )
    )
    assert session._harness.system == "explicit"
