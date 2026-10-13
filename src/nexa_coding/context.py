"""启动时发现项目指令：全局文件在前，项目根到 cwd 的文件逐级追加。"""

from nexa_coding.resources import NexaResourcePaths, ResourceError
from nexa_coding.system_prompt import ProjectContextFile

PROJECT_MARKERS = (".git", "pyproject.toml", "uv.lock", "setup.py", "package.json")


def discover_project_context(paths: NexaResourcePaths) -> tuple[ProjectContextFile, ...]:
    """只读取祖先链，不扫描子目录；同名规则叠加，相同真实路径去重。"""
    candidates = [paths.paths.home / "AGENTS.md", paths.paths.agents_home / "AGENTS.md"]
    if paths.cwd is not None:
        cwd = paths.cwd.expanduser().resolve()
        ancestors = (cwd, *cwd.parents)
        root = next(
            (
                directory
                for directory in ancestors
                if any((directory / marker).exists() for marker in PROJECT_MARKERS)
            ),
            cwd,
        )
        # 找最近项目根，避免把仓库外的父目录规则自动带入。
        chain = ancestors[: ancestors.index(root) + 1]
        candidates.extend(directory / "AGENTS.md" for directory in reversed(chain))
        candidates.extend((cwd / ".nexa" / "AGENTS.md", cwd / ".agents" / "AGENTS.md"))

    seen = set()
    files = []
    for candidate in candidates:
        if not candidate.is_file():
            continue
        path = candidate.expanduser().resolve()
        if path in seen:
            continue
        seen.add(path)
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as error:
            raise ResourceError(f"无法读取项目指令 {path}: {error}") from error
        files.append(ProjectContextFile(path=path, content=content))
    return tuple(files)
