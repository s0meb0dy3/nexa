"""NEXA 所有用户数据位置的唯一真相源。

凡是"文件该放哪/去哪找"的问题，都问 NexaPaths，不要自己拼路径——
单一真相源能避免各处硬编码漂移。分两级：
- 用户级：集中在 ~/.nexa（和 ~/.agents，兼容通用 agent 技能目录）
- 项目级：跟 cwd 走（<cwd>/.nexa、<cwd>/.agents），项目覆盖用户
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path


@dataclass(frozen=True)
class NexaPaths:
    """用户数据位置的 canonical 定义。"""

    # 用户级家目录。
    home: Path = Path.home() / ".nexa"
    # 通用 agent 资源目录（Tau 约定：.agents 是正常会话环境的一部分）。
    agents_home: Path = Path.home() / ".agents"

    # ── 用户级路径 ────────────────────────────────────────────────────────

    @property
    def config_file(self) -> Path:
        """用户配置文件：供应商档案、默认供应商等。"""
        return self.home / "config.toml"

    @property
    def sessions_dir(self) -> Path:
        """所有项目的会话账本集中存放在家目录。"""
        return self.home / "sessions"

    @property
    def user_skills_dir(self) -> Path:
        """用户级技能目录。"""
        return self.home / "skills"

    @property
    def user_prompts_dir(self) -> Path:
        """用户级提示模板目录。"""
        return self.home / "prompts"

    @property
    def user_agents_skills_dir(self) -> Path:
        """~/.agents/skills：跨项目共享的通用技能。"""
        return self.agents_home / "skills"

    @property
    def user_agents_prompts_dir(self) -> Path:
        """~/.agents/prompts：跨项目共享的通用模板。"""
        return self.agents_home / "prompts"

    # ── 项目级路径 ────────────────────────────────────────────────────────

    def project_skills_dir(self, cwd: Path) -> Path:
        """项目自己的技能目录（优先级最高，可覆盖用户级）。"""
        return cwd / ".nexa" / "skills"

    def project_agents_skills_dir(self, cwd: Path) -> Path:
        """项目里 .agents/skills（团队共享，优先级最高）。"""
        return cwd / ".agents" / "skills"

    def project_prompts_dir(self, cwd: Path) -> Path:
        """项目自己的提示模板目录。"""
        return cwd / ".nexa" / "prompts"

    def project_agents_prompts_dir(self, cwd: Path) -> Path:
        """项目里 .agents/prompts。"""
        return cwd / ".agents" / "prompts"

    def project_session_dir(self, cwd: Path) -> Path:
        """某个项目的会话目录：家目录集中 + 按项目隔离。

        目录用"可读名-短哈希"命名：可读名来自路径清洗，短哈希保证
        不同项目绝不重名（即使清洗后同名）。
        """
        name = f"{_slugify_path(cwd)}-{sha256(str(cwd).encode()).hexdigest()[:8]}"
        return self.sessions_dir / name

    def default_session_path(self, cwd: Path) -> Path:
        """某个项目的默认会话账本文件。"""
        return self.project_session_dir(cwd) / "default.jsonl"


def _slugify_path(cwd: Path) -> str:
    """把路径清洗成可读的目录名：非字母数字转 -，压缩连续 -，去首尾。"""

    slug = "".join(ch if ch.isalnum() else "-" for ch in str(cwd))
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-")


__all__ = ["NexaPaths"]
