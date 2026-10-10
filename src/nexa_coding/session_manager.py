"""项目内的会话文件与当前选择。小规模会话直接扫描，无需额外索引。"""

from __future__ import annotations

import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from nexa_agent.messages import UserMessage
from nexa_agent.provider import ModelProvider
from nexa_agent.session.entries import MessageEntry
from nexa_agent.session.storage import JsonlStorage
from nexa_coding.config import load_config
from nexa_coding.paths import NexaPaths
from nexa_coding.providers import build_provider


@dataclass(frozen=True)
class SessionSummary:
    id: str
    title: str
    messages: int
    updated_at: float


class SessionManager:
    def __init__(self, directory: Path, *, config_file: Path | None = None) -> None:
        self.directory = directory
        self._config_file = config_file or NexaPaths().config_file

    def resolve_provider(self, name: str) -> ModelProvider:
        """恢复旧分支的供应商配置，TUI 不再解析消息账本。"""
        config = load_config(self._config_file)
        if name not in config.providers:
            raise ValueError(f"会话使用的供应商 {name} 不在配置中")
        return build_provider(config.providers[name])

    def path_for(self, session_id: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", session_id):
            raise ValueError("会话 ID 格式无效")
        path = self.directory / f"{session_id}.jsonl"
        if not path.is_file():
            raise ValueError(f"找不到会话 {session_id}")
        return path

    def current_path(self) -> Path:
        try:
            return self.path_for((self.directory / "current.txt").read_text().strip())
        except (OSError, ValueError):
            return self.directory / "default.jsonl"

    def new_path(self) -> Path:
        return self.directory / f"{uuid4().hex}.jsonl"

    def select(self, path: Path) -> None:
        self.path_for(path.stem)
        self.directory.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", dir=self.directory, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(path.stem + "\n")
        try:
            temporary.replace(self.directory / "current.txt")
        finally:
            temporary.unlink(missing_ok=True)

    def list_sessions(self) -> list[SessionSummary]:
        # ponytail: 直接扫描账本；会话数量或体积大时再加索引。
        sessions = []
        for path in self.directory.glob("*.jsonl"):
            entries = JsonlStorage(path).read_all()
            messages = [e.message for e in entries if isinstance(e, MessageEntry)]
            title = next((m.text for m in messages if isinstance(m, UserMessage)), "空会话")
            title = " ".join(title.split())[:60]
            sessions.append(SessionSummary(path.stem, title, len(messages), path.stat().st_mtime))
        return sorted(sessions, key=lambda s: s.updated_at, reverse=True)
