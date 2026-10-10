"""编码会话：把持久化会话层接到 agent 运行上。

CodingSession 是"可续聊的 coding agent"入口：
- 新建：读空账本 → 建 harness → 记 SessionInfoEntry + ModelChangeEntry
- 恢复：读账本 → 重放 SessionState → 用恢复的消息建 harness
- prompt / continue_：喂 harness 跑一轮，跑完后把新消息追加进账本
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, cast

from nexa_agent.events import AgentEvent
from nexa_agent.harness import AgentHarness, AgentHarnessConfig
from nexa_agent.messages import AgentMessage
from nexa_agent.provider import ModelProvider, ThinkingLevel
from nexa_agent.session.entries import (
    Entry,
    LeafEntry,
    MessageEntry,
    ModelChangeEntry,
    SessionInfoEntry,
    ThinkingChangeEntry,
)
from nexa_agent.session.memory import SessionState
from nexa_agent.session.storage import JsonlStorage
from nexa_coding.paths import NexaPaths
from nexa_coding.resources import NexaResourcePaths
from nexa_coding.skills import Skill, expand_skill_command, load_skills
from nexa_coding.system_prompt import BuildSystemPromptOptions, build_system_prompt
from nexa_coding.tools import create_coding_tools

DEFAULT_MODEL = "deepseek-flash"


# ── 存储接口 ─────────────────────────────────────────────────────────────────


class SessionStorage(Protocol):
    """session 层需要的存储能力。

    JsonlStorage 已实现，测试里可用内存实现替换（避免碰磁盘）。
    """

    def append(self, entry: Entry) -> None: ...

    def read_all(self) -> list[Entry]: ...


# ── 配置 ─────────────────────────────────────────────────────────────────────


@dataclass
class CodingSessionConfig:
    """创建 CodingSession 所需的最小配置。

    Attributes:
        provider: 模型 Provider。
        model: 模型名。
        system: 系统提示词；为 None 时由 build_system_prompt 自动构建。
        storage: 会话账本存储；为 None 时按项目自动落到
            ~/.nexa/sessions/<项目>-<hash>/default.jsonl（见 NexaPaths）。
        cwd: 工具可访问的项目目录，也决定会话落盘位置和项目资源发现。
        skills: 已加载的技能列表，prompt 里可触发 /skill:name。
        resource_paths: 资源发现配置；为 None 时按 cwd 自动构造。
    """

    provider: ModelProvider
    model: str = DEFAULT_MODEL
    thinking: ThinkingLevel = "default"
    system: str | None = None
    # 为 None 时在 load() 里按 config.cwd 算项目隔离路径（default_factory 拿不到 cwd）。
    storage: SessionStorage | None = None
    cwd: str | Path = Path.cwd()
    skills: list[Skill] = field(default_factory=list)
    resource_paths: NexaResourcePaths | None = None


# ── 会话主类 ─────────────────────────────────────────────────────────────────


class CodingSession:
    """可续聊的持久化编码会话。

    用法：
        session = CodingSession.load(config)
        async for event in session.prompt("读取 README.md"):
            print(event)
        # 重启后
        session2 = CodingSession.load(config)  # 从账本恢复
        async for event in session2.continue_():
            print(event)
    """

    def __init__(
        self,
        *,
        provider: ModelProvider,
        model: str,
        system: str,
        storage: SessionStorage,
        cwd: Path,
        harness: AgentHarness,
        entries: list[Entry],
        skills: list[Skill] | None = None,
    ) -> None:
        """由 load() 内部调用，一般不要直接构造。"""
        self._provider = provider
        self._thinking: ThinkingLevel = "default"
        self.thinking_notice: str | None = None
        self._model = model
        self._system = system
        self._storage = storage
        self._cwd = cwd
        self._harness = harness
        # 账本当前全部条目（用于生成下一个 id 和找上一条）。
        self._entries = entries
        # 自增计数器，保证每个条目 id 唯一。
        self._id_counter = self._next_id_from_entries(self._entries)
        # 已加载的技能，prompt 里可触发 /skill:name。
        self.skills: list[Skill] = skills or []

    # ── 类方法：加载 / 新建 ──────────────────────────────────────────────────

    @classmethod
    def load(
        cls, config: CodingSessionConfig, *, model_override: str | None = None
    ) -> CodingSession:
        """加载（或新建）一个会话。

        读账本 → 重放 SessionState → 用恢复的消息建 harness → 注册工具。
        空会话则追加 SessionInfoEntry + ModelChangeEntry。
        model_override 用于 /model 显式选择新模型，避免旧模型阻止会话加载。
        """
        cwd = Path(config.cwd).resolve()

        # 存储：显式传入的优先；否则按项目隔离落到 ~/.nexa/sessions/<项目>-<hash>/。
        storage = config.storage or JsonlStorage(NexaPaths().default_session_path(cwd))

        entries = storage.read_all()
        state = SessionState.from_entries(entries)

        model = model_override or state.model or config.model
        requested = state.thinking if state.model else config.thinking
        options = config.provider.thinking_options(model)
        thinking = requested if requested in options else config.provider.default_thinking
        provider = config.provider.with_thinking(model, thinking)

        # 空会话：先落一条会话信息 + 一条模型记录。
        if not entries:
            session_info = SessionInfoEntry(id="info", parent_id=None, cwd=str(cwd))
            model_change = ModelChangeEntry(
                id="model",
                parent_id="info",
                model=config.model,
                provider=getattr(config.provider, "name", None),
                thinking=thinking,
            )
            storage.append(session_info)
            storage.append(model_change)
            entries = [session_info, model_change]

        # 加载技能：config.skills 优先，否则从资源目录扫描（用户 + 项目四级来源）。
        # 注意顺序：先加载技能，再拼系统提示词（技能要进 prompt）。
        resource_paths = config.resource_paths or NexaResourcePaths(cwd=cwd)
        skills = config.skills or load_skills(resource_paths)

        # 系统提示词：没给就用 build_system_prompt 自动构建。
        tools = create_coding_tools(cwd)
        system = config.system or build_system_prompt(
            BuildSystemPromptOptions(cwd=cwd, tools=tools, skills=skills)
        )

        # 用恢复的消息建 harness；空会话则从空历史开始。
        harness = AgentHarness(
            AgentHarnessConfig(
                provider=provider,
                model=model,
                system=system,
                tools=tools,
            )
        )
        harness.replace_messages(state.messages)

        session = cls(
            provider=provider,
            model=model,
            system=system,
            storage=storage,
            cwd=cwd,
            harness=harness,
            entries=entries,
            skills=skills,
        )
        session._thinking = thinking
        session.thinking_notice = (
            f"模型 {model} 不支持 {requested}，思考已切换为 {thinking}"
            if state.model and thinking != requested
            else None
        )
        return session

    # ── 对外 API ─────────────────────────────────────────────────────────────

    def select_model(self, provider: ModelProvider, model: str, provider_name: str) -> bool:
        """账本写入成功之后，才发布新的运行配置。"""
        if self._harness.is_running:
            raise RuntimeError("运行中不能切换模型")
        options = provider.thinking_options(model)
        thinking = self.thinking if self.thinking in options else provider.default_thinking
        reset = thinking != self.thinking
        provider = provider.with_thinking(model, thinking)
        entry = ModelChangeEntry(
            id=self._new_id("model"),
            parent_id=self._entries[-1].id if self._entries else None,
            model=model,
            provider=provider_name,
            thinking=thinking,
        )
        self._storage.append(entry)
        self._entries.append(entry)
        self._harness.select_model(provider, model)
        self._provider = provider
        self._model = model
        self._thinking = thinking
        return reset

    @property
    def provider(self) -> ModelProvider:
        return self._provider

    @property
    def thinking(self) -> ThinkingLevel:
        return self._thinking

    @property
    def thinking_options(self) -> tuple[ThinkingLevel, ...]:
        return self._provider.thinking_options(self.model)

    def set_thinking(self, level: str) -> None:
        if self._harness.is_running:
            raise RuntimeError("运行中不能修改思考设置")
        if level == self.thinking:
            return
        setting = cast(ThinkingLevel, level)
        candidate = self._provider.with_thinking(self.model, setting)
        entry = ThinkingChangeEntry(
            id=self._new_id("thinking"),
            parent_id=self._entries[-1].id if self._entries else None,
            thinking=setting,
        )
        self._storage.append(entry)
        self._entries.append(entry)
        self._harness.select_model(candidate, self.model)
        self._provider = candidate
        self._thinking = setting

    @property
    def model(self) -> str:
        """本次会话实际使用的模型（包括账本恢复的选择）。"""
        return self._harness.model

    @property
    def messages(self) -> tuple[AgentMessage, ...]:
        """当前会话历史（只读）。"""
        return self._harness.messages

    async def prompt(self, text: str) -> AsyncIterator[AgentEvent]:
        """发送用户消息，跑一轮，yield 事件；跑完后把新消息落盘。

        支持技能命令：输入 /skill:name 参数 时，先展开成技能文本再喂给模型。

        关键设计：持久化在 run 完成后做（非逐消息），避免和 harness
        内存里的 transcript 双重保存。
        """
        # 命中技能命令就把输入替换成展开后的技能文本，否则用原文。
        expanded = expand_skill_command(text, self.skills)
        resolved = expanded if expanded is not None else text

        before = len(self._harness.messages)
        async for event in self._harness.prompt(resolved):
            yield event
        # 跑完：把这一轮新增的消息追加成 MessageEntry，再记一条 LeafEntry。
        self._persist_new_messages(before)

    async def continue_(self) -> AsyncIterator[AgentEvent]:
        """不追加用户消息，从当前历史继续跑；跑完后同样落盘。"""
        before = len(self._harness.messages)
        async for event in self._harness.continue_():
            yield event
        self._persist_new_messages(before)

    def handle_command(self, text: str) -> str | None:
        """兼容旧调用者；交互命令由 TUI 执行，元数据只有一份。"""
        from nexa_coding.commands import COMMANDS, parse_command

        try:
            command = parse_command(text)
        except ValueError as error:
            return str(error)
        if command is None:
            return None
        name, _ = command
        if name == "help":
            return "\n".join(f"/{key} {value}" for key, value in COMMANDS.items())
        if name == "exit":
            return "退出"
        return "请在交互界面执行此命令"

    # ── 内部实现 ─────────────────────────────────────────────────────────────

    def _persist_new_messages(self, before: int) -> None:
        """把 harness 历史里 from 位置之后的新消息追加进账本。"""
        new_messages = self._harness.messages[before:]
        if not new_messages:
            return

        last_entry_id: str | None = self._entries[-1].id if self._entries else None
        for message in new_messages:
            entry_id = self._new_id("message")
            entry = MessageEntry(id=entry_id, parent_id=last_entry_id, message=message)
            self._storage.append(entry)
            self._entries.append(entry)
            last_entry_id = entry_id

        # 叶子指针指向最新一条消息。
        leaf = LeafEntry(
            id=self._new_id("leaf"), parent_id=last_entry_id, target_id=last_entry_id or ""
        )
        self._storage.append(leaf)
        self._entries.append(leaf)

    def _new_id(self, prefix: str) -> str:
        """生成一个唯一 id，如 message-1、leaf-2。"""
        self._id_counter += 1
        return f"{prefix}-{self._id_counter}"

    @staticmethod
    def _next_id_from_entries(entries: list[Entry]) -> int:
        """从已有条目里找出当前最大编号，作为计数起点。"""
        max_num = 0
        for entry in entries:
            # 条目 id 形如 "message-3"，取数字部分。
            try:
                num = int(entry.id.rsplit("-", 1)[-1])
            except ValueError:
                continue
            max_num = max(max_num, num)
        return max_num


__all__ = ["CodingSession", "CodingSessionConfig", "SessionStorage"]
