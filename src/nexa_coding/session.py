"""编码会话：把持久化会话层接到 agent 运行上。

CodingSession 是"可续聊的 coding agent"入口：
- 新建：读空账本 → 建 harness → 记 SessionInfoEntry + ModelChangeEntry
- 恢复：读账本 → 重放 SessionState → 用恢复的消息建 harness
- prompt / continue_：喂 harness 跑一轮，监听完整消息并立即追加进账本
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Callable
from contextlib import aclosing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, cast

from nexa_agent.events import AgentEndEvent, AgentEvent, MessageEndEvent
from nexa_agent.harness import AgentHarness, AgentHarnessConfig
from nexa_agent.messages import AgentMessage, AssistantMessage, TokenUsage, ToolResultMessage
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
        resolve_provider: 按账本中的供应商名字恢复配置，由应用提供。
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
    resolve_provider: Callable[[str], ModelProvider] | None = None


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
        config: CodingSessionConfig,
        provider: ModelProvider,
        storage: SessionStorage,
        harness: AgentHarness,
        entries: list[Entry],
        active_leaf_id: str,
        skills: list[Skill] | None = None,
    ) -> None:
        """由 load() 内部调用，一般不要直接构造。"""
        self._config = config
        self._provider = provider
        self._thinking: ThinkingLevel = "default"
        self.thinking_notice: str | None = None
        self._storage = storage
        self._harness = harness
        # 全部分支留在账本里，Harness 只持有当前分支的消息。
        self._entries = entries
        self._active_leaf_id = active_leaf_id
        self._save_failed = False
        self._harness.subscribe(self)
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

        provider, model, thinking = cls._model_settings(config, state, model_override)
        requested = state.thinking if state.model else config.thinking

        # 空会话：先落一条会话信息 + 一条模型记录。
        if not entries:
            session_info = SessionInfoEntry(id="info", parent_id=None, cwd=str(cwd))
            model_change = ModelChangeEntry(
                id="model",
                parent_id="info",
                model=model,
                provider=getattr(provider, "name", None),
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
            ),
            messages=state.messages,
        )

        session = cls(
            config=config,
            provider=provider,
            storage=storage,
            harness=harness,
            entries=entries,
            active_leaf_id=state.active_leaf_id or entries[-1].id,
            skills=skills,
        )
        session._thinking = thinking
        session.thinking_notice = (
            f"模型 {model} 不支持 {requested}，思考已切换为 {thinking}"
            if state.model and thinking != requested
            else None
        )
        return session

    @staticmethod
    def _model_settings(
        config: CodingSessionConfig, state: SessionState, model_override: str | None = None
    ) -> tuple[ModelProvider, str, ThinkingLevel]:
        provider = config.provider
        if (
            model_override is None
            and state.provider
            and state.provider != getattr(provider, "name", None)
        ):
            if config.resolve_provider is None:
                raise ValueError(f"无法恢复供应商 {state.provider}，请提供 resolve_provider")
            provider = config.resolve_provider(state.provider)
        model = model_override or state.model or config.model
        requested = state.thinking if state.model else config.thinking
        thinking = (
            requested
            if requested in provider.thinking_options(model)
            else provider.default_thinking
        )
        return provider.with_thinking(model, thinking), model, thinking

    def _require_idle(self) -> None:
        if self._harness.is_running:
            raise RuntimeError("运行中不能修改会话，请先结束当前任务")
        if self._save_failed:
            raise RuntimeError("会话保存失败，请先解决存储问题，再用 /resume 重新加载会话")

    @property
    def active_leaf_id(self) -> str | None:
        return self._active_leaf_id

    @staticmethod
    def _pending_tools(messages: tuple[AgentMessage, ...] | list[AgentMessage]) -> set[str]:
        pending: set[str] = set()
        for message in messages:
            if isinstance(message, AssistantMessage):
                pending.update(call.id for call in message.tool_calls)
            elif isinstance(message, ToolResultMessage):
                pending.discard(message.tool_call_id)
        return pending

    def branch_choices(self) -> list[tuple[str, str]]:
        """显示所有分支上的安全消息节点；工具调用组中间不能继续对话。"""
        # shortcut: 小会话逐节点回放；长会话选择变慢时再缓存路径状态。
        choices = []
        for entry in self._entries:
            if isinstance(entry, MessageEntry):
                state = SessionState.from_entries(self._entries, leaf_id=entry.id)
                if not self._pending_tools(state.messages):
                    preview = " ".join(entry.message.text.split())[:70] or "工具结果"
                    indent = "  " * (len(state.messages) - 1)
                    marker = " ← 当前" if entry.id == self._active_leaf_id else ""
                    choices.append(
                        (
                            entry.id,
                            f"{indent}• {entry.id} · {entry.message.role} · {preview}{marker}",
                        )
                    )
        return choices

    def branch_to_entry(self, entry_id: str) -> None:
        """只改变当前路径，旧分支留在账本中；后续消息接在选中节点后。"""
        self._require_idle()
        if entry_id not in {key for key, _ in self.branch_choices()}:
            raise ValueError("请选择完整消息节点；工具调用必须已有全部结果")
        state = SessionState.from_entries(self._entries, leaf_id=entry_id)
        provider, model, thinking = self._model_settings(self._config, state)
        self._append_entry(
            LeafEntry(id=self._new_id("leaf"), parent_id=entry_id, target_id=entry_id)
        )
        self._active_leaf_id = entry_id
        self._harness.select_model(provider, model)
        self._harness.replace_messages(state.messages)
        self._provider, self._thinking = provider, thinking

    # ── 对外 API ─────────────────────────────────────────────────────────────

    def select_model(self, provider: ModelProvider, model: str, provider_name: str) -> bool:
        """账本写入成功之后，才发布新的运行配置。"""
        self._require_idle()
        options = provider.thinking_options(model)
        thinking = self.thinking if self.thinking in options else provider.default_thinking
        reset = thinking != self.thinking
        provider = provider.with_thinking(model, thinking)
        entry = ModelChangeEntry(
            id=self._new_id("model"),
            parent_id=self._active_leaf_id,
            model=model,
            provider=provider_name,
            thinking=thinking,
        )
        self._append_entry(entry)
        self._harness.select_model(provider, model)
        self._provider = provider
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
        self._require_idle()
        if level == self.thinking:
            return
        setting = cast(ThinkingLevel, level)
        candidate = self._provider.with_thinking(self.model, setting)
        entry = ThinkingChangeEntry(
            id=self._new_id("thinking"),
            parent_id=self._active_leaf_id,
            thinking=setting,
        )
        self._append_entry(entry)
        self._harness.select_model(candidate, self.model)
        self._provider = candidate
        self._thinking = setting

    @property
    def model(self) -> str:
        """本次会话实际使用的模型（包括账本恢复的选择）。"""
        return self._harness.model

    @property
    def latest_usage(self) -> TokenUsage | None:
        """查找当前模型最近一次完整响应的实测值，恢复时复用消息账本。"""
        return next(
            (
                message.usage
                for message in reversed(self.messages)
                if isinstance(message, AssistantMessage)
                and message.usage is not None
                and message.usage.model == self.model
            ),
            None,
        )

    @property
    def messages(self) -> tuple[AgentMessage, ...]:
        """当前会话历史（只读）。"""
        return self._harness.messages

    async def prompt(self, text: str) -> AsyncGenerator[AgentEvent, None]:
        """展开技能后运行；完整消息由监听器保存，不依赖界面消费到结尾。"""
        self._require_idle()
        if self._pending_tools(self.messages):
            raise ValueError("上次工具调用未完成，请用 /tree 回到完整的消息节点再继续")
        expanded = expand_skill_command(text, self.skills)
        resolved = expanded if expanded is not None else text
        async with aclosing(self._harness.prompt(resolved)) as stream:
            async for event in stream:
                yield event

    async def continue_(self) -> AsyncGenerator[AgentEvent, None]:
        self._require_idle()
        if self._pending_tools(self.messages):
            raise ValueError("上次工具调用未完成，请用 /tree 回到完整的消息节点再继续")
        async with aclosing(self._harness.continue_()) as stream:
            async for event in stream:
                yield event

    async def on_event(self, event: AgentEvent) -> None:
        """Harness 在 yield 前等待这里：只存完整消息，不存流式片段。"""
        if isinstance(event, MessageEndEvent):
            self._append_entry(
                MessageEntry(
                    id=self._new_id("message"),
                    parent_id=self._active_leaf_id,
                    message=event.message,
                )
            )
        elif isinstance(event, AgentEndEvent):
            self._append_entry(
                LeafEntry(
                    id=self._new_id("leaf"),
                    parent_id=self._active_leaf_id,
                    target_id=self._active_leaf_id,
                )
            )

    def _append_entry(self, entry: Entry) -> None:
        # 保存失败后阻止继续写入，避免下一轮跳过未保存的消息。
        try:
            self._storage.append(entry)
        except Exception:
            if isinstance(entry, MessageEntry):
                self._save_failed = True
            raise
        self._entries.append(entry)
        if not isinstance(entry, LeafEntry):
            self._active_leaf_id = entry.id

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
