"""编码会话：把持久化会话层接到 agent 运行上。

CodingSession 是"可续聊的 coding agent"入口：
- 新建：读空账本 → 建 harness → 记 SessionInfoEntry + ModelChangeEntry
- 恢复：读账本 → 重放 SessionState → 用恢复的消息建 harness
- prompt / continue_：喂 harness 跑一轮，监听完整消息并立即追加进账本
"""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator, Callable
from contextlib import aclosing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, TypeGuard, cast

from nexa_agent.events import AgentEndEvent, AgentEvent, MessageEndEvent
from nexa_agent.harness import AgentHarness, AgentHarnessConfig
from nexa_agent.messages import (
    AgentMessage,
    AssistantMessage,
    TokenUsage,
    ToolResultMessage,
    UserMessage,
)
from nexa_agent.provider import ModelProvider, ThinkingLevel
from nexa_agent.session.entries import (
    CompactionEntry,
    Entry,
    LeafEntry,
    MessageEntry,
    ModelChangeEntry,
    SessionInfoEntry,
    ThinkingChangeEntry,
)
from nexa_agent.session.memory import SessionState
from nexa_agent.session.storage import JsonlStorage
from nexa_agent.session.tree import path_to_entry
from nexa_coding.compaction import SUMMARY_SYSTEM, cut_point, estimate_tokens, serialize_history
from nexa_coding.context import discover_project_context
from nexa_coding.paths import NexaPaths
from nexa_coding.resources import NexaResourcePaths
from nexa_coding.skills import Skill, expand_skill_command, load_skills
from nexa_coding.system_prompt import BuildSystemPromptOptions, build_system_prompt
from nexa_coding.tools import create_coding_tools

DEFAULT_MODEL = "deepseek-flash"


@dataclass(frozen=True)
class BranchNode:
    """会话树的显示节点；只有完整助手回答允许跳转。"""

    id: str
    parent_id: str | None
    role: str
    text: str
    selectable: bool = False
    current: bool = False


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
        self._harness.context_preparer = self._prepare_context
        self.compacting = False
        self._compaction_id: str | None = None
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
        tools = create_coding_tools(
            cwd, skill_dirs=tuple(Path(skill.path).parent for skill in skills)
        )
        system = config.system or build_system_prompt(
            BuildSystemPromptOptions(
                cwd=cwd,
                tools=tools,
                skills=skills,
                context_files=discover_project_context(resource_paths),
            )
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
        session._compaction_id = state.compaction.id if state.compaction else None
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
        if self.compacting:
            raise RuntimeError("正在压缩上下文，请先等待或取消")
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

    def branch_nodes(self) -> list[BranchNode]:
        """投影出对话树：隐藏设置记录和工具正文，保留实际父子关系。"""

        def visible(entry: Entry) -> TypeGuard[MessageEntry]:
            return isinstance(entry, MessageEntry) and (
                isinstance(entry.message, UserMessage)
                or isinstance(entry.message, AssistantMessage)
                and not entry.message.tool_calls
            )

        active_path = path_to_entry(self._entries, self._active_leaf_id)
        current = next((entry.id for entry in reversed(active_path) if visible(entry)), None)
        nodes = []
        # shortcut: 小会话逐节点回放；长会话选择变慢时再缓存路径状态。
        for entry in self._entries:
            if not visible(entry):
                continue
            path = path_to_entry(self._entries, entry.id)
            previous = next((i for i in range(len(path) - 2, -1, -1) if visible(path[i])), -1)
            parent = path[previous].id if previous >= 0 else None
            message = entry.message
            selectable = isinstance(message, AssistantMessage) and not self._pending_tools(
                [item.message for item in path if isinstance(item, MessageEntry)]
            )
            tools = sum(
                len(item.message.tool_calls)
                for item in path[previous + 1 : -1]
                if isinstance(item, MessageEntry) and isinstance(item.message, AssistantMessage)
            )
            if tools:
                summary_id = entry.id + "-tools"
                nodes.append(BranchNode(summary_id, parent, "tools", f"执行了 {tools} 次工具调用"))
                parent = summary_id
            nodes.append(
                BranchNode(
                    entry.id,
                    parent,
                    message.role,
                    " ".join(message.text.split()) or "（无正文）",
                    selectable,
                    entry.id == current,
                )
            )
        return nodes

    def branch_choices(self) -> list[tuple[str, str]]:
        """界面和直接 /tree ID 使用同一条选择规则。"""
        return [(node.id, node.text) for node in self.branch_nodes() if node.selectable]

    def branch_to_entry(self, entry_id: str) -> None:
        """只改变当前路径，旧分支留在账本中；后续消息接在选中节点后。"""
        self._require_idle()
        if entry_id not in {key for key, _ in self.branch_choices()}:
            raise ValueError("请选择完整助手回答；工具调用必须已有全部结果")
        state = SessionState.from_entries(self._entries, leaf_id=entry_id)
        provider, model, thinking = self._model_settings(self._config, state)
        self._append_entry(
            LeafEntry(id=self._new_id("leaf"), parent_id=entry_id, target_id=entry_id)
        )
        self._active_leaf_id = entry_id
        self._harness.select_model(provider, model)
        self._harness.replace_messages(state.messages)
        self._provider, self._thinking = provider, thinking
        self._compaction_id = state.compaction.id if state.compaction else None

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
        path = path_to_entry(self._entries, self._active_leaf_id)
        start = next(
            (i + 1 for i in reversed(range(len(path))) if isinstance(path[i], CompactionEntry)), 0
        )
        return next(
            (
                entry.message.usage
                for entry in reversed(path[start:])
                if isinstance(entry, MessageEntry)
                and isinstance(entry.message, AssistantMessage)
                and entry.message.usage is not None
                and entry.message.usage.model == self.model
            ),
            None,
        )

    @property
    def compaction_id(self) -> str | None:
        return self._compaction_id

    def _budget(self) -> tuple[int, int]:
        # 未知上限仍允许手动压缩；已知模型留出输出与新增内容空间。
        limit = self.provider.context_window(self.model)
        reserve = min(limit // 3, max(16384, limit // 10)) if limit else 16384
        return (limit - reserve if limit else 0), (min(20000, limit // 5) if limit else 20000)

    def _fixed_tokens(self) -> int:
        definitions = [
            {"name": t.name, "description": t.description, "parameters": t.parameters}
            for t in self._harness.tools
        ]
        return estimate_tokens(
            [
                UserMessage(
                    content=self._harness.system + json.dumps(definitions, ensure_ascii=False)
                )
            ]
        )

    async def _prepare_context(self, messages: list[AgentMessage]) -> list[AgentMessage]:
        threshold, _ = self._budget()
        if not threshold:
            return messages
        used = self._fixed_tokens() + estimate_tokens(messages)
        usage = self.latest_usage
        if usage:
            index = next(
                i
                for i, message in reversed(list(enumerate(messages)))
                if isinstance(message, AssistantMessage) and message.usage is usage
            )
            used = max(used, usage.total_tokens + estimate_tokens(messages[index + 1 :]))
        if used >= threshold:
            await self._compact("")
            return list(self.messages)
        return messages

    async def compact(self, instructions: str = "") -> str:
        """主动压缩只能在空闲时执行；自动入口仅在完整工具批次后调用。"""
        self._require_idle()
        return await self._compact(instructions)

    async def _compact(self, instructions: str) -> str:
        if self._save_failed:
            raise RuntimeError("会话保存失败，请先重新加载")
        state = SessionState.from_entries(self._entries, leaf_id=self._active_leaf_id)
        threshold, keep = self._budget()
        # 重复压缩至少合并一条真实消息，不能只改写独立的旧摘要。
        cut = cut_point(state.messages, keep, minimum=2 if state.compaction else 1)
        # 不能只重新总结旧摘要，否则空间不会释放。
        if not any(state.message_ids[:cut]):
            raise ValueError("暂无新增历史可压缩")
        request = UserMessage(
            content="历史记录：\n"
            + serialize_history(state.messages[:cut])
            + "\n总结重点："
            + instructions
        )
        if (
            threshold
            and estimate_tokens([UserMessage(content=SUMMARY_SYSTEM), request]) >= threshold
        ):
            raise ValueError("待总结内容仍超过上下文预算，请减少单条输入或工具输出")
        self.compacting = True
        try:
            summary = None
            async with aclosing(
                self.provider.stream_response(
                    model=self.model, system=SUMMARY_SYSTEM, messages=[request], tools=[]
                )
            ) as stream:
                async for event in stream:
                    if event.type == "error":
                        raise RuntimeError(event.message)
                    if event.type == "response_end":
                        if event.message.tool_calls or event.finish_reason in (
                            "length",
                            "max_tokens",
                        ):
                            raise ValueError("摘要响应未完整完成")
                        summary = event.message.text.strip()
            if not summary:
                raise ValueError("模型未返回完整摘要")
            projected = [UserMessage(content="历史任务摘要：\n" + summary), *state.messages[cut:]]
            size = self._fixed_tokens() + estimate_tokens(projected)
            if (
                size >= self._fixed_tokens() + estimate_tokens(state.messages)
                or threshold
                and size >= threshold
            ):
                raise ValueError("压缩未释放足够上下文，保留原历史")
            kept_id = state.message_ids[cut]
            assert kept_id is not None
            # 摘要和边界先落盘，成功后才发布新的模型上下文。
            self._append_entry(
                CompactionEntry(
                    id=self._new_id("compact"),
                    parent_id=self._active_leaf_id,
                    summary=summary,
                    first_kept_entry_id=kept_id,
                )
            )
            self._harness.replace_messages(projected)
            self._compaction_id = self._active_leaf_id
            return summary
        finally:
            self.compacting = False

    @property
    def messages(self) -> tuple[AgentMessage, ...]:
        """当前发给模型的有效上下文（只读）。"""
        return self._harness.messages

    @property
    def transcript(self) -> tuple[AgentMessage, ...]:
        """当前分支的原始消息用于显示，不受压缩影响。"""
        return tuple(
            entry.message
            for entry in path_to_entry(self._entries, self._active_leaf_id)
            if isinstance(entry, MessageEntry)
        )

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
