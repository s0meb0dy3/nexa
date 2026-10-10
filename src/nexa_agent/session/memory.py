"""把会话账本回放成可用的会话状态（SessionState）。

回放就是把一条条 entry 按顺序"重演"一遍：
- MessageEntry → 追加一条消息到 messages
- ModelChangeEntry → 更新当前模型
- LabelEntry → 更新当前标签

因为账本是 append-only 的，回放后得到的就是"截至某条记录时"的最新状态。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from nexa_agent.messages import AgentMessage
from nexa_agent.provider import ThinkingLevel
from nexa_agent.session.entries import (
    Entry,
    LabelEntry,
    LeafEntry,
    MessageEntry,
    ModelChangeEntry,
    ThinkingChangeEntry,
)
from nexa_agent.session.tree import path_to_entry


@dataclass
class SessionState:
    """回放后的会话状态。

    Attributes:
        messages: 当前对话历史（直接复用 AgentMessage，可喂回 AgentLoop）。
        model: 当前使用的模型名。
        label: 会话标签，未设置时为 None。
    """

    messages: list[AgentMessage] = field(default_factory=list)
    model: str = ""
    provider: str | None = None
    thinking: ThinkingLevel = "default"
    label: str | None = None
    active_leaf_id: str | None = None

    @classmethod
    def from_entries(cls, entries: list[Entry], *, leaf_id: str | None = None) -> SessionState:
        """从账本条目回放出会话状态。

        Args:
            entries: 账本里的全部条目。
            leaf_id: 指定恢复节点；为 None 时恢复账本的当前分支。

        Returns:
            回放后的 SessionState。
        """

        # 分支选择用 LeafEntry；普通追加直接以末条记录为当前节点。
        # 旧账本里的 leaf 后若有模型设置，也按最新记录恢复。
        if entries and leaf_id is None:
            last = entries[-1]
            leaf_id = last.target_id if isinstance(last, LeafEntry) else last.id
        if leaf_id is not None:
            entries = path_to_entry(entries, leaf_id)

        state = cls(active_leaf_id=leaf_id)
        for entry in entries:
            if isinstance(entry, MessageEntry):
                # 消息条目直接取出内嵌的 AgentMessage，原样追加。
                state.messages.append(entry.message)
            elif isinstance(entry, ModelChangeEntry):
                # 模型切换条目：账本只记新状态，回放时覆盖当前模型。
                state.model = entry.model
                state.provider = entry.provider
                state.thinking = entry.thinking
            elif isinstance(entry, ThinkingChangeEntry):
                state.thinking = entry.thinking
            elif isinstance(entry, LabelEntry):
                # 标签条目：回放时覆盖当前标签。
                state.label = entry.label
        return state


__all__ = ["SessionState"]
