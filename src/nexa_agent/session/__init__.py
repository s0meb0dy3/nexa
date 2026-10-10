"""nexa_agent.session: 会话的持久化与回放。

把一次会话的所有状态（消息、模型切换、标签）以 append-only 的 JSONL
账本形式落盘，之后再按需回放成 SessionState。五个子模块各管一件事：
- entries:  账本里每一条记录（entry）的模型
- jsonl:    条目 <-> 一行 JSON 的序列化
- storage:  文件的追加与读取
- tree:     按 parent_id 构成的树里找路径
- memory:   把条目回放成可用的会话状态
"""

from nexa_agent.session.entries import (
    BaseEntry,
    Entry,
    LabelEntry,
    LeafEntry,
    MessageEntry,
    ModelChangeEntry,
    SessionInfoEntry,
    ThinkingChangeEntry,
)
from nexa_agent.session.jsonl import entry_from_line, entry_to_line
from nexa_agent.session.memory import SessionState
from nexa_agent.session.storage import JsonlStorage
from nexa_agent.session.tree import path_to_entry

__all__ = [
    "BaseEntry",
    "Entry",
    "JsonlStorage",
    "LabelEntry",
    "LeafEntry",
    "MessageEntry",
    "ModelChangeEntry",
    "SessionInfoEntry",
    "ThinkingChangeEntry",
    "SessionState",
    "entry_from_line",
    "entry_to_line",
    "path_to_entry",
]
