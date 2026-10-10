"""会话账本里的条目（entry）模型。

一条 entry 就是 JSONL 账本里的一行，记录会话中的一次"事件"。
所有条目共用 BaseEntry 的 id / parent_id / type，用 type 字段区分具体种类，
靠 Pydantic 的 discriminator 自动解析成对应的子类。
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

from nexa_agent.messages import AgentMessage, WireModel
from nexa_agent.provider import ThinkingLevel

# ── 条目基类 ─────────────────────────────────────────────────────────────────


class BaseEntry(WireModel):
    """所有会话条目的公共字段。

    Attributes:
        id: 这条条目的唯一编号。
        parent_id: 上一条条目的编号，形成一棵树；根条目为 None。
        type: 条目类型，也用作 Pydantic 的判别字段。
    """

    # id 由调用方生成（例如一条消息记录），这里不负责造号。
    id: str
    # parent_id 指向它的"前一条"，从而把条目串成树。
    parent_id: str | None = None
    # 判别字段：type="message" → MessageEntry，依此类推。
    type: Literal["message", "model_change", "label", "session_info", "leaf", "thinking_change"]


# ── 消息条目 ─────────────────────────────────────────────────────────────────


class MessageEntry(BaseEntry):
    """一次会话消息的记录。

    直接内嵌一条 AgentMessage（messages.py 里的 UserMessage /
    AssistantMessage / ToolResultMessage，按 role 判别），
    回放时原样取出即可，不需要再自造扁平字段。
    """

    type: Literal["message"] = "message"
    # 真正要持久化的消息本体。
    message: AgentMessage


# ── 模型切换条目 ─────────────────────────────────────────────────────────────


class ModelChangeEntry(BaseEntry):
    """会话中途切换模型的记录。

    append-only 账本只记录"新状态"，不记录迁移过程：
    model 是切换后的模型名，回放时用它覆盖当前值即可。
    """

    type: Literal["model_change"] = "model_change"
    # 切换后的模型名（新状态）。
    model: str
    # 可选的提供商名字，比如 "deepseek"、"openai"。
    provider: str | None = None
    thinking: ThinkingLevel = "default"


class ThinkingChangeEntry(BaseEntry):
    """会话的模型思考设置；与界面展开状态无关。"""

    type: Literal["thinking_change"] = "thinking_change"
    thinking: ThinkingLevel


# ── 标签条目 ─────────────────────────────────────────────────────────────────


class LabelEntry(BaseEntry):
    """给会话打标签的记录，例如给某次会话起个名字。"""

    type: Literal["label"] = "label"
    # 标签内容，比如 "重构 session 层"。
    label: str


# ── 会话信息条目 ─────────────────────────────────────────────────────────────


class SessionInfoEntry(BaseEntry):
    """会话初始化时的元信息快照。

    新会话的第一条记录：记住这个会话是围绕哪个目录建的。
    """

    type: Literal["session_info"] = "session_info"
    # 会话工作目录（绝对路径）。
    cwd: str


# ── 叶子指针条目 ─────────────────────────────────────────────────────────────


class LeafEntry(BaseEntry):
    """指向"当前会话最新消息"的指针。

    每次 prompt/continue 跑完后追加一条，target_id 指向最新的 MessageEntry。
    下次需要恢复时，直接读这条指针就知道从哪里续，不用沿树走到底。
    """

    type: Literal["leaf"] = "leaf"
    # 指向最新一条消息条目的 id。
    target_id: str


# ── 条目判别联合 ─────────────────────────────────────────────────────────────


# 用 type 字段区分具体是哪种条目；type="message" 就解析成 MessageEntry。
type Entry = Annotated[
    MessageEntry
    | ModelChangeEntry
    | ThinkingChangeEntry
    | LabelEntry
    | SessionInfoEntry
    | LeafEntry,
    Field(discriminator="type"),
]

__all__ = [
    "BaseEntry",
    "Entry",
    "LabelEntry",
    "LeafEntry",
    "MessageEntry",
    "ModelChangeEntry",
    "SessionInfoEntry",
    "ThinkingChangeEntry",
]
