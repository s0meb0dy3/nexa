"""按 parent_id 构成的树里找路径。

条目用 parent_id 串成树：每条记录指向前一条。给定一个叶子的 id，
path_to_entry 会从它一路沿 parent_id 走回根，再反转成"根 -> 叶子"的顺序。
"""

from __future__ import annotations

from nexa_agent.session.entries import Entry


class TreeError(ValueError):
    """树遍历出错（找不到节点或父节点悬空）时抛出。"""


def path_to_entry(entries: list[Entry], leaf_id: str) -> list[Entry]:
    """从叶子沿 parent_id 走回根，返回"根 -> 叶子"顺序的路径。

    Args:
        entries: 全部条目（顺序无关，内部会建索引）。
        leaf_id: 起点的叶子条目 id。

    Returns:
        从根到叶子的条目列表，包含叶子本身。

    Raises:
        TreeError: leaf_id 不存在，或某条记录的 parent_id 找不到对应节点
            （即树不完整 / 悬空）时抛出。
    """

    # 先建 id -> entry 的索引，方便 O(1) 查找每个节点。
    by_id: dict[str, Entry] = {entry.id: entry for entry in entries}

    if leaf_id not in by_id:
        raise TreeError(f"找不到 id 为 {leaf_id!r} 的条目")

    # 从叶子开始，沿 parent_id 一路向上收集；根条目的 parent_id 是 None。
    path: list[Entry] = []
    seen: set[str] = set()
    current_id: str | None = leaf_id
    while current_id is not None:
        if current_id in seen:
            raise TreeError(f"条目 {current_id!r} 的父节点形成循环")
        seen.add(current_id)
        node = by_id.get(current_id)
        if node is None:
            # 某条记录的 parent_id 指向了一个不存在的节点，树断了。
            raise TreeError(f"条目 {current_id!r} 的父节点不存在")
        path.append(node)
        current_id = node.parent_id

    # 上面收集的顺序是"叶子 -> 根"，反转成"根 -> 叶子"。
    path.reverse()
    return path


__all__ = ["TreeError", "path_to_entry"]
