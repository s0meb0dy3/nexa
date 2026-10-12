"""会话、模型选择窗口和会话树。"""

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.screen import ModalScreen
from textual.widgets import Input, OptionList, Static
from textual.widgets.option_list import Option

from nexa_coding.session import BranchNode


class ChoiceScreen(ModalScreen[str | None]):
    BINDINGS = [Binding("escape", "dismiss(None)", "返回")]
    DEFAULT_CSS = """
    ChoiceScreen { align: center middle; background: #00000080; }
    #choice-panel { width: 85%; max-width: 100; height: 70%; padding: 1 2;
        border: solid #79d6cf; background: #142026; }
    #choice-title { height: 2; color: #79d6cf; }
    #choice-search { height: 3; }
    #choice-list { height: 1fr; border: none; }
    """

    def __init__(self, title: str, choices: list[tuple[str, str]]) -> None:
        super().__init__()
        self._title = title
        self._choices = choices

    def compose(self) -> ComposeResult:
        from textual.containers import Vertical

        with Vertical(id="choice-panel"):
            yield Static(self._title, id="choice-title", markup=False)
            yield Input(placeholder="搜索…（Tab 切换到列表，Esc 返回）", id="choice-search")
            yield OptionList(id="choice-list")

    def on_mount(self) -> None:
        self._filter("")
        self.query_one(OptionList).focus()

    def _filter(self, query: str) -> None:
        listing = self.query_one(OptionList)
        listing.clear_options()
        listing.add_options(
            [
                Option(Text(label), id=key)
                for key, label in self._choices
                if query.casefold() in label.casefold()
            ]
        )
        if listing.option_count:
            listing.highlighted = 0

    def on_input_changed(self, event: Input.Changed) -> None:
        self._filter(event.value)

    def on_input_submitted(self) -> None:
        listing = self.query_one(OptionList)
        if listing.highlighted is not None:
            self.dismiss(listing.get_option_at_index(listing.highlighted).id)

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        self.dismiss(event.option.id)


class TreeScreen(ChoiceScreen):
    """只在真正分叉时缩进；问题和工具摘要用于阅读，完整回答用于跳转。"""

    DEFAULT_CSS = """
    TreeScreen { align: center middle; background: #00000080; }
    #tree-panel { width: 92%; height: 82%; padding: 1 2;
        border: solid #38565e; background: #0d1418; }
    #tree-title { height: 1; margin-bottom: 1; color: #79d6cf; }
    #tree-search { height: 1; border: none; padding: 0; margin-bottom: 1;
        background: transparent; }
    #tree-list { height: 1fr; border: none; padding: 0; background: transparent;
        text-wrap: nowrap; text-overflow: ellipsis; }
    #tree-list > .option-list--option { padding: 0; }
    #tree-list > .option-list--option-highlighted { background: #223a42; }
    #tree-list > .option-list--option-disabled { color: #8fa5b2; }
    #tree-help { height: 1; margin-top: 1; color: #8fa5b2; }
    """

    def __init__(self, nodes: list[BranchNode]) -> None:
        super().__init__("会话树", [])
        self._rows: list[tuple[str, BranchNode]] = []
        self._questions: dict[str, str] = {}
        children: dict[str | None, list[BranchNode]] = {}
        for node in nodes:
            children.setdefault(node.parent_id, []).append(node)

        stack: list[tuple[BranchNode, str, str, str]] = []

        def push_children(parent: str | None, prefix: str, question: str) -> None:
            siblings = children.get(parent, [])
            # 栈后进先出，倒序放入以保持原来的深度优先显示顺序。
            for index in reversed(range(len(siblings))):
                node = siblings[index]
                last = index == len(siblings) - 1
                fork = len(siblings) > 1
                edge = ("└─ " if last else "├─ ") if fork else ""
                # 单链不增加缩进，分叉才保留连接线和分支层级。
                continuation = ("   " if last else "│  ") if fork else ""
                stack.append((node, prefix + edge, prefix + continuation, question))

        push_children(None, "", "")
        while stack:
            node, label_prefix, child_prefix, question = stack.pop()
            self._rows.append((label_prefix, node))
            node_question = node.text if node.role == "user" else question
            self._questions[node.id] = node_question
            push_children(node.id, child_prefix, node_question)

    def compose(self) -> ComposeResult:
        from textual.containers import Vertical

        with Vertical(id="tree-panel"):
            yield Static("会话树", id="tree-title")
            yield Input(placeholder="搜索对话…", id="tree-search")
            yield OptionList(id="tree-list", compact=True)
            yield Static("↑↓ 选择回答 · Enter 从此处继续 · Esc 返回 · 保留其他分支", id="tree-help")

    def _filter(self, query: str) -> None:
        listing = self.query_one(OptionList)
        listing.clear_options()
        selected = None
        for prefix, node in self._rows:
            # 按问题搜索时，同时保留对应回答和工具摘要。
            if query.casefold() not in f"{self._questions[node.id]} {node.text}".casefold():
                continue
            label = Text(prefix, style="#627580", no_wrap=True, overflow="ellipsis")
            # 标记放在正文前，长回答被截断后仍能看见当前位置。
            if node.current:
                label.append("● 当前 · ", style="#79d6cf")
            if node.role == "tools":
                label.append(f"[{node.text}]", style="#627580")
            else:
                role = "你" if node.role == "user" else "助手"
                color = "#79d6cf" if node.role == "user" else "#d8e2e7"
                label.append(f"{role}：{node.text}", style=color)
            listing.add_option(Option(label, id=node.id, disabled=not node.selectable))
            if node.selectable and node.current:
                selected = listing.option_count - 1
        listing.highlighted = (
            selected
            if selected is not None
            else next(
                (
                    i
                    for i in range(listing.option_count)
                    if not listing.get_option_at_index(i).disabled
                ),
                None,
            )
        )
        self.call_after_refresh(listing.scroll_to_highlight)

    def on_input_submitted(self) -> None:
        listing = self.query_one(OptionList)
        if listing.highlighted is not None:
            option = listing.get_option_at_index(listing.highlighted)
            if not option.disabled:
                self.dismiss(option.id)
