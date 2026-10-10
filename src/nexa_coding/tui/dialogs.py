"""会话和模型共用的简单搜索选择窗口。"""

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.screen import ModalScreen
from textual.widgets import Input, OptionList, Static
from textual.widgets.option_list import Option


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
