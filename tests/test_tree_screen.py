"""会话树的分叉、单行显示与选择行为。"""

from textual.app import App
from textual.widgets import Input, OptionList

from nexa_coding.session import BranchNode
from nexa_coding.tui.dialogs import TreeScreen


async def test_tree_layout_search_and_selection():
    nodes = [
        BranchNode("q1", None, "user", "介绍项目"),
        BranchNode("a1", "q1", "assistant", "项目说明", True),
        BranchNode("q2", "a1", "user", "原来的问题"),
        BranchNode("tools", "q2", "tools", "执行了 8 次工具调用"),
        BranchNode("a2", "tools", "assistant", "长回答" * 200, True, True),
        BranchNode("q3", "a1", "user", "另一个问题"),
        BranchNode("a3", "q3", "assistant", "分支回答", True),
    ]
    screen = TreeScreen(nodes)
    assert [prefix for prefix, _ in screen._rows] == ["", "", "├─ ", "│  ", "│  ", "└─ ", "   "]
    selected = []
    app = App()
    async with app.run_test(size=(100, 30)) as pilot:
        app.push_screen(screen, selected.append)
        await pilot.pause()
        listing = screen.query_one(OptionList)
        assert listing.get_option_at_index(listing.highlighted).id == "a2"
        assert listing.virtual_size.height == len(nodes)
        assert "● 当前" in listing.get_option("a2").prompt.plain
        await pilot.resize_terminal(60, 20)
        await pilot.pause()
        assert listing.virtual_size.height == len(nodes)
        search = screen.query_one(Input)
        search.value = "原来的问题"
        await pilot.pause()
        assert listing.option_count == 3
        assert listing.get_option_at_index(listing.highlighted).id == "a2"
        search.value = "没有匹配"
        await pilot.pause()
        assert listing.highlighted is None
        search.value = "介绍项目"
        await pilot.pause()
        assert listing.option_count == 2
        listing.focus()
        await pilot.press("enter")
        await pilot.pause()
        assert selected == ["a1"]


async def test_tree_escape_preserves_location():
    app = App()
    selected = []
    async with app.run_test() as pilot:
        app.push_screen(
            TreeScreen([BranchNode("a", None, "assistant", "回答", True, True)]), selected.append
        )
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
        assert selected == [None]
