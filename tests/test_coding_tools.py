"""测试 Coding Tools：read、write、edit、bash。"""

from __future__ import annotations

import pytest

from nexa_coding.tools import create_coding_tools

# ── 辅助函数 ──────────────────────────────────────────────────────────────────


def _get_tool(tools: tuple, name: str):
    """按名称获取工具。"""
    for tool in tools:
        if tool.name == name:
            return tool
    raise ValueError(f"工具 {name} 不存在")


async def _execute(tool, **kwargs):
    """执行工具并返回结果。"""
    return await tool.execute("test-call-id", kwargs)


# ── 测试：工具集合 ────────────────────────────────────────────────────────────


def test_create_coding_tools_returns_four_tools(tmp_path):
    """create_coding_tools 应该返回四个工具。"""
    tools = create_coding_tools(tmp_path)
    assert len(tools) == 4

    names = {tool.name for tool in tools}
    assert names == {"read", "write", "edit", "bash"}


# ── 测试：read 工具 ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_read_basic(tmp_path):
    """read 应该能读取文件内容。"""
    tools = create_coding_tools(tmp_path)
    read_tool = _get_tool(tools, "read")

    # 创建测试文件
    test_file = tmp_path / "test.txt"
    test_file.write_text("Hello, World!", encoding="utf-8")

    result = await _execute(read_tool, path="test.txt")
    assert "Hello, World!" in result.text


@pytest.mark.asyncio
async def test_read_offset_limit(tmp_path):
    """read 的 offset 和 limit 应该正确工作。"""
    tools = create_coding_tools(tmp_path)
    read_tool = _get_tool(tools, "read")

    # 创建多行文件
    test_file = tmp_path / "test.txt"
    lines = [f"Line {i}\n" for i in range(1, 11)]
    test_file.write_text("".join(lines), encoding="utf-8")

    # 读取第 3-5 行
    result = await _execute(read_tool, path="test.txt", offset=3, limit=3)
    assert "Line 3" in result.text
    assert "Line 4" in result.text
    assert "Line 5" in result.text
    assert "Line 1" not in result.text
    assert "Line 6" not in result.text


@pytest.mark.asyncio
async def test_read_file_not_found(tmp_path):
    """read 不存在的文件应该返回错误。"""
    tools = create_coding_tools(tmp_path)
    read_tool = _get_tool(tools, "read")

    result = await _execute(read_tool, path="not_exist.txt")
    assert "错误" in result.text
    assert "不存在" in result.text
    assert result.is_error


@pytest.mark.asyncio
async def test_read_directory_as_file(tmp_path):
    """read 目录应该返回错误。"""
    tools = create_coding_tools(tmp_path)
    read_tool = _get_tool(tools, "read")

    # 创建目录
    test_dir = tmp_path / "test_dir"
    test_dir.mkdir()

    result = await _execute(read_tool, path="test_dir")
    assert "错误" in result.text
    assert "不是文件" in result.text


@pytest.mark.asyncio
async def test_read_path_traversal(tmp_path):
    """read 应该拒绝越界路径。"""
    tools = create_coding_tools(tmp_path)
    read_tool = _get_tool(tools, "read")

    result = await _execute(read_tool, path="../../../etc/passwd")
    assert "错误" in result.text
    assert "越界" in result.text


# ── 测试：write 工具 ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_write_basic(tmp_path):
    """write 应该能写入文件。"""
    tools = create_coding_tools(tmp_path)
    write_tool = _get_tool(tools, "write")

    result = await _execute(write_tool, path="test.txt", content="Hello!")
    assert "已写入" in result.text

    # 验证文件内容
    test_file = tmp_path / "test.txt"
    assert test_file.read_text(encoding="utf-8") == "Hello!"


@pytest.mark.asyncio
async def test_write_creates_parent_dirs(tmp_path):
    """write 应该自动创建父目录。"""
    tools = create_coding_tools(tmp_path)
    write_tool = _get_tool(tools, "write")

    result = await _execute(write_tool, path="a/b/c/test.txt", content="Nested!")
    assert "已写入" in result.text

    # 验证文件和目录
    test_file = tmp_path / "a" / "b" / "c" / "test.txt"
    assert test_file.exists()
    assert test_file.read_text(encoding="utf-8") == "Nested!"


# ── 测试：edit 工具 ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_edit_basic(tmp_path):
    """edit 应该能替换文件内容。"""
    tools = create_coding_tools(tmp_path)
    edit_tool = _get_tool(tools, "edit")

    # 创建测试文件
    test_file = tmp_path / "test.txt"
    test_file.write_text("Hello, World!", encoding="utf-8")

    result = await _execute(
        edit_tool,
        path="test.txt",
        edits=[{"oldText": "World", "newText": "Python"}],
    )
    assert "已应用" in result.text

    # 验证文件内容
    assert test_file.read_text(encoding="utf-8") == "Hello, Python!"


@pytest.mark.asyncio
async def test_edit_multiple_edits(tmp_path):
    """edit 应该能处理多个替换。"""
    tools = create_coding_tools(tmp_path)
    edit_tool = _get_tool(tools, "edit")

    # 创建测试文件
    test_file = tmp_path / "test.txt"
    test_file.write_text("foo bar baz", encoding="utf-8")

    result = await _execute(
        edit_tool,
        path="test.txt",
        edits=[
            {"oldText": "foo", "newText": "FOO"},
            {"oldText": "bar", "newText": "BAR"},
        ],
    )
    assert "已应用 2 处替换" in result.text

    # 验证文件内容
    assert test_file.read_text(encoding="utf-8") == "FOO BAR baz"


@pytest.mark.asyncio
async def test_edit_fails_no_change_on_error(tmp_path):
    """edit 失败时不应该修改文件。"""
    tools = create_coding_tools(tmp_path)
    edit_tool = _get_tool(tools, "edit")

    # 创建测试文件
    test_file = tmp_path / "test.txt"
    original_content = "Hello, World!"
    test_file.write_text(original_content, encoding="utf-8")

    # 第二个替换不存在
    result = await _execute(
        edit_tool,
        path="test.txt",
        edits=[
            {"oldText": "Hello", "newText": "Hi"},
            {"oldText": "NotExists", "newText": "Something"},
        ],
    )
    assert "错误" in result.text
    assert "未找到" in result.text

    # 验证文件没有被修改
    assert test_file.read_text(encoding="utf-8") == original_content


@pytest.mark.asyncio
async def test_edit_fails_on_non_unique(tmp_path):
    """edit 应该拒绝非唯一的 oldText。"""
    tools = create_coding_tools(tmp_path)
    edit_tool = _get_tool(tools, "edit")

    # 创建测试文件，有重复内容
    test_file = tmp_path / "test.txt"
    test_file.write_text("foo foo foo", encoding="utf-8")

    result = await _execute(
        edit_tool,
        path="test.txt",
        edits=[{"oldText": "foo", "newText": "bar"}],
    )
    assert "错误" in result.text
    assert "出现" in result.text
    assert "唯一" in result.text


# ── 测试：bash 工具 ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_basic_success(tmp_path):
    """bash 应该能执行命令并捕获输出。"""
    tools = create_coding_tools(tmp_path)
    bash_tool = _get_tool(tools, "bash")

    result = await _execute(bash_tool, command="echo 'Hello'")
    assert "Hello" in result.text
    assert "退出码：0" in result.text
    assert not result.is_error


@pytest.mark.asyncio
async def test_captures_stderr(tmp_path):
    """bash 应该能捕获 stderr。"""
    tools = create_coding_tools(tmp_path)
    bash_tool = _get_tool(tools, "bash")

    result = await _execute(bash_tool, command="echo 'error' >&2")
    assert "error" in result.text


@pytest.mark.asyncio
async def test_non_zero_exit_code(tmp_path):
    """bash 应该能处理非零退出码。"""
    tools = create_coding_tools(tmp_path)
    bash_tool = _get_tool(tools, "bash")

    result = await _execute(bash_tool, command="exit 42")
    assert "退出码：42" in result.text
    assert result.is_error
    assert "错误" in result.text


@pytest.mark.asyncio
async def test_runs_in_cwd(tmp_path):
    """bash 应该在项目 cwd 下运行。"""
    tools = create_coding_tools(tmp_path)
    bash_tool = _get_tool(tools, "bash")

    # 创建一个文件
    test_file = tmp_path / "test.txt"
    test_file.write_text("test", encoding="utf-8")

    # pwd 应该返回 cwd
    result = await _execute(bash_tool, command="pwd")
    assert str(tmp_path) in result.text


@pytest.mark.asyncio
async def test_timeout(tmp_path):
    """bash 应该能处理超时。"""
    tools = create_coding_tools(tmp_path)
    bash_tool = _get_tool(tools, "bash")

    # 设置很短的超时
    result = await _execute(bash_tool, command="sleep 10", timeout=0.1)
    assert "超时" in result.text
    assert "错误" in result.text


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_bash_stops_child_processes(tmp_path, cancel):
    import asyncio
    import os

    if os.name != "posix":
        pytest.skip("进程组清理适用于 POSIX")
    tool = _get_tool(create_coding_tools(tmp_path), "bash")
    task = asyncio.create_task(
        _execute(
            tool,
            command="(sleep 0.4; touch survived) & touch ready; wait",
            timeout=5 if cancel else 0.1,
        )
    )
    try:
        async with asyncio.timeout(2):
            while not (tmp_path / "ready").exists():
                await asyncio.sleep(0.01)
        if cancel:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            result = await task
            assert result.is_error
            assert result.details["timed_out"]
        await asyncio.sleep(0.5)
        assert not (tmp_path / "survived").exists()
    finally:
        if not task.done():
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
