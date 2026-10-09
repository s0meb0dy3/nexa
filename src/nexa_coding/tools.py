"""本地 Coding Tools：read、write、edit、bash。

这些工具让 Agent 能够操作本地文件系统：
- read: 读取文件内容
- write: 写入文件
- edit: 精确替换文件内容
- bash: 执行 shell 命令

所有工具都遵循安全原则：
- 相对路径以项目 cwd 解析
- 拒绝越界路径
- 大输出截断并提示
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
from pathlib import Path
from typing import Any

from nexa_agent.tools import AgentTool, AgentToolResult

# ── 常量 ──────────────────────────────────────────────────────────────────────

# 单次读取/输出的最大字符数，超过就截断
MAX_OUTPUT_CHARS = 50_000

# bash 命令默认超时（秒）
DEFAULT_BASH_TIMEOUT = 30


# ── 辅助函数 ──────────────────────────────────────────────────────────────────


def _resolve_path(cwd: Path, path_str: str) -> Path:
    """解析路径，确保在项目 cwd 内。

    Args:
        cwd: 项目根目录
        path_str: 用户提供的路径（可以是相对或绝对）

    Returns:
        解析后的绝对路径

    Raises:
        ValueError: 路径越界
    """
    path = Path(path_str)

    # 相对路径以 cwd 解析
    if not path.is_absolute():
        path = cwd / path

    # 规范化（处理 .. 等）
    path = path.resolve()

    # 检查是否在项目目录内
    try:
        path.relative_to(cwd.resolve())
    except ValueError as e:
        raise ValueError(f"路径越界：{path} 不在项目目录 {cwd} 内") from e

    return path


def _truncate(text: str, max_chars: int = MAX_OUTPUT_CHARS) -> tuple[str, bool]:
    """截断文本，返回 (截断后的文本, 是否被截断)。"""
    if len(text) <= max_chars:
        return text, False

    truncated = text[:max_chars]
    remaining = len(text) - max_chars
    hint = f"\n\n... [输出被截断，剩余 {remaining} 字符。使用 offset 参数继续读取]"
    return truncated + hint, True


def _make_success(text: str, **details: Any) -> AgentToolResult:
    """创建成功结果。"""
    return AgentToolResult(content=text, details=details if details else None)


def _make_error(message: str, **details: Any) -> AgentToolResult:
    """创建失败结果。"""
    return AgentToolResult(
        content=f"错误：{message}", details=details if details else None, is_error=True
    )


# ── 工具实现 ──────────────────────────────────────────────────────────────────


def _create_read_tool(cwd: Path) -> AgentTool:
    """创建 read 工具：读取文件内容。"""

    async def execute(tool_call_id: str, arguments: dict[str, Any]) -> AgentToolResult:
        # 解析参数
        path_str = arguments.get("path")
        if not isinstance(path_str, str):
            return _make_error("缺少必需参数 'path'（字符串）")

        offset = arguments.get("offset", 1)
        limit = arguments.get("limit")

        # 类型检查
        if not isinstance(offset, int) or offset < 1:
            return _make_error("'offset' 必须是正整数")
        if limit is not None and (not isinstance(limit, int) or limit < 1):
            return _make_error("'limit' 必须是正整数")

        # 解析路径
        try:
            path = _resolve_path(cwd, path_str)
        except ValueError as e:
            return _make_error(str(e))

        # 检查文件是否存在
        if not path.exists():
            return _make_error(f"文件不存在：{path}")

        # 检查是否是文件
        if not path.is_file():
            return _make_error(f"不是文件：{path}")

        # 读取文件
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return _make_error("文件不是有效的 UTF-8 编码")
        except Exception as e:
            return _make_error(f"读取失败：{e}")

        # 按行分割，应用 offset 和 limit
        lines = content.splitlines(keepends=True)
        total_lines = len(lines)

        # offset 是 1-indexed
        start_idx = offset - 1
        if start_idx >= total_lines:
            return _make_success(f"文件共 {total_lines} 行，offset {offset} 超出范围")

        # 应用 limit
        end_idx = start_idx + limit if limit is not None else total_lines

        selected_lines = lines[start_idx:end_idx]
        result = "".join(selected_lines)

        # 截断
        result, was_truncated = _truncate(result)

        # 添加行号信息
        end_line = start_idx + len(selected_lines)
        info = f"文件：{path}\n总行数：{total_lines}\n显示：第 {offset}-{end_line} 行\n\n"
        return _make_success(info + result, truncated=was_truncated)

    return AgentTool(
        name="read",
        description="读取文件内容（UTF-8）。支持 offset（起始行，1-indexed）和 limit（行数）参数。",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "文件路径（相对或绝对）"},
                "offset": {
                    "type": "integer",
                    "description": "起始行号（1-indexed），默认 1",
                    "default": 1,
                },
                "limit": {
                    "type": "integer",
                    "description": "读取行数，默认读取全部",
                },
            },
            "required": ["path"],
        },
        execute_fn=execute,
    )


def _create_write_tool(cwd: Path) -> AgentTool:
    """创建 write 工具：写入文件。"""

    async def execute(tool_call_id: str, arguments: dict[str, Any]) -> AgentToolResult:
        # 解析参数
        path_str = arguments.get("path")
        if not isinstance(path_str, str):
            return _make_error("缺少必需参数 'path'（字符串）")

        content = arguments.get("content")
        if not isinstance(content, str):
            return _make_error("缺少必需参数 'content'（字符串）")

        # 解析路径
        try:
            path = _resolve_path(cwd, path_str)
        except ValueError as e:
            return _make_error(str(e))

        # 创建父目录
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            return _make_error(f"创建目录失败：{e}")

        # 写入文件
        try:
            path.write_text(content, encoding="utf-8")
        except Exception as e:
            return _make_error(f"写入失败：{e}")

        return _make_success(f"已写入 {len(content)} 字符到 {path}")

    return AgentTool(
        name="write",
        description="写入文件内容（UTF-8）。自动创建父目录。",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "文件路径（相对或绝对）"},
                "content": {"type": "string", "description": "文件内容"},
            },
            "required": ["path", "content"],
        },
        execute_fn=execute,
    )


def _create_edit_tool(cwd: Path) -> AgentTool:
    """创建 edit 工具：精确替换文件内容。"""

    async def execute(tool_call_id: str, arguments: dict[str, Any]) -> AgentToolResult:
        # 解析参数
        path_str = arguments.get("path")
        if not isinstance(path_str, str):
            return _make_error("缺少必需参数 'path'（字符串）")

        edits = arguments.get("edits")
        if not isinstance(edits, list):
            return _make_error("缺少必需参数 'edits'（列表）")

        if len(edits) == 0:
            return _make_error("'edits' 不能为空")

        # 验证每个 edit 的格式
        for i, edit in enumerate(edits):
            if not isinstance(edit, dict):
                return _make_error(f"edits[{i}] 必须是字典")
            if "oldText" not in edit or "newText" not in edit:
                return _make_error(f"edits[{i}] 必须包含 'oldText' 和 'newText'")
            if not isinstance(edit["oldText"], str) or not isinstance(edit["newText"], str):
                return _make_error(f"edits[{i}] 的 'oldText' 和 'newText' 必须是字符串")

        # 解析路径
        try:
            path = _resolve_path(cwd, path_str)
        except ValueError as e:
            return _make_error(str(e))

        # 检查文件是否存在
        if not path.exists():
            return _make_error(f"文件不存在：{path}")

        if not path.is_file():
            return _make_error(f"不是文件：{path}")

        # 读取文件
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return _make_error("文件不是有效的 UTF-8 编码")
        except Exception as e:
            return _make_error(f"读取失败：{e}")

        # 先验证所有替换，再执行
        new_content = content
        replacements = []

        for i, edit in enumerate(edits):
            old_text = edit["oldText"]
            new_text = edit["newText"]

            # 检查 oldText 是否存在
            count = new_content.count(old_text)
            if count == 0:
                return _make_error(f"edits[{i}] 的 'oldText' 在文件中未找到")
            if count > 1:
                return _make_error(f"edits[{i}] 的 'oldText' 在文件中出现 {count} 次，必须唯一")

            # 记录替换
            replacements.append((old_text, new_text))
            new_content = new_content.replace(old_text, new_text, 1)

        # 所有验证通过，写入文件
        try:
            path.write_text(new_content, encoding="utf-8")
        except Exception as e:
            return _make_error(f"写入失败：{e}")

        return _make_success(f"已应用 {len(replacements)} 处替换到 {path}")

    return AgentTool(
        name="edit",
        description=(
            "精确替换文件内容。提供 oldText → newText 的替换列表。"
            "先验证所有替换，再写入文件；任意一个失败则不改文件。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "文件路径（相对或绝对）"},
                "edits": {
                    "type": "array",
                    "description": "替换列表，每项包含 oldText 和 newText",
                    "items": {
                        "type": "object",
                        "properties": {
                            "oldText": {"type": "string", "description": "要被替换的文本"},
                            "newText": {"type": "string", "description": "替换成的文本"},
                        },
                        "required": ["oldText", "newText"],
                    },
                },
            },
            "required": ["path", "edits"],
        },
        execute_fn=execute,
    )


def _create_bash_tool(cwd: Path) -> AgentTool:
    """创建 bash 工具：执行 shell 命令。"""

    async def execute(tool_call_id: str, arguments: dict[str, Any]) -> AgentToolResult:
        # 解析参数
        command = arguments.get("command")
        if not isinstance(command, str):
            return _make_error("缺少必需参数 'command'（字符串）")

        timeout = arguments.get("timeout", DEFAULT_BASH_TIMEOUT)
        if not isinstance(timeout, (int, float)) or timeout <= 0:
            return _make_error("'timeout' 必须是正数")

        # 执行命令
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                cwd=str(cwd),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=os.name == "posix",
            )

            async def stop_process() -> None:
                # POSIX 下连同 shell 子进程一起停止，避免取消后命令继续运行。
                with contextlib.suppress(ProcessLookupError):
                    if os.name == "posix":
                        os.killpg(proc.pid, signal.SIGKILL)
                    else:
                        proc.kill()
                await proc.communicate()

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=timeout,
                )
                timed_out = False
            except TimeoutError:
                await stop_process()
                timed_out = True
                stdout, stderr = b"", b""
            except asyncio.CancelledError:
                await stop_process()
                raise

        except Exception as e:
            return _make_error(f"执行失败：{e}")

        # 解码输出
        try:
            stdout_text = stdout.decode("utf-8", errors="replace") if stdout else ""
            stderr_text = stderr.decode("utf-8", errors="replace") if stderr else ""
        except Exception:
            stdout_text, stderr_text = "", ""

        # 组合输出
        output_parts = []
        if stdout_text:
            output_parts.append(stdout_text)
        if stderr_text:
            output_parts.append(f"[stderr]\n{stderr_text}")

        output = "\n".join(output_parts) if output_parts else "(无输出)"

        # 截断
        output, was_truncated = _truncate(output)

        # 构建结果
        if timed_out:
            result_text = f"命令超时（>{timeout}秒），已终止。\n\n{output}"
            return _make_error(result_text, exit_code=None, timed_out=True)

        exit_code = proc.returncode
        result_text = f"退出码：{exit_code}\n\n{output}"

        if exit_code == 0:
            return _make_success(result_text, exit_code=exit_code, truncated=was_truncated)
        else:
            return _make_error(result_text, exit_code=exit_code, truncated=was_truncated)

    return AgentTool(
        name="bash",
        description=f"在项目目录下执行 shell 命令。默认超时 {DEFAULT_BASH_TIMEOUT} 秒。",
        parameters={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "要执行的 shell 命令"},
                "timeout": {
                    "type": "number",
                    "description": f"超时秒数，默认 {DEFAULT_BASH_TIMEOUT}",
                    "default": DEFAULT_BASH_TIMEOUT,
                },
            },
            "required": ["command"],
        },
        execute_fn=execute,
    )


# ── 工厂函数 ──────────────────────────────────────────────────────────────────


def create_coding_tools(cwd: str | Path) -> tuple[AgentTool, ...]:
    """创建一组 coding tools。

    Args:
        cwd: 项目根目录，所有相对路径都基于此解析

    Returns:
        包含 read、write、edit、bash 四个工具的元组
    """
    cwd_path = Path(cwd).resolve()

    return (
        _create_read_tool(cwd_path),
        _create_write_tool(cwd_path),
        _create_edit_tool(cwd_path),
        _create_bash_tool(cwd_path),
    )


__all__ = ["create_coding_tools"]
