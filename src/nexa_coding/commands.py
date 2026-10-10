"""本地命令元数据，同时用于帮助和补全。"""

from pathlib import Path

COMMANDS = {
    "help": "显示命令和快捷键",
    "new": "新建会话，保留旧记录",
    "resume": "选择已有会话；也可输入 /resume ID",
    "model": "选择配置中的模型；也可输入 /model 档案名",
    "thinking": "查看思考设置；/thinking 等级 修改模型思考",
    "exit": "等待取消清理后退出",
}


def parse_command(text: str) -> tuple[str, str] | None:
    stripped = text.strip()
    if not stripped.startswith("/") or stripped.startswith("/skill:"):
        return None
    parts = stripped.split(maxsplit=1)
    token = parts[0]
    argument = parts[1] if len(parts) > 1 else ""
    name = token[1:]
    if name not in COMMANDS:
        # 绝对路径仍可以作为普通提示词输入。
        if "/" in name or Path(token).exists():
            return None
        raise ValueError(f"未知命令 {token}；输入 /help 查看可用命令")
    if argument.strip() and name in ("help", "new", "exit"):
        raise ValueError(f"/{name} 不接受参数")
    return name, argument.strip()


def completions(text: str) -> list[str]:
    if not text.startswith("/") or any(c.isspace() for c in text):
        return []
    return [name for name in COMMANDS if f"/{name}".startswith(text)]
