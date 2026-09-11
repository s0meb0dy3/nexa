"""单次执行 prompt 的命令行入口。"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import TextIO

from nexa_agent.harness import AgentHarness, AgentHarnessConfig
from nexa_agent.provider import ModelProvider
from nexa_ai.openai_compatible import OpenAICompatibleProvider
from nexa_coding.config import ConfigError, ProviderProfile, load_config, resolve_profile
from nexa_coding.paths import NexaPaths
from nexa_coding.rendering import PrintOutputMode, create_event_renderer
from nexa_coding.tools import create_coding_tools
from nexa_coding.tui import NexaTuiApp

SYSTEM_PROMPT = "你是一个谨慎的 coding agent。需要时使用工具，并简洁地报告结果。"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析这个一次性命令所需的最小参数。"""

    parser = argparse.ArgumentParser(description="运行一次 coding agent prompt")
    parser.add_argument(
        "-p",
        "--print",
        dest="prompt",
        help="要执行的 prompt（print 模式必填；TUI 模式不需要）",
    )
    parser.add_argument(
        "--provider",
        default=None,
        help="使用 config.toml 里的哪个供应商档案（默认用 default_provider）",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="覆盖档案里的模型名（默认用档案里的 model）",
    )
    parser.add_argument(
        "--output",
        type=PrintOutputMode,
        default=PrintOutputMode.text,
        choices=list(PrintOutputMode),
        help="输出形态：text / json / transcript（默认 text）",
    )
    parser.add_argument("--tui", action="store_true", help="打开交互式 TUI 界面")
    parser.add_argument("--cwd", type=Path, default=Path.cwd(), help="工具可访问的项目目录")
    args = parser.parse_args(argv)

    # print 模式（默认）必须有 -p；TUI 模式不需要。
    if not args.tui and not args.prompt:
        parser.error("print 模式需要 -p/--print 参数（或用 --tui 打开交互界面）")

    return args


def build_provider(profile: ProviderProfile) -> ModelProvider:
    """把供应商档案变成具体的 Provider。

    这里是配置（数据）和 nexa_ai（实现）之间唯一的接缝：将来接 Anthropic
    等新协议，只需在这里加一个分支 + 在 nexa_ai 加一个类，调用方都不用动。
    """

    if profile.api == "openai":
        return OpenAICompatibleProvider(
            name=profile.name,
            api_key=profile.api_key,
            base_url=profile.base_url,
        )
    raise ConfigError(f"未知的 api 类型 {profile.api!r}（档案 {profile.name!r}）")


def resolve_provider(
    provider_name: str | None = None,
    *,
    config_path: Path | None = None,
) -> tuple[ModelProvider, ProviderProfile]:
    """读配置、选档案、造 Provider，返回 (provider, profile)。

    Args:
        provider_name: --provider 指定的档案名；None 表示用默认档案。
        config_path: 配置文件路径；None 时用 NexaPaths 的 canonical 位置。

    Raises:
        ConfigError: 配置缺失/错误，或档案选不出来。
    """

    path = config_path if config_path is not None else NexaPaths().config_file
    config = load_config(path)
    profile = resolve_profile(config, provider_name)
    return build_provider(profile), profile


async def run_prompt(
    args: argparse.Namespace,
    *,
    provider: ModelProvider,
    stdout: TextIO = sys.stdout,
    stderr: TextIO = sys.stderr,
) -> int:
    """运行一条 prompt，并把最终回答和运行诊断分流输出。

    provider 由调用方（main 或测试）解析好传入，这里只负责跑和渲染。
    """

    try:
        cwd = args.cwd.resolve()
        if not cwd.is_dir():
            print(f"错误：--cwd 不是目录：{cwd}", file=stderr)
            return 2

        harness = AgentHarness(
            AgentHarnessConfig(
                provider=provider,
                model=args.model,
                system=SYSTEM_PROMPT,
                tools=create_coding_tools(cwd),
            )
        )

        # 渲染器把事件流变成输出；只观察，不改 agent 行为。
        renderer = create_event_renderer(args.output, stdout=stdout, stderr=stderr)

        async for event in harness.prompt(args.prompt):
            renderer.render(event)

        # finish() 返回是否成功，直接作为 CLI 退出码。
        return 0 if renderer.finish() else 1
    except (OSError, ValueError) as error:
        print(f"错误：{error}", file=stderr)
        return 2


def main(argv: list[str] | None = None) -> None:
    """控制台脚本入口。"""

    args = parse_args(argv)
    try:
        provider, profile = resolve_provider(args.provider)
    except ConfigError as error:
        # 配置问题（没建 config.toml、档案不存在等）：友好提示而不是堆栈。
        print(f"错误：{error}", file=sys.stderr)
        raise SystemExit(2) from None

    # 命令行 --model 优先，否则用档案里的默认模型。
    args.model = args.model or profile.model

    if args.tui:
        # Textual 自管事件循环，直接调用 App.run()，不要包 asyncio.run()。
        NexaTuiApp(provider, model=args.model, cwd=args.cwd.resolve()).run()
        return
    raise SystemExit(asyncio.run(run_prompt(args, provider=provider)))


__all__ = ["build_provider", "main", "parse_args", "resolve_provider", "run_prompt"]
