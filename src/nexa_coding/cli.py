"""交互式编码助手的命令行入口。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from nexa_agent.provider import ModelProvider
from nexa_ai.openai_compatible import OpenAICompatibleProvider
from nexa_coding.config import ConfigError, ProviderProfile, load_config, resolve_profile
from nexa_coding.paths import NexaPaths
from nexa_coding.tui import NexaTuiApp


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析供应商、模型和工作目录，默认启动交互界面。"""

    parser = argparse.ArgumentParser(description="启动 NEXA 交互式编码助手")
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
    parser.add_argument("--cwd", type=Path, default=Path.cwd(), help="工具可访问的项目目录")
    return parser.parse_args(argv)


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


def main(argv: list[str] | None = None) -> None:
    """控制台脚本入口。"""

    args = parse_args(argv)
    cwd = args.cwd.resolve()
    if not cwd.is_dir():
        print(f"错误：--cwd 不是目录：{cwd}", file=sys.stderr)
        raise SystemExit(2)

    try:
        provider, profile = resolve_provider(args.provider)
    except ConfigError as error:
        # 配置问题（没建 config.toml、档案不存在等）：友好提示而不是堆栈。
        print(f"错误：{error}", file=sys.stderr)
        raise SystemExit(2) from None

    # Textual 自管事件循环；所有任务统一经 CodingSession 执行和保存。
    model = args.model or profile.model
    NexaTuiApp(provider, model=model, cwd=cwd).run()


__all__ = ["build_provider", "main", "parse_args", "resolve_provider"]
