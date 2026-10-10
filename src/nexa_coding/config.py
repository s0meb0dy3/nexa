"""NEXA 的用户配置文件：供应商档案与默认供应商。

配置存在 ~/.nexa/config.toml（路径由 NexaPaths.config_file 决定），形如：

    default_provider = "deepseek"

    [providers.deepseek]
    base_url = "https://api.deepseek.com"
    model    = "deepseek-flash"
    api_key  = "sk-..."

    [providers.ollama]           # api 可省，默认 "openai"
    base_url = "http://localhost:11434/v1"
    model    = "qwen3:8b"
    api_key  = "ollama"

本模块只负责"读文件 + 解析成数据"，不 import nexa_ai，也不构造 Provider——
把档案变成 Provider 是 providers.build_provider 的事，这样数据与实现解耦。
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

# 供应商协议类型；多个供应商可共用同一种协议实现（openai 兼容最常见）。
DEFAULT_API = "openai"

# 缺配置时给用户看的最小示例，方便直接照抄到 config.toml。
_CONFIG_EXAMPLE = """\
default_provider = "deepseek"

[providers.deepseek]
base_url = "https://api.deepseek.com"
model    = "deepseek-flash"
api_key  = "sk-..."
"""


class ConfigError(ValueError):
    """配置文件缺失、格式错误或档案不完整时抛出。"""


@dataclass(frozen=True)
class ProviderProfile:
    """一个供应商接入点的全部信息。

    Attributes:
        name: 档案名（就是 config.toml 里 [providers.X] 的 X）。
        base_url: API 根地址。
        model: 默认模型名。
        api_key: 访问凭证。
        api: 协议类型；目前只实现 "openai"（OpenAI 兼容协议）。
    """

    name: str
    base_url: str
    model: str
    api_key: str
    api: str = DEFAULT_API


@dataclass(frozen=True)
class NexaConfig:
    """一份解析后的完整配置。

    Attributes:
        path: 配置来源文件（报错信息里用来指引用户去改哪个文件）。
        default_provider: 未显式指定 --provider 时用哪个档案；可为 None。
        providers: 档案名 -> 档案。
    """

    path: Path
    default_provider: str | None = None
    providers: dict[str, ProviderProfile] = field(default_factory=dict)


def load_config(path: Path) -> NexaConfig:
    """读取并解析配置文件。

    文件不存在时返回空配置（不算错误）——这样"还没建配置"和"配置为空"
    是同一条路径，调用方只需处理 resolve_profile 的报错。

    Raises:
        ConfigError: TOML 语法错误、providers 结构不对、档案字段缺失或类型不对。
    """

    if not path.is_file():
        return NexaConfig(path=path)

    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as error:
        raise ConfigError(f"配置文件 {path} 不是合法的 TOML：{error}") from error
    except OSError as error:
        raise ConfigError(f"无法读取配置文件 {path}：{error}") from error

    # default_provider：可省，给了必须是字符串。
    default_provider = raw.get("default_provider")
    if default_provider is not None and not isinstance(default_provider, str):
        raise ConfigError(f"配置文件 {path} 的 default_provider 必须是字符串")

    # providers：可省；给了必须是"表 of 表"。
    providers_raw = raw.get("providers", {})
    if not isinstance(providers_raw, dict):
        raise ConfigError(f"配置文件 {path} 的 [providers] 必须是表（table）")

    providers: dict[str, ProviderProfile] = {}
    for name, body in providers_raw.items():
        providers[name] = _parse_profile(path, name, body)

    return NexaConfig(path=path, default_provider=default_provider, providers=providers)


def resolve_profile(config: NexaConfig, name: str | None = None) -> ProviderProfile:
    """选出一个档案。

    优先级：显式 name > config.default_provider > 唯一档案。
    name 不存在、default_provider 不存在、或没有可选项时报 ConfigError。

    Raises:
        ConfigError: 选不出档案（附上配置路径、可用档案和最小示例）。
    """

    if name is not None:
        profile = config.providers.get(name)
        if profile is None:
            raise ConfigError(_no_such_profile_message(config, name))
        return profile

    if config.default_provider is not None:
        profile = config.providers.get(config.default_provider)
        if profile is None:
            raise ConfigError(
                f"配置文件 {config.path} 的 default_provider={config.default_provider!r} "
                f"没有对应档案；{_available_message(config)}"
            )
        return profile

    # 没有默认值：恰好一个档案就自动用它，多个则要求用户指明。
    if len(config.providers) == 1:
        return next(iter(config.providers.values()))

    if not config.providers:
        raise ConfigError(
            f"没有找到可用的供应商配置（读取自 {config.path}）。\n"
            f"请创建该文件，内容形如：\n\n{_CONFIG_EXAMPLE}"
        )

    raise ConfigError(f"配置里有多个档案但没指定默认；{_available_message(config)}")


def _parse_profile(path: Path, name: str, body: object) -> ProviderProfile:
    """把 [providers.<name>] 表解析成 ProviderProfile，字段缺失/类型错时报错。"""

    if not isinstance(body, dict):
        raise ConfigError(f"配置文件 {path} 的 [providers.{name}] 必须是表（table）")

    def _require(key: str) -> str:
        value = body.get(key)
        if value is None:
            raise ConfigError(f"配置文件 {path} 的 [providers.{name}] 缺少 {key!r}")
        if not isinstance(value, str):
            raise ConfigError(f"配置文件 {path} 的 [providers.{name}].{key} 必须是字符串")
        return value

    api = body.get("api", DEFAULT_API)
    if not isinstance(api, str):
        raise ConfigError(f"配置文件 {path} 的 [providers.{name}].api 必须是字符串")

    return ProviderProfile(
        name=name,
        base_url=_require("base_url"),
        model=_require("model"),
        api_key=_require("api_key"),
        api=api,
    )


def _available_message(config: NexaConfig) -> str:
    """拼出"有哪些档案可用"的提示片段。"""

    if not config.providers:
        return f"当前没有任何档案，请编辑 {config.path}"
    names = "、".join(sorted(config.providers))
    return f"可用档案：{names}（用 --provider NAME 指定）"


def _no_such_profile_message(config: NexaConfig, name: str) -> str:
    """指定了不存在的档案时的错误信息。"""

    return f"找不到供应商档案 {name!r}（读取自 {config.path}）；{_available_message(config)}"


__all__ = [
    "ConfigError",
    "NexaConfig",
    "ProviderProfile",
    "load_config",
    "resolve_profile",
]
