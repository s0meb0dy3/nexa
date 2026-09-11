"""测试配置文件解析与档案选择。"""

from __future__ import annotations

import pytest

from nexa_coding.config import (
    ConfigError,
    NexaConfig,
    ProviderProfile,
    load_config,
    resolve_profile,
)


def _write(tmp_path, text: str):
    """把 TOML 文本写进临时配置文件并返回路径。"""

    path = tmp_path / "config.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_load_config_parses_profiles_and_default(tmp_path):
    """正常解析：档案字段齐全，default_provider 生效。"""
    path = _write(
        tmp_path,
        """
default_provider = "deepseek"

[providers.deepseek]
base_url = "https://api.deepseek.com"
model = "deepseek-chat"
api_key = "sk-abc"
""",
    )

    config = load_config(path)

    assert config.default_provider == "deepseek"
    assert set(config.providers) == {"deepseek"}
    profile = config.providers["deepseek"]
    assert profile.name == "deepseek"
    assert profile.base_url == "https://api.deepseek.com"
    assert profile.model == "deepseek-chat"
    assert profile.api_key == "sk-abc"
    assert profile.api == "openai"  # 缺省协议


def test_load_config_api_field_can_override(tmp_path):
    """显式 api 字段被保留（为将来非 OpenAI 协议留的接缝）。"""
    path = _write(
        tmp_path,
        """
[providers.claude]
api = "anthropic"
base_url = "https://api.anthropic.com"
model = "claude-sonnet-4"
api_key = "sk-ant"
""",
    )

    profile = load_config(path).providers["claude"]
    assert profile.api == "anthropic"


def test_load_config_missing_file_returns_empty(tmp_path):
    """文件不存在不算错误：返回空配置，走 resolve_profile 的报错路径。"""
    config = load_config(tmp_path / "nope.toml")

    assert config.default_provider is None
    assert config.providers == {}


def test_load_config_rejects_bad_toml(tmp_path):
    """语法错误 → ConfigError（含文件路径）。"""
    path = _write(tmp_path, "this is not = = toml")

    with pytest.raises(ConfigError, match="TOML"):
        load_config(path)


def test_load_config_rejects_missing_field(tmp_path):
    """档案缺 api_key → ConfigError，点名档案和字段。"""
    path = _write(
        tmp_path,
        """
[providers.bad]
base_url = "https://x"
model = "m"
""",
    )

    with pytest.raises(ConfigError, match="api_key"):
        load_config(path)


def test_load_config_rejects_wrong_type(tmp_path):
    """字段类型不对 → ConfigError。"""
    path = _write(
        tmp_path,
        """
[providers.bad]
base_url = 123
model = "m"
api_key = "k"
""",
    )

    with pytest.raises(ConfigError, match="base_url"):
        load_config(path)


def test_load_config_rejects_non_table_providers(tmp_path):
    """providers 不是表 → ConfigError。"""
    path = _write(tmp_path, 'providers = "nope"')

    with pytest.raises(ConfigError, match="providers"):
        load_config(path)


def test_resolve_profile_prefers_explicit_name(tmp_path):
    """显式 name 优先于 default_provider。"""
    config = NexaConfig(
        path=tmp_path,
        default_provider="a",
        providers={
            "a": ProviderProfile("a", "https://a", "ma", "ka"),
            "b": ProviderProfile("b", "https://b", "mb", "kb"),
        },
    )

    assert resolve_profile(config, "b").name == "b"


def test_resolve_profile_uses_default(tmp_path):
    """没给 name 时用 default_provider。"""
    config = NexaConfig(
        path=tmp_path,
        default_provider="b",
        providers={
            "a": ProviderProfile("a", "https://a", "ma", "ka"),
            "b": ProviderProfile("b", "https://b", "mb", "kb"),
        },
    )

    assert resolve_profile(config, None).name == "b"


def test_resolve_profile_single_profile_needs_no_default(tmp_path):
    """只有一个档案且没设默认 → 自动用它。"""
    config = NexaConfig(
        path=tmp_path,
        providers={"only": ProviderProfile("only", "https://x", "m", "k")},
    )

    assert resolve_profile(config).name == "only"


def test_resolve_profile_unknown_name_lists_available(tmp_path):
    """指定了不存在的档案 → 报错并列出可用档案。"""
    config = NexaConfig(
        path=tmp_path,
        providers={"a": ProviderProfile("a", "https://a", "ma", "ka")},
    )

    with pytest.raises(ConfigError, match="ghost"):
        resolve_profile(config, "ghost")


def test_resolve_profile_empty_config_shows_example(tmp_path):
    """完全没有配置 → 报错里给出路径和最小示例。"""
    config = NexaConfig(path=tmp_path / "config.toml")

    with pytest.raises(ConfigError) as exc_info:
        resolve_profile(config)

    message = str(exc_info.value)
    assert "config.toml" in message
    assert "providers.deepseek" in message  # 示例内容


def test_resolve_profile_ambiguous_without_default(tmp_path):
    """多个档案又没设默认 → 报错（不能替你猜）。"""
    config = NexaConfig(
        path=tmp_path,
        providers={
            "a": ProviderProfile("a", "https://a", "ma", "ka"),
            "b": ProviderProfile("b", "https://b", "mb", "kb"),
        },
    )

    with pytest.raises(ConfigError, match="默认"):
        resolve_profile(config)
