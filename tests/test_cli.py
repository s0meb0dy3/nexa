"""测试交互式入口的参数、配置解析和启动行为。"""

from __future__ import annotations

import pytest

from nexa_ai.fake import FakeProvider
from nexa_ai.openai_compatible import OpenAICompatibleProvider
from nexa_coding import cli
from nexa_coding.cli import build_provider, parse_args, resolve_provider
from nexa_coding.config import ConfigError, ProviderProfile


def test_parse_args_defaults_to_interactive(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    args = parse_args([])

    assert args.provider is None
    assert args.model is None
    assert args.cwd == tmp_path


@pytest.mark.parametrize(
    "option", [["-p", "hi"], ["--print", "hi"], ["--output", "json"], ["--tui"]]
)
def test_parse_args_rejects_removed_modes(option):
    with pytest.raises(SystemExit) as error:
        parse_args(option)
    assert error.value.code == 2


@pytest.mark.parametrize("override", [None, "custom-model"])
def test_main_starts_tui_with_selected_project_and_model(monkeypatch, tmp_path, override):
    """验证入口会传递供应商、规范化目录和模型覆盖，并启动交互界面。"""
    provider = FakeProvider([])
    profile = ProviderProfile("ollama", "http://localhost/v1", "default-model", "key")
    calls = []

    def resolve(name):
        assert name == "ollama"
        return provider, profile

    class App:
        def __init__(self, selected_provider, *, model, cwd):
            calls.append((selected_provider, model, cwd))

        def run(self):
            calls.append("started")

    monkeypatch.setattr(cli, "resolve_provider", resolve)
    monkeypatch.setattr(cli, "NexaTuiApp", App)
    args = ["--provider", "ollama", "--cwd", str(tmp_path / ".." / tmp_path.name)]
    if override:
        args += ["--model", override]

    cli.main(args)

    assert calls == [(provider, override or profile.model, tmp_path), "started"]


def test_main_rejects_invalid_project_before_loading_config(monkeypatch, tmp_path, capsys):
    def resolve(name):
        pytest.fail("无效目录不应继续加载配置或启动界面")

    monkeypatch.setattr(cli, "resolve_provider", resolve)
    with pytest.raises(SystemExit) as error:
        cli.main(["--cwd", str(tmp_path / "missing")])

    assert error.value.code == 2
    assert "--cwd 不是目录" in capsys.readouterr().err


def test_main_reports_config_error(monkeypatch, capsys):
    def resolve(name):
        raise ConfigError("缺少供应商配置")

    monkeypatch.setattr(cli, "resolve_provider", resolve)
    with pytest.raises(SystemExit) as error:
        cli.main([])

    assert error.value.code == 2
    assert "缺少供应商配置" in capsys.readouterr().err


def test_build_provider_openai_compatible():
    """openai 协议档案 → OpenAICompatibleProvider，字段透传。"""
    profile = ProviderProfile(
        name="kimi",
        base_url="https://api.moonshot.cn/v1/",
        model="kimi-k2",
        api_key="sk-x",
    )

    provider = build_provider(profile)

    assert isinstance(provider, OpenAICompatibleProvider)
    assert provider.name == "kimi"
    assert provider.base_url == "https://api.moonshot.cn/v1"  # 尾部斜杠被去掉
    assert provider.api_key == "sk-x"


def test_build_provider_rejects_unknown_api():
    """未知 api 类型 → ConfigError（不静默降级）。"""
    profile = ProviderProfile("x", "https://x", "m", "k", api="gemini")

    with pytest.raises(ConfigError, match="gemini"):
        build_provider(profile)


def test_resolve_provider_reads_config(tmp_path):
    """resolve_provider 从指定配置文件选档案并造出 Provider。"""
    config = tmp_path / "config.toml"
    config.write_text(
        """
default_provider = "deepseek"

[providers.deepseek]
base_url = "https://api.deepseek.com"
model = "deepseek-flash"
api_key = "sk-abc"
""",
        encoding="utf-8",
    )

    provider, profile = resolve_provider(config_path=config)

    assert profile.model == "deepseek-flash"
    assert provider.name == "deepseek"


def test_resolve_provider_missing_config_raises(tmp_path):
    """没有配置文件 → ConfigError（main 会转成友好提示）。"""
    with pytest.raises(ConfigError):
        resolve_provider(config_path=tmp_path / "absent.toml")
