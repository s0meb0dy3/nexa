"""测试单次执行 prompt 的命令行入口。"""

from __future__ import annotations

import argparse
from io import StringIO

import pytest

from nexa_agent.messages import AssistantMessage, TextContent
from nexa_agent.provider_events import ProviderErrorEvent, ProviderResponseEndEvent
from nexa_ai.fake import FakeProvider
from nexa_ai.openai_compatible import OpenAICompatibleProvider
from nexa_coding.cli import build_provider, parse_args, resolve_provider, run_prompt
from nexa_coding.config import ConfigError, ProviderProfile
from nexa_coding.rendering import PrintOutputMode


def _args(tmp_path) -> argparse.Namespace:
    return argparse.Namespace(
        prompt="读取 README",
        model="test-model",
        provider=None,
        cwd=tmp_path,
        output=PrintOutputMode.text,
    )


@pytest.mark.asyncio
async def test_run_prompt_prints_final_answer_and_registers_coding_tools(tmp_path):
    """CLI 应输出最终回答，并将四个本地工具交给 Provider。"""
    provider = FakeProvider(
        [
            [
                ProviderResponseEndEvent(
                    message=AssistantMessage(content=[TextContent(text="总结完成")])
                )
            ]
        ]
    )
    stdout, stderr = StringIO(), StringIO()

    exit_code = await run_prompt(_args(tmp_path), provider=provider, stdout=stdout, stderr=stderr)

    assert exit_code == 0
    assert stdout.getvalue() == "总结完成\n"
    assert stderr.getvalue() == ""
    assert {tool.name for tool in provider.calls[0][3]} == {"read", "write", "edit", "bash"}


@pytest.mark.asyncio
async def test_run_prompt_returns_nonzero_for_provider_error(tmp_path):
    """Provider 失败时 CLI 应在 stderr 报错并返回非零。"""
    provider = FakeProvider([[ProviderErrorEvent(message="连接失败")]])
    stdout, stderr = StringIO(), StringIO()

    exit_code = await run_prompt(_args(tmp_path), provider=provider, stdout=stdout, stderr=stderr)

    assert exit_code == 1
    assert stdout.getvalue() == ""
    assert "连接失败" in stderr.getvalue()


@pytest.mark.asyncio
async def test_run_prompt_reports_empty_provider_error(tmp_path):
    """Provider 没有错误详情时，也应给脚本调用者留下可读诊断。"""
    provider = FakeProvider([[ProviderErrorEvent(message="")]])
    stdout, stderr = StringIO(), StringIO()

    exit_code = await run_prompt(_args(tmp_path), provider=provider, stdout=stdout, stderr=stderr)

    assert exit_code == 1
    assert stdout.getvalue() == ""
    assert "Provider 未提供错误详情" in stderr.getvalue()


def test_parse_args_provider_and_model_default_none():
    """--provider/--model 缺省为 None（由配置档案补上），-p 仍必填。"""
    args = parse_args(["-p", "hi"])

    assert args.provider is None
    assert args.model is None
    assert args.prompt == "hi"


def test_parse_args_accepts_provider():
    """--provider 被正确解析。"""
    args = parse_args(["-p", "hi", "--provider", "ollama"])

    assert args.provider == "ollama"


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
model = "deepseek-chat"
api_key = "sk-abc"
""",
        encoding="utf-8",
    )

    provider, profile = resolve_provider(config_path=config)

    assert profile.model == "deepseek-chat"
    assert provider.name == "deepseek"


def test_resolve_provider_missing_config_raises(tmp_path):
    """没有配置文件 → ConfigError（main 会转成友好提示）。"""
    with pytest.raises(ConfigError):
        resolve_provider(config_path=tmp_path / "absent.toml")
