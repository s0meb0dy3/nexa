"""把已验证的供应商档案转换成模型 Provider。"""

from urllib.parse import urlparse

from nexa_agent.provider import ModelProvider
from nexa_ai.anthropic import AnthropicProvider
from nexa_ai.deepseek import DeepSeekProvider
from nexa_ai.minimax import MiniMaxProvider
from nexa_ai.openai_compatible import OpenAICompatibleProvider
from nexa_coding.config import ConfigError, ProviderProfile


def build_provider(profile: ProviderProfile) -> ModelProvider:
    if profile.api in ("openai", "anthropic"):
        hostname = urlparse(profile.base_url).hostname
        provider_type: type[OpenAICompatibleProvider]
        if hostname == "api.deepseek.com" or profile.name == "deepseek":
            provider_type = DeepSeekProvider
        elif hostname in ("api.minimax.cn", "api.minimax.io") or profile.name == "minimax":
            provider_type = MiniMaxProvider
        else:
            provider_type = OpenAICompatibleProvider
        provider = provider_type(
            name=profile.name, api_key=profile.api_key, base_url=profile.base_url
        )
        if profile.api == "anthropic":
            return AnthropicProvider(
                name=profile.name,
                api_key=profile.api_key,
                base_url=profile.base_url,
                model_settings=provider,
            )
        return provider
    raise ConfigError(f"未知的 api 类型 {profile.api!r}（档案 {profile.name!r}）")
