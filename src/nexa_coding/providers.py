"""把已验证的供应商档案转换成模型 Provider。"""

from urllib.parse import urlparse

from nexa_agent.provider import ModelProvider
from nexa_ai.deepseek import DeepSeekProvider
from nexa_ai.openai_compatible import OpenAICompatibleProvider
from nexa_coding.config import ConfigError, ProviderProfile


def build_provider(profile: ProviderProfile) -> ModelProvider:
    if profile.api == "openai":
        provider_type = (
            DeepSeekProvider
            if urlparse(profile.base_url).hostname == "api.deepseek.com"
            or profile.name == "deepseek"
            else OpenAICompatibleProvider
        )
        return provider_type(name=profile.name, api_key=profile.api_key, base_url=profile.base_url)
    raise ConfigError(f"未知的 api 类型 {profile.api!r}（档案 {profile.name!r}）")
