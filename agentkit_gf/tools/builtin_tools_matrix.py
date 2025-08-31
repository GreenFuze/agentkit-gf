from __future__ import annotations
from enum import Enum
from typing import Iterable, Mapping

class BuiltinTool(Enum):
    WEB_SEARCH = "web_search"
    CODE_EXECUTION = "code_execution"
    URL_CONTEXT = "url_context"

# Normalize model prefixes → provider key
PROVIDER_ALIASES: Mapping[str, str] = {
    "openai": "openai",
    "anthropic": "anthropic",
    "groq": "groq",
    "google": "google",
    "google-gla": "google",
    "google-vertex": "google",
    "bedrock": "bedrock",
    "mistral": "mistral",
    "cohere": "cohere",
    "huggingface": "huggingface",
}

# Provider → supported builtin tools (adjust to your truth source)
SUPPORTED_BY_PROVIDER: Mapping[str, set[BuiltinTool]] = {
    "openai": {BuiltinTool.WEB_SEARCH, BuiltinTool.CODE_EXECUTION},
    "anthropic": {BuiltinTool.WEB_SEARCH, BuiltinTool.CODE_EXECUTION},
    "groq": {BuiltinTool.WEB_SEARCH},
    "google": {BuiltinTool.WEB_SEARCH, BuiltinTool.CODE_EXECUTION, BuiltinTool.URL_CONTEXT},
    "bedrock": set(),
    "mistral": set(),
    "cohere": set(),
    "huggingface": set(),
}

def provider_key_from_model(model: str) -> str:
    if not model or ":" not in model:
        raise ValueError(f"Invalid model string: {model!r} (expected '<provider>:<model>')")
    prefix = model.split(":", 1)[0].lower()
    provider = PROVIDER_ALIASES.get(prefix, prefix)

    if provider not in SUPPORTED_BY_PROVIDER:
        raise ValueError(f"Unknown provider '{provider}' (from model {model!r})")

    return provider

def validate_builtin_tools(provider_key: str, requested: Iterable[BuiltinTool]) -> list[BuiltinTool]:
    requested_list = list(requested)

    supported = SUPPORTED_BY_PROVIDER[provider_key]
    bad = [t for t in requested_list if t not in supported]
    if bad:
        raise ValueError(
            f"Provider '{provider_key}' does not support: {[t.value for t in bad]}. "
            f"Supported: {[t.value for t in sorted(supported, key=lambda x: x.value)]}"
        )

    return requested_list
