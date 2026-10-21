"""서빙 엔진 패키지.

Strategy(엔진 선택) + Adapter(외부 HTTP API 래핑) + Template(generate).
"""

from .airllm import AirLLMServingAdapter
from .base import BaseServingAdapter
from .factory import canonical_engine, get_serving_adapter, reset_serving_adapters
from .ollama import OllamaServingAdapter
from .openai_compatible import OpenAICompatibleServingAdapter

__all__ = [
    "AirLLMServingAdapter",
    "BaseServingAdapter",
    "canonical_engine",
    "get_serving_adapter",
    "reset_serving_adapters",
    "OllamaServingAdapter",
    "OpenAICompatibleServingAdapter",
]
