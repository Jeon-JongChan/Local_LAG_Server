"""서빙 엔진 어댑터 팩토리 모듈.

엔진 종류('llama.cpp', 'ollama', 'vllm', 'sglang', 'exllamav3', 'mlc_llm', 'llamafile', 'openai' 등)에 따라
적합한 BaseServingAdapter 인스턴스를 반환하고 캐싱합니다.
"""

from __future__ import annotations

import os
from typing import Dict, Optional
from .base import BaseServingAdapter
from .exllamav3 import ExLlamaV3ServingAdapter
from .llama_cpp import LlamaCppServingAdapter
from .llamafile import LlamafileServingAdapter
from .mlc_llm import MlcLlmServingAdapter
from .ollama import OllamaServingAdapter
from .openai_compatible import OpenAICompatibleServingAdapter
from .sglang import SglangServingAdapter
from .vllm import VllmServingAdapter

try:
    from settings import LLAMA_CPP_BASE_URL
except ModuleNotFoundError:
    from app.settings import LLAMA_CPP_BASE_URL

_active_engines: Dict[str, BaseServingAdapter] = {}


def get_serving_adapter(engine_type: Optional[str] = None) -> BaseServingAdapter:
    """
    서빙 엔진 어댑터 팩토리 함수.

    Args:
        engine_type: 'llama.cpp' | 'ollama' | 'vllm' | 'sglang' | 'exllamav3' | 'mlc_llm' | 'llamafile' | 'openai'
                     (기본값: 환경변수 SERVING_ENGINE 또는 'llama.cpp')

    Returns:
        BaseServingAdapter 인스턴스
    """
    global _active_engines

    target_engine = (engine_type or os.getenv("SERVING_ENGINE", "llama.cpp")).strip().lower()

    if target_engine in _active_engines:
        return _active_engines[target_engine]

    adapter: BaseServingAdapter
    if target_engine in ("llama.cpp", "llama_cpp", "llamacpp"):
        adapter = LlamaCppServingAdapter()
    elif target_engine in ("ollama",):
        adapter = OllamaServingAdapter()
    elif target_engine in ("vllm",):
        adapter = VllmServingAdapter()
    elif target_engine in ("sglang",):
        adapter = SglangServingAdapter()
    elif target_engine in ("exllamav3", "exllama", "exllama_v3", "tabbyapi"):
        adapter = ExLlamaV3ServingAdapter()
    elif target_engine in ("mlc_llm", "mlc", "mlcllm"):
        adapter = MlcLlmServingAdapter()
    elif target_engine in ("llamafile", "mozilla_llamafile"):
        adapter = LlamafileServingAdapter()
    elif target_engine in ("openai", "openai_compatible"):
        adapter = OpenAICompatibleServingAdapter(base_url=LLAMA_CPP_BASE_URL)
    else:
        # 기타 임의의 OpenAI 규격 호환 백엔드 fallback
        adapter = OpenAICompatibleServingAdapter(base_url=LLAMA_CPP_BASE_URL)

    _active_engines[target_engine] = adapter
    return adapter


def reset_serving_adapters() -> None:
    """테스트용 캐시 리셋 함수."""
    global _active_engines
    _active_engines.clear()
