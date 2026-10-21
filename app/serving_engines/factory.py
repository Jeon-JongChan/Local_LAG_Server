"""서빙 엔진 팩토리.

Strategy: SERVING_ENGINE 이름으로 구현 선택.
Adapter: 외부 HTTP API 를 BaseServingAdapter 로 맞춤.
OpenAI 호환은 프로필 dict, API가 다른 Ollama만 별도 클래스.
설정은 SettingsFacade 경유 상수만 사용한다.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .base import BaseServingAdapter
from .ollama import OllamaServingAdapter
from .openai_compatible import OpenAICompatibleServingAdapter
from settings import (
    EXLLAMA_API_KEY,
    EXLLAMA_BASE_URL,
    EXLLAMA_MODEL,
    LLAMA_CPP_API_KEY,
    LLAMA_CPP_BASE_URL,
    LLAMA_CPP_MODEL,
    LLAMAFILE_API_KEY,
    LLAMAFILE_BASE_URL,
    LLAMAFILE_MODEL,
    MLC_LLM_API_KEY,
    MLC_LLM_BASE_URL,
    MLC_LLM_MODEL,
    REQUEST_TIMEOUT,
    SERVING_ENGINE,
    SGLANG_API_KEY,
    SGLANG_BASE_URL,
    SGLANG_MODEL,
    VLLM_API_KEY,
    VLLM_BASE_URL,
    VLLM_MODEL,
)

_ALIASES: Dict[str, str] = {
    "llama.cpp": "llama.cpp",
    "llama_cpp": "llama.cpp",
    "llamacpp": "llama.cpp",
    "ollama": "ollama",
    "vllm": "vllm",
    "sglang": "sglang",
    "exllamav3": "exllamav3",
    "exllama": "exllamav3",
    "exllama_v3": "exllamav3",
    "tabbyapi": "exllamav3",
    "mlc_llm": "mlc_llm",
    "mlc": "mlc_llm",
    "mlcllm": "mlc_llm",
    "llamafile": "llamafile",
    "mozilla_llamafile": "llamafile",
    "openai": "openai",
    "openai_compatible": "openai",
    "airllm": "airllm",
    "air_llm": "airllm",
    "air-llm": "airllm",
}

# OpenAI 호환 엔진 프로필 (settings 상수 — 파사드 env>preset>default 반영)
_OPENAI_PROFILES: Dict[str, Dict[str, Any]] = {
    "vllm": {
        "base_url": VLLM_BASE_URL,
        "model_name": VLLM_MODEL,
        "api_key": VLLM_API_KEY,
        "health_path": "/health",
        "extra": {"best_of": 1, "ignore_eos": False, "repetition_penalty": 1.05},
    },
    "sglang": {
        "base_url": SGLANG_BASE_URL,
        "model_name": SGLANG_MODEL,
        "api_key": SGLANG_API_KEY,
        "health_path": "/health",
        "extra": {"skip_special_tokens": True, "top_p": 0.9},
    },
    "exllamav3": {
        "base_url": EXLLAMA_BASE_URL,
        "model_name": EXLLAMA_MODEL,
        "api_key": EXLLAMA_API_KEY,
        "health_path": "/health",
        "extra": {"min_p": 0.05, "top_p": 0.9},
    },
    "mlc_llm": {
        "base_url": MLC_LLM_BASE_URL,
        "model_name": MLC_LLM_MODEL,
        "api_key": MLC_LLM_API_KEY,
        "health_path": "/health",
        "extra": {
            "repetition_penalty": 1.05,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
        },
    },
    "llamafile": {
        "base_url": LLAMAFILE_BASE_URL,
        "model_name": LLAMAFILE_MODEL,
        "api_key": LLAMAFILE_API_KEY,
        "health_path": "/health",
        "extra": {"cache_prompt": True},
    },
}

_active_engines: Dict[str, BaseServingAdapter] = {}


def canonical_engine(engine_type: Optional[str] = None) -> str:
    """엔진 이름 정규화. 미지정 시 settings.SERVING_ENGINE."""
    raw = (engine_type or SERVING_ENGINE or "llama.cpp").strip().lower()
    return _ALIASES.get(raw, raw)


def _build_openai_profile(name: str, profile: Dict[str, Any]) -> OpenAICompatibleServingAdapter:
    """프로필 dict로 OpenAI 호환 어댑터 생성."""
    return OpenAICompatibleServingAdapter(
        base_url=profile["base_url"],
        model_name=profile.get("model_name"),
        api_key=profile.get("api_key") or "no-key",
        timeout=REQUEST_TIMEOUT,
        extra_options=profile.get("extra"),
        health_path=profile.get("health_path", "/health"),
        engine_type=name,
    )


def _build(engine: str) -> BaseServingAdapter:
    """정규화된 엔진 이름으로 어댑터 인스턴스 생성."""
    if engine == "airllm":
        from .airllm import AirLLMServingAdapter
        return AirLLMServingAdapter()

    if engine == "ollama":
        adapter = OllamaServingAdapter()
        adapter.engine_type = "ollama"
        return adapter

    if engine == "llama.cpp":
        return OpenAICompatibleServingAdapter(
            base_url=LLAMA_CPP_BASE_URL,
            model_name=LLAMA_CPP_MODEL,
            api_key=LLAMA_CPP_API_KEY,
            timeout=REQUEST_TIMEOUT,
            extra_options={"cache_prompt": True},
            health_path="/health",
            health_ok_statuses=("ok",),
            engine_type="llama.cpp",
        )

    if engine in _OPENAI_PROFILES:
        return _build_openai_profile(engine, _OPENAI_PROFILES[engine])

    return OpenAICompatibleServingAdapter(
        base_url=LLAMA_CPP_BASE_URL,
        model_name=LLAMA_CPP_MODEL,
        api_key=LLAMA_CPP_API_KEY,
        timeout=REQUEST_TIMEOUT,
        engine_type="openai",
    )


def get_serving_adapter(engine_type: Optional[str] = None) -> BaseServingAdapter:
    """서빙 엔진 어댑터 팩토리. 엔진별 인스턴스를 캐시한다."""
    target = canonical_engine(engine_type)
    if target in _active_engines:
        return _active_engines[target]
    adapter = _build(target)
    _active_engines[target] = adapter
    return adapter


def reset_serving_adapters() -> None:
    """테스트용 캐시 리셋."""
    _active_engines.clear()
