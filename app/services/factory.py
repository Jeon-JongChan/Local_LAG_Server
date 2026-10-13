"""
서비스 관리 어댑터 팩토리 모듈.
"""

from __future__ import annotations

from typing import Optional
from .base import BaseEngineManager, BaseModelManager
from .hf_model_manager import HuggingFaceModelManager
from .llama_engine_manager import LlamaEngineManager

_active_model_manager: Optional[BaseModelManager] = None
_active_engine_manager: Optional[BaseEngineManager] = None


def get_model_manager(provider: str = "huggingface") -> BaseModelManager:
    """모델 다운로드 관리자 어댑터 팩토리 함수."""
    global _active_model_manager
    if _active_model_manager is None:
        _active_model_manager = HuggingFaceModelManager()
    return _active_model_manager


def get_engine_manager(engine_name: str = "llama-server") -> BaseEngineManager:
    """서빙 엔진 런타임 관리자 어댑터 팩토리 함수."""
    global _active_engine_manager
    if _active_engine_manager is None:
        _active_engine_manager = LlamaEngineManager()
    return _active_engine_manager
