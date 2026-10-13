"""임베딩 어댑터 팩토리 모듈.

설정값에 따라 적합한 임베딩 어댑터 인스턴스를 생성하고 단일 인스턴스를 캐싱합니다.
"""

from __future__ import annotations

from typing import Optional
from .base import BaseEmbeddingAdapter

try:
    from settings import EMBED_DEVICE, EMBED_MODEL
except ModuleNotFoundError:
    from app.settings import EMBED_DEVICE, EMBED_MODEL

_active_adapter: Optional[BaseEmbeddingAdapter] = None


def _resolve_device() -> str:
    """torch 디바이스 자동 감지 (cuda / cpu)."""
    if EMBED_DEVICE not in ("auto", ""):
        return EMBED_DEVICE
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def get_embedding_adapter(
    model_name: Optional[str] = None,
    device: Optional[str] = None,
) -> BaseEmbeddingAdapter:
    """
    임베딩 어댑터 팩토리 함수.

    Args:
        model_name: 대상 임베딩 모델 ID (기본: settings.EMBED_MODEL)
        device: 실행 디바이스 (기본: 자동 감지)

    Returns:
        BaseEmbeddingAdapter 서브클래스 인스턴스
    """
    global _active_adapter

    target_model = model_name or EMBED_MODEL
    target_device = device or _resolve_device()

    if _active_adapter is not None:
        if _active_adapter.model_name == target_model and _active_adapter.device == target_device:
            return _active_adapter

    # 모델명에 따른 구체 어댑터 분기 (향후 OpenAI, Cohere, BGE 등 확장 용이)
    from .qwen import QwenEmbeddingAdapter
    _active_adapter = QwenEmbeddingAdapter(model_name=target_model, device=target_device)
    return _active_adapter

