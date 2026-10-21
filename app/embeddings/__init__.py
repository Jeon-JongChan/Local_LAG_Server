"""임베딩 패키지.

Qwen, Gemma, BGE 등 다양한 임베딩 모델을 단일 EmbeddingAdapter로 심플하게 제공합니다.
"""

from __future__ import annotations

from typing import Optional

from .base import BaseEmbeddingAdapter, EmbeddingAdapter, ensure_gguf_model, is_gguf_model
from settings import EMBED_DEVICE, EMBED_MODEL

_active_adapter: Optional[EmbeddingAdapter] = None


def _resolve_device() -> str:
    """ACCELERATOR 기준 디바이스 결정 (cuda / cpu)."""
    from settings import ACCELERATOR, EMBED_DEVICE

    if EMBED_DEVICE not in ("auto", ""):
        return EMBED_DEVICE
    if ACCELERATOR == "cuda":
        return "cuda"
    if ACCELERATOR == "cpu":
        return "cpu"
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def get_embedding_adapter(
    model_name: Optional[str] = None,
    device: Optional[str] = None,
) -> EmbeddingAdapter:
    """임베딩 어댑터 싱글톤 인스턴스를 반환합니다."""
    global _active_adapter

    target_model = model_name or EMBED_MODEL
    target_device = device or _resolve_device()

    if _active_adapter is not None:
        if (
            _active_adapter.model_name == target_model
            and _active_adapter.device == target_device
        ):
            return _active_adapter
        reset_embedding_adapter()

    _active_adapter = EmbeddingAdapter(model_name=target_model, device=target_device)
    return _active_adapter


def reset_embedding_adapter() -> None:
    """활성 임베딩 어댑터 리소스를 정리하고 싱글톤을 초기화합니다."""
    global _active_adapter
    if _active_adapter is not None:
        _active_adapter.close()
        _active_adapter = None


__all__ = [
    "BaseEmbeddingAdapter",
    "EmbeddingAdapter",
    "get_embedding_adapter",
    "reset_embedding_adapter",
    "is_gguf_model",
    "ensure_gguf_model",
]
