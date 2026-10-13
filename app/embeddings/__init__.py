"""임베딩 패키지."""

from .base import BaseEmbeddingAdapter
from .factory import get_embedding_adapter

__all__ = ["BaseEmbeddingAdapter", "get_embedding_adapter"]

