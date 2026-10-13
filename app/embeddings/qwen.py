"""
Qwen 계열 문장 임베딩 어댑터 구현체.

SentenceTransformer 라이브러리를 기반으로 로컬 모델 가중치를 로드하고 추론합니다.
Qwen3-Embedding 특화 left-padding 및 질의용 'query' 프롬프트를 자동 적용합니다.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional
import numpy as np

from .base import BaseEmbeddingAdapter

try:
    from settings import EMBED_MODELS_DIR, hf_model_dir
except ModuleNotFoundError:
    from app.settings import EMBED_MODELS_DIR, hf_model_dir


class QwenEmbeddingAdapter(BaseEmbeddingAdapter):
    """Qwen 임베딩 모델 어댑터 (SentenceTransformers 기반)."""

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        super().__init__(model_name=model_name, device=device)
        self._model = None
        self._is_qwen = "qwen" in model_name.lower()

    def _ensure_loaded(self):
        """SentenceTransformer 모델 로드 (지연 로딩)."""
        if self._model is not None:
            return self._model

        from sentence_transformers import SentenceTransformer
        # 로컬 다운로드 디렉터리 확인
        local_path = hf_model_dir(self.model_name, EMBED_MODELS_DIR)
        load_target = str(local_path) if local_path.is_dir() else self.model_name

        kwargs = {"device": self.device}
        if self._is_qwen:
            kwargs["tokenizer_kwargs"] = {"padding_side": "left"}

        print(f"[embedding] 모델 로드 중: {self.model_name} ({self.device})", flush=True)
        self._model = SentenceTransformer(load_target, **kwargs)
        return self._model

    def _embed_single_query(self, query: str) -> np.ndarray:
        """단일 질의 인코딩 (query 프롬프트 적용)."""
        model = self._ensure_loaded()
        kwargs = {
            "normalize_embeddings": True,
            "show_progress_bar": False,
            "convert_to_numpy": True,
        }
        if self._is_qwen:
            kwargs["prompt_name"] = "query"

        vec = model.encode([query], **kwargs)[0]
        return np.asarray(vec, dtype=np.float32)

    def _embed_batch_documents(self, documents: List[str]) -> np.ndarray:
        """복수 문서 배치 인코딩 (문서는 프롬프트 미적용)."""
        model = self._ensure_loaded()
        kwargs = {
            "normalize_embeddings": True,
            "show_progress_bar": False,
            "convert_to_numpy": True,
        }
        vecs = model.encode(documents, **kwargs)
        return np.asarray(vecs, dtype=np.float32)
