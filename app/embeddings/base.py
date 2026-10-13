"""
임베딩 어댑터 베이스 모듈.

템플릿 메서드 패턴(Template Method Pattern)을 적용하여
입력 유효성 검증, 쿼리/문서 분기, 배치 처리 및 진행률 통지 공통 흐름을 제어하고,
실제 벡터 인코딩은 서브클래스에서 구체화합니다.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Sequence
import numpy as np


class BaseEmbeddingAdapter(ABC):
    """임베딩 추상 베이스 어댑터 클래스."""

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        """
        초기화 메서드.

        Args:
            model_name: 모델 리포지토리 ID 또는 로컬 경로
            device: 실행 디바이스 (cpu, cuda 등)
        """
        self.model_name = model_name
        self.device = device
        self._dimension: Optional[int] = None

    @property
    def dimension(self) -> int:
        """임베딩 벡터 차원 수 반환."""
        if self._dimension is None:
            # 1회 임베딩을 통해 차원 수 자동 파악
            sample = self.embed_query("ping")
            self._dimension = int(sample.shape[0])
        return self._dimension

    def embed_query(self, query: str) -> np.ndarray:
        """
        [템플릿 메서드] 단일 검색 질문(Query)을 임베딩합니다.

        공통 전처리 -> 구체 서브클래스 인코딩 -> 후처리를 수행합니다.
        """
        cleaned = (query or "").strip()
        if not cleaned:
            # 빈 질문일 경우 0벡터 반환
            dim = self._dimension or 1024
            return np.zeros((dim,), dtype=np.float32)

        vec = self._embed_single_query(cleaned)
        return np.asarray(vec, dtype=np.float32)

    def embed_documents(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = 16,
        on_progress: Optional[Callable[[int, int, str], None]] = None,
    ) -> np.ndarray:
        """
        [템플릿 메서드] 복수 문서 청크(Passages)를 배치 단위로 임베딩합니다.

        공통 배치 분할 루프 -> 서브클래스 배치 인코딩 -> 진행률 콜백 -> 결합을 수행합니다.
        """
        if not texts:
            dim = self._dimension or 1024
            return np.zeros((0, dim), dtype=np.float32)

        total = len(texts)
        parts: List[np.ndarray] = []

        for start in range(0, total, batch_size):
            batch = list(texts[start : start + batch_size])
            # 서브클래스 배치 인코딩 호출
            batch_vecs = self._embed_batch_documents(batch)
            parts.append(np.asarray(batch_vecs, dtype=np.float32))

            done = min(start + len(batch), total)
            if on_progress:
                on_progress(done, total, f"임베딩 진행 중: {done}/{total}")

        if len(parts) == 1:
            return parts[0]
        return np.vstack(parts)

    @abstractmethod
    def _embed_single_query(self, query: str) -> np.ndarray:
        """구체 서브클래스에서 구현할 단일 질의 인코딩 핵심 로직."""
        pass

    @abstractmethod
    def _embed_batch_documents(self, documents: List[str]) -> np.ndarray:
        """구체 서브클래스에서 구현할 복수 문서 배치 인코딩 핵심 로직."""
        pass

