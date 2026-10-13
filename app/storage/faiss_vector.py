"""
FAISS 기반 벡터 인덱스 저장소 어댑터 구현체.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple
import faiss
import numpy as np

from .base import BaseVectorStore

try:
    from settings import FAISS_PATH, INDEX_DIR
except ModuleNotFoundError:
    from app.settings import FAISS_PATH, INDEX_DIR


class FaissVectorStore(BaseVectorStore):
    """FAISS (IndexFlatIP) 기반 Dense 벡터 저장소 어댑터."""

    def __init__(self, index_path: Path = FAISS_PATH) -> None:
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        self.index_path = index_path
        self._index: Optional[faiss.Index] = None
        self._dim: Optional[int] = None
        self._load_if_exists()

    def _load_if_exists(self) -> None:
        if self.index_path.is_file():
            self._index = faiss.read_index(str(self.index_path))
            self._dim = self._index.d

    def _init_index(self, dim: int) -> None:
        self._dim = dim
        self._index = faiss.IndexFlatIP(dim)

    def add(self, ids: List[int], vectors: np.ndarray) -> None:
        """벡터 L2 정규화 후 인덱스에 추가 (ids는 순차 매핑)."""
        if len(ids) == 0 or vectors.size == 0:
            return

        mat = np.ascontiguousarray(vectors, dtype=np.float32)
        if mat.ndim == 1:
            mat = mat.reshape(1, -1)

        dim = mat.shape[1]
        if self._index is None:
            self._init_index(dim)

        faiss.normalize_L2(mat)
        self._index.add(mat)

    def search(self, query_vec: np.ndarray, top_k: int = 8) -> List[Tuple[int, float]]:
        """질의 벡터와 유사한 상위 (인덱스_id, 코사인_유사도) 반환."""
        if self._index is None or self._index.ntotal == 0:
            return []

        q = np.ascontiguousarray(query_vec, dtype=np.float32)
        if q.ndim == 1:
            q = q.reshape(1, -1)
        faiss.normalize_L2(q)

        k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(q, k)

        out: List[Tuple[int, float]] = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < 0:
                continue
            out.append((int(idx), float(score)))
        return out

    def count(self) -> int:
        return self._index.ntotal if self._index is not None else 0

    def reset(self) -> None:
        self._index = None
        self._dim = None
        if self.index_path.is_file():
            self.index_path.unlink()

    def save(self) -> None:
        if self._index is not None:
            faiss.write_index(self._index, str(self.index_path))

    def close(self) -> None:
        self.save()
        self._index = None
