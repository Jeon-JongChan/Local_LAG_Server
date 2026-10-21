"""RAG 통합 스토리지 Facade.

VectorStore(FAISS) + ChunkStore(SQLite FTS5) 서브시스템을 감춰
search / search_bm25 / add_chunks 단일 인터페이스를 제공한다.
RRF 융합(rrf_fuse)은 순수 함수로 파사드 옆에 둔다.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np

from .base import BaseChunkStore, BaseVectorStore
from .faiss_vector import FaissVectorStore
from .sqlite_chunk import SqliteChunkStore


def rrf_fuse(
    dense_hits: List[Dict[str, Any]],
    bm25_hits: List[Dict[str, Any]],
    *,
    k: int = 60,
    top_n: int = 8,
) -> List[Dict[str, Any]]:
    """Reciprocal Rank Fusion (RRF) 점수 융합."""
    rrf_scores: Dict[int, float] = {}
    doc_map: Dict[int, Dict[str, Any]] = {}

    for rank, doc in enumerate(dense_hits, start=1):
        did = doc["id"]
        doc_map[did] = doc
        rrf_scores[did] = rrf_scores.get(did, 0.0) + 1.0 / (k + rank)

    for rank, doc in enumerate(bm25_hits, start=1):
        did = doc["id"]
        if did not in doc_map:
            doc_map[did] = doc
        rrf_scores[did] = rrf_scores.get(did, 0.0) + 1.0 / (k + rank)

    sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
    out: List[Dict[str, Any]] = []
    for did in sorted_ids[:top_n]:
        item = dict(doc_map[did])
        item["score"] = rrf_scores[did]
        out.append(item)
    return out


class RagStorage:
    """FAISS 벡터 스토어와 SQLite 청크 스토어의 결합 파사드."""

    def __init__(
        self,
        vector_store: Optional[BaseVectorStore] = None,
        chunk_store: Optional[BaseChunkStore] = None,
    ) -> None:
        self.vector_store = vector_store or FaissVectorStore()
        self.chunk_store = chunk_store or SqliteChunkStore()

    def count(self) -> int:
        return self.chunk_store.count()

    def search(self, query_vec: np.ndarray, top_k: int = 8) -> List[Dict[str, Any]]:
        """Dense 유사도 검색 후 SQLite 메타데이터를 결합하여 반환."""
        pairs = self.vector_store.search(query_vec, top_k=top_k)
        if not pairs:
            return []

        ids = [p[0] for p in pairs]
        chunks = self.chunk_store.get_chunks(ids)

        out: List[Dict[str, Any]] = []
        for cid, score in pairs:
            if cid in chunks:
                item = dict(chunks[cid])
                item["score"] = score
                out.append(item)
        return out

    def search_bm25(self, query: str, top_k: int = 16) -> List[Dict[str, Any]]:
        """BM25 키워드 전문 검색."""
        return self.chunk_store.search_bm25(query, top_k=top_k)

    def add_chunks(self, chunks: List[Dict[str, Any]], vectors: np.ndarray) -> None:
        """인제스트 시 복수 청크 및 임베딩 벡터 저장."""
        if not chunks:
            return
        start_id = self.count()
        ids = list(range(start_id, start_id + len(chunks)))

        for cid, c in zip(ids, chunks):
            self.chunk_store.insert_chunk(
                chunk_id=cid,
                file_name=c["file_name"],
                source_path=c["source_path"],
                page=c.get("page"),
                bbox=c.get("bbox"),
                element_type=c.get("element_type"),
                text=c["text"],
            )

        self.vector_store.add(ids, vectors)
        self.vector_store.save()

    def is_ingested(self, rel_path: str, size: int, mtime: float) -> bool:
        return self.chunk_store.is_ingested(rel_path, size, mtime)

    def update_manifest(self, rel_path: str, size: int, mtime: float) -> None:
        self.chunk_store.update_manifest(rel_path, size, mtime)

    def reset(self) -> None:
        self.chunk_store.reset()
        self.vector_store.reset()

    def close(self) -> None:
        self.vector_store.close()
        self.chunk_store.close()
