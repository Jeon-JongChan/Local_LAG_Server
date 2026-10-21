"""storage 패키지."""

from __future__ import annotations

from typing import Optional

from .base import (
    BaseChatStore,
    BaseChunkStore,
    BaseVectorStore,
)
from .faiss_vector import FaissVectorStore
from .rag_storage import RagStorage, rrf_fuse
from .sqlite_chat import SqliteChatStore
from .sqlite_chunk import SqliteChunkStore


def get_chat_store(db_path: Optional[str] = None) -> SqliteChatStore:
    """대화 세션 저장소 인스턴스."""
    return SqliteChatStore(db_path=db_path) if db_path is not None else SqliteChatStore()


def get_vector_store(index_path: Optional[str] = None) -> FaissVectorStore:
    """벡터 인덱스 인스턴스."""
    return FaissVectorStore(index_path=index_path) if index_path is not None else FaissVectorStore()


def get_chunk_store(db_path: Optional[str] = None) -> SqliteChunkStore:
    """청크/FTS5 저장소 인스턴스."""
    return SqliteChunkStore(db_path=db_path) if db_path is not None else SqliteChunkStore()


def get_rag_storage() -> RagStorage:
    """RAG 하이브리드 스토리지."""
    return RagStorage(
        vector_store=get_vector_store(),
        chunk_store=get_chunk_store(),
    )


__all__ = [
    "BaseChatStore",
    "BaseVectorStore",
    "BaseChunkStore",
    "FaissVectorStore",
    "SqliteChunkStore",
    "SqliteChatStore",
    "RagStorage",
    "rrf_fuse",
    "get_chat_store",
    "get_vector_store",
    "get_chunk_store",
    "get_rag_storage",
]
