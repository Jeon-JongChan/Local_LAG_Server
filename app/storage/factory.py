"""저장소(DB) 어댑터 팩토리 모듈.
"""

from __future__ import annotations

from typing import Optional
from .base import BaseChatStore, BaseChunkStore, BaseVectorStore
from .faiss_vector import FaissVectorStore
from .rag_storage import RagStorage
from .sqlite_chat import SqliteChatStore
from .sqlite_chunk import SqliteChunkStore

def get_chat_store(db_path: Optional[str] = None) -> BaseChatStore:
    """대화 세션 저장소 어댑터 인스턴스 반환 (멀티스레드 및 close() 안전성 보장)."""
    if db_path is not None:
        return SqliteChatStore(db_path=db_path)
    return SqliteChatStore()


def get_vector_store(index_dir: Optional[str] = None) -> BaseVectorStore:
    """벡터 인덱스 어댑터 생성."""
    if index_dir is not None:
        return FaissVectorStore(index_dir=index_dir)
    return FaissVectorStore()


def get_chunk_store(db_path: Optional[str] = None) -> BaseChunkStore:
    """청크/FTS5 저장소 어댑터 생성."""
    if db_path is not None:
        return SqliteChunkStore(db_path=db_path)
    return SqliteChunkStore()


def get_rag_storage() -> RagStorage:
    """RAG 하이브리드 통합 스토리지 인스턴스 생성."""
    return RagStorage(
        vector_store=get_vector_store(),
        chunk_store=get_chunk_store(),
    )

