"""
저장소(DB) 추상 베이스 어댑터 모듈.

대화 세션 저장소(ChatStore), 벡터 인덱스 저장소(VectorStore), 청크 본문/전문검색 저장소(ChunkStore)
각각의 추상 인터페이스 및 데이터 규약을 정의합니다.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class BaseChatStore(ABC):
    """대화 세션 및 메시지 저장소 추상 인터페이스."""

    @abstractmethod
    def list_sessions(self) -> List[Dict[str, Any]]:
        """모든 대화 세션 목록 조회."""
        pass

    @abstractmethod
    def create_session(self, title: str = "새 대화", model: str = "") -> Dict[str, Any]:
        """새 세션 생성 및 반환."""
        pass

    @abstractmethod
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """세션 메타데이터 및 전체 메시지 내역 조회."""
        pass

    @abstractmethod
    def delete_session(self, session_id: str) -> bool:
        """세션 삭제."""
        pass

    @abstractmethod
    def list_messages(self, session_id: str) -> List[Dict[str, Any]]:
        """세션의 모든 메시지 목록 조회."""
        pass

    @abstractmethod
    def add_message(self, session_id: str, role: str, content: str) -> Dict[str, Any]:
        """세션에 메시지 추가."""
        pass

    @abstractmethod
    def close(self) -> None:
        """연결 해제."""
        pass


class BaseVectorStore(ABC):
    """벡터 인덱스(Dense) 저장소 추상 인터페이스."""

    @abstractmethod
    def add(self, ids: List[int], vectors: np.ndarray) -> None:
        """벡터 및 해당 ID 추가."""
        pass

    @abstractmethod
    def search(self, query_vec: np.ndarray, top_k: int = 8) -> List[Tuple[int, float]]:
        """질의 벡터와 유사한 상위 청크 (id, 유사도_점수) 목록 반환."""
        pass

    @abstractmethod
    def count(self) -> int:
        """색인된 총 벡터 수."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """인덱스 초기화."""
        pass

    @abstractmethod
    def save(self) -> None:
        """디스크 영구 저장."""
        pass

    @abstractmethod
    def close(self) -> None:
        """리소스 정리."""
        pass


class BaseChunkStore(ABC):
    """청크 본문, 메타데이터 및 BM25 키워드 검색 저장소 추상 인터페이스."""

    @abstractmethod
    def insert_chunk(
        self,
        chunk_id: int,
        file_name: str,
        source_path: str,
        page: Optional[int],
        bbox: Optional[str],
        element_type: Optional[str],
        text: str,
    ) -> None:
        """청크 본문 및 메타데이터 저장."""
        pass

    @abstractmethod
    def get_chunks(self, ids: List[int]) -> Dict[int, Dict[str, Any]]:
        """ID 목록으로 청크 딕셔너리 일괄 조회."""
        pass

    @abstractmethod
    def search_bm25(self, query: str, top_k: int = 16) -> List[Dict[str, Any]]:
        """키워드 기반 BM25 전문 검색."""
        pass

    @abstractmethod
    def is_ingested(self, rel_path: str, size: int, mtime: float) -> bool:
        """동일 파일(크기·수정일) 색인 여부 확인."""
        pass

    @abstractmethod
    def update_manifest(self, rel_path: str, size: int, mtime: float) -> None:
        """인제스트 매니페스트 갱신."""
        pass

    @abstractmethod
    def count(self) -> int:
        """저장된 총 청크 수."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """테이블 초기화."""
        pass

    @abstractmethod
    def close(self) -> None:
        """DB 연결 해제."""
        pass
