"""
리랭커 추상 베이스 어댑터 모듈.

템플릿 메서드 패턴(Template Method Pattern)을 적용하여
후보 문서 검증, 엣지 케이스 처리, 안정 정렬 및 출처 메타데이터 보존 공통 흐름을 제어하고,
실제 연관도 채점 로직(_calculate_scores)은 서브클래스에서 구체화합니다.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Sequence


class RerankerError(Exception):
    """리랭킹 처리 중 발생하는 기본 예외."""
    pass


class RerankerUnavailableError(RerankerError):
    """지정된 리랭커 백엔드를 사용할 수 없을 때 발생하는 예외 (미설치, 서버 오프라인 등)."""
    pass


class BaseRerankerAdapter(ABC):
    """리랭커 추상 베이스 어댑터 클래스."""

    backend_name: str = "base"

    def rerank(
        self,
        query: str,
        candidates: Sequence[Dict[str, Any]],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        [템플릿 메서드] 후보 문서들을 질문과의 연관성 점수로 정렬하여 상위 top_k개를 반환합니다.

        공통 엣지 케이스 검증 -> 서브클래스 채점 -> 점수 기준 정렬 -> 메타데이터 보존 주입을 수행합니다.
        """
        # 1. 엣지 케이스 공통 처리
        if not candidates:
            return []
        if top_k <= 0:
            return []
        if len(candidates) == 1:
            enriched = dict(candidates[0])
            enriched.setdefault("rerank_score", 1.0)
            enriched["rerank_rank"] = 1
            enriched["rerank_backend"] = self.backend_name
            return [enriched]

        # 2. 서브클래스 채점 훅 호출
        scores = self._calculate_scores(query, candidates)

        # 3. 점수 내림차순 안정 정렬 (동점 시 원래 순서 보존)
        scored_pairs = list(zip(candidates, scores))
        ranked = sorted(
            scored_pairs,
            key=lambda x: float(x[1]),
            reverse=True,
        )

        # 4. 공통 메타데이터 주입 및 Top-K 슬라이싱
        out: List[Dict[str, Any]] = []
        for rank_idx, (item, score) in enumerate(ranked[:top_k], start=1):
            enriched = dict(item)
            enriched["rerank_score"] = float(score)
            enriched["rerank_rank"] = rank_idx
            enriched["rerank_backend"] = self.backend_name
            out.append(enriched)

        return out

    @abstractmethod
    def _calculate_scores(
        self,
        query: str,
        candidates: Sequence[Dict[str, Any]],
    ) -> List[float]:
        """구체 서브클래스에서 구현할 질의-후보쌍 점수 계산 로직."""
        pass

    def close(self) -> None:
        """리소스 정리 훅 (필요 시 오버라이드)."""
        pass
