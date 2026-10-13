"""
clm 리랭커 어댑터 구현체 (Contrastive-LM REST API).
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence
import requests

from .base import BaseRerankerAdapter, RerankerUnavailableError

try:
    from settings import (
        RERANK_CLM_ACTION_CACHE,
        RERANK_CLM_BASE_URL,
        RERANK_CLM_TIMEOUT,
    )
except ModuleNotFoundError:
    from app.settings import (
        RERANK_CLM_ACTION_CACHE,
        RERANK_CLM_BASE_URL,
        RERANK_CLM_TIMEOUT,
    )


class CLMRerankerAdapter(BaseRerankerAdapter):
    """Contrastive-LM (clm-serve) REST API 어댑터."""

    backend_name = "clm"

    def __init__(
        self,
        base_url: str = RERANK_CLM_BASE_URL,
        timeout: float = RERANK_CLM_TIMEOUT,
        action_cache: bool = RERANK_CLM_ACTION_CACHE,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.action_cache = action_cache
        self._session = requests.Session()

    def _calculate_scores(
        self,
        query: str,
        candidates: Sequence[Dict[str, Any]],
    ) -> List[float]:
        endpoint = f"{self.base_url}/v1/rank"
        answers = [str(c.get("text") or "") for c in candidates]

        payload = {
            "context": "Given the user query, rank the most relevant documents providing supporting evidence.",
            "question": query,
            "answers": answers,
            "action_cache": self.action_cache,
        }

        try:
            resp = self._session.post(
                endpoint,
                json=payload,
                timeout=self.timeout,
                headers={"Content-Type": "application/json"},
            )
        except requests.exceptions.ConnectionError as exc:
            raise RerankerUnavailableError(
                f"[clm] UNAVAILABLE: CLM 서빙 서버({self.base_url})에 연결할 수 없습니다.\n"
                f"원인: ConnectionRefused. 'clm-serve' 데몬이 실행 중인지 확인하세요."
            ) from exc
        except requests.exceptions.Timeout as exc:
            raise RerankerUnavailableError(
                f"[clm] TIMEOUT: CLM 서버({endpoint}) 응답 시간이 초과되었습니다 ({self.timeout}초)."
            ) from exc
        except Exception as exc:
            raise RerankerUnavailableError(f"[clm] 호출 실패: {exc}") from exc

        if resp.status_code != 200:
            raise RerankerUnavailableError(f"[clm] HTTP {resp.status_code} 오류: {resp.text[:300]}")

        data = resp.json()
        ranked_list = data.get("ranked") or data.get("results") or []
        if not ranked_list:
            raise RerankerUnavailableError("[clm] 서버로부터 유효한 순위 결과를 받지 못했습니다.")

        # 각 후보 인덱스별 점수 매핑
        scores_by_idx: Dict[int, float] = {}
        for item in ranked_list:
            cand_ref = item.get("candidate")
            prob = float(item.get("prob", item.get("score", 0.0)))
            if isinstance(cand_ref, int) and 0 <= cand_ref < len(candidates):
                scores_by_idx[cand_ref] = prob
            elif isinstance(cand_ref, str):
                for idx, c in enumerate(candidates):
                    if idx not in scores_by_idx and str(c.get("text") or "") == cand_ref:
                        scores_by_idx[idx] = prob
                        break

        # 후보 순서대로 점수 리스트 반환
        return [scores_by_idx.get(i, 0.0) for i in range(len(candidates))]

    def close(self) -> None:
        self._session.close()
