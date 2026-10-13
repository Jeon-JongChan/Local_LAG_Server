"""
laya 리랭커 어댑터 구현체 (Laya Multilingual).
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence
from .base import BaseRerankerAdapter, RerankerUnavailableError

try:
    from settings import (
        RERANK_LAYA_DECISION,
        RERANK_LAYA_MAX_LENGTH,
        RERANK_LAYA_MODEL,
    )
except ModuleNotFoundError:
    from app.settings import (
        RERANK_LAYA_DECISION,
        RERANK_LAYA_MAX_LENGTH,
        RERANK_LAYA_MODEL,
    )


class LayaRerankerAdapter(BaseRerankerAdapter):
    """Laya Multilingual 결정 모델 어댑터."""

    backend_name = "laya"

    def __init__(
        self,
        model_id: str = RERANK_LAYA_MODEL,
        decision_method: str = RERANK_LAYA_DECISION,
        max_length: int = RERANK_LAYA_MAX_LENGTH,
    ) -> None:
        self.model_id = model_id
        self.decision_method = decision_method
        self.max_length = max_length
        self._agent = None

    def _ensure_loaded(self):
        if self._agent is not None:
            return self._agent

        try:
            import laya
        except ImportError as exc:
            raise RerankerUnavailableError(
                f"[laya] 'laya' 라이브러리가 설치되어 있지 않습니다.\n"
                f"설치 명령: pip install laya\n"
                f"원인: {exc}"
            ) from exc

        try:
            print(f"[laya] Laya 모델 로드 중: {self.model_id}", flush=True)
            self._agent = laya.load(self.model_id)
            return self._agent
        except Exception as exc:
            raise RerankerUnavailableError(
                f"[laya] 모델 로드 실패: {self.model_id} — {exc}"
            ) from exc

    def _score_single(self, agent: Any, query: str, doc_text: str) -> float:
        truncated_doc = doc_text[: self.max_length] if len(doc_text) > self.max_length else doc_text
        state = {"query": query, "document": truncated_doc}

        if self.decision_method == "choice":
            questions = {
                "relevance": {
                    "type": "choice",
                    "instructions": "Determine if the document provides direct evidence for the query.",
                    "criteria": {
                        "relevant": "contains key facts, figures, or answers to the query",
                        "irrelevant": "off-topic or does not answer the query",
                    },
                }
            }
        else:
            questions = {
                "is_relevant": {
                    "type": "noul",
                    "instructions": "Does this document contain factual evidence that answers the query?",
                }
            }

        res = agent.predict(state, questions)
        answers = res.get("answers", {}) if isinstance(res, dict) else {}

        if self.decision_method == "choice":
            rel_res = answers.get("relevance", {})
            if isinstance(rel_res, dict):
                return float(rel_res.get("relevant", 0.0))
            return 1.0 if str(rel_res).lower() == "relevant" else 0.0
        else:
            val = answers.get("is_relevant")
            if isinstance(val, (int, float)):
                return float(val)
            if isinstance(val, dict):
                return float(val.get("prob", val.get("yes", 0.0)))
            return 1.0 if str(val).lower() in ("yes", "true", "1") else 0.0

    def _calculate_scores(
        self,
        query: str,
        candidates: Sequence[Dict[str, Any]],
    ) -> List[float]:
        agent = self._ensure_loaded()
        scores: List[float] = []
        for cand in candidates:
            text = str(cand.get("text") or "")
            try:
                score = self._score_single(agent, query, text)
            except Exception as e:
                print(f"[laya] 개별 청크 점수화 오류: {e}", flush=True)
                score = 0.0
            scores.append(score)
        return scores

    def close(self) -> None:
        self._agent = None
