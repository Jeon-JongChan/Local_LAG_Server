"""
laya 리랭커 어댑터 구현체 (Laya Multilingual).

laya 라이브러리가 미설치된 경우 자동 치유(Auto-healing)를 통해 pip 설치를 시도합니다.
"""

from __future__ import annotations

import subprocess
import sys
from typing import Any, Dict, List, Sequence
from .base import BaseRerankerAdapter, RerankerUnavailableError

from settings import (
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
        except ImportError:
            print(
                "[laya] 'laya' 라이브러리가 미설치되어 있어 자동 설치를 진행합니다 (pip install laya)...",
                flush=True,
            )
            cmd = [sys.executable, "-m", "pip", "install", "laya"]
            res = subprocess.run(cmd)
            if res.returncode == 0:
                try:
                    import laya
                except ImportError as exc2:
                    raise RerankerUnavailableError(
                        f"[laya] 라이브러리 설치 후 임포트 실패: {exc2}"
                    ) from exc2
            else:
                raise RerankerUnavailableError(
                    f"[laya] 라이브러리 자동 설치 실패 (pip exit code {res.returncode})"
                )

        try:
            print(f"[laya] Laya 모델 로드 중: {self.model_id}", flush=True)
            self._agent = laya.load(self.model_id)
            return self._agent
        except Exception as exc:
            raise RerankerUnavailableError(
                f"[laya] 모델 로드 실패: {self.model_id} - {exc}"
            ) from exc

    def _build_questions(self) -> Dict[str, Dict[str, Any]]:
        """Laya 결정 프롬프트 질의 정의 생성."""
        if self.decision_method == "choice":
            return {
                "relevance": {
                    "type": "choice",
                    "instructions": "Determine if the document provides direct evidence for the query.",
                    "criteria": {
                        "relevant": "contains key facts, figures, or answers to the query",
                        "irrelevant": "off-topic or does not answer the query",
                    },
                }
            }
        return {
            "is_relevant": {
                "type": "noul",
                "instructions": "Does this document contain factual evidence that answers the query?",
            }
        }

    def _extract_score(self, answers: Dict[str, Any]) -> float:
        """Laya 응답에서 관련도 점수(0.0 ~ 1.0) 추출."""
        if self.decision_method == "choice":
            rel_res = answers.get("relevance", {})
            if isinstance(rel_res, dict):
                probs = rel_res.get("probabilities", {})
                if isinstance(probs, dict) and "relevant" in probs:
                    return float(probs["relevant"])
                return float(rel_res.get("relevant", 0.0))
            return 1.0 if str(rel_res).lower() == "relevant" else 0.0
        else:
            val = answers.get("is_relevant")
            if isinstance(val, (int, float)):
                return float(val)
            if isinstance(val, dict):
                for key in ("noul", "prob", "probability", "yes", "score"):
                    if key in val and isinstance(val[key], (int, float)):
                        return float(val[key])
                return 0.0
            return 1.0 if str(val).lower() in ("yes", "true", "1") else 0.0

    def _calculate_scores(
        self,
        query: str,
        candidates: Sequence[Dict[str, Any]],
    ) -> List[float]:
        if not candidates:
            return []
        agent = self._ensure_loaded()
        questions = self._build_questions()

        states = []
        for cand in candidates:
            text = str(cand.get("text") or "")
            truncated = (
                text[: self.max_length]
                if len(text) > self.max_length
                else text
            )
            states.append({"query": query, "document": truncated})

        # predict_batch 지원 시 한 번의 포워드 패스로 고속 처리
        if hasattr(agent, "predict_batch"):
            try:
                results = agent.predict_batch(states, questions)
                return [
                    self._extract_score(
                        r.get("answers", {}) if isinstance(r, dict) else {}
                    )
                    for r in results
                ]
            except Exception as e:
                print(f"[laya] 배치 점수화 실패, 순차 처리로 대체: {e}", flush=True)

        scores: List[float] = []
        for state in states:
            try:
                res = agent.predict(state, questions)
                answers = res.get("answers", {}) if isinstance(res, dict) else {}
                scores.append(self._extract_score(answers))
            except Exception as e:
                print(f"[laya] 개별 청크 점수화 오류: {e}", flush=True)
                scores.append(0.0)
        return scores

    def close(self) -> None:
        self._agent = None
