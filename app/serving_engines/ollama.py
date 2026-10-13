"""Ollama 전용 서빙 엔진 어댑터.

Ollama의 로컬 REST API(/api/chat) 및 OpenAI 호환 모드를 모두 지원합니다.
"""

from __future__ import annotations

import os
import json
from typing import Any, Dict, List, Optional
import requests

from .base import BaseServingAdapter

try:
    from settings import LLAMA_NUM_PREDICT, LLAMA_NUM_PREDICT_THINK, REQUEST_TIMEOUT
except ModuleNotFoundError:
    from app.settings import LLAMA_NUM_PREDICT, LLAMA_NUM_PREDICT_THINK, REQUEST_TIMEOUT


class OllamaServingAdapter(BaseServingAdapter):
    """Ollama 로컬 서빙 엔진 어댑터."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout: float = REQUEST_TIMEOUT,
    ) -> None:
        target_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
        target_model = model_name or os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
        super().__init__(base_url=target_url, model_name=target_model, timeout=timeout)
        self._session = requests.Session()

    def health_check(self) -> bool:
        """Ollama 서버 구동 여부 확인 (GET /api/tags)."""
        try:
            resp = self._session.get(f"{self.base_url}/api/tags", timeout=3.0)
            return resp.status_code == 200
        except Exception:
            return False

    def _call_engine(self, messages: List[Dict[str, str]], options: Dict[str, Any]) -> str:
        """Ollama /api/chat 엔드포인트를 호출하여 답변 생성."""
        endpoint = f"{self.base_url}/api/chat"
        use_think = options.get("use_think", False)
        temperature = options.get("temperature", 0.2)
        num_predict = LLAMA_NUM_PREDICT_THINK if use_think else LLAMA_NUM_PREDICT

        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "keep_alive": "30m",  # 모델 언로드 방지로 콜드스타트 지연 제거
            "options": {
                "temperature": temperature,
                "num_predict": num_predict,
                "num_ctx": 8192,  # RAG 문서 컨텍스트 보존을 위한 8K 컨텍스트 확장
                "repeat_penalty": 1.1,
            },
        }

        try:
            resp = self._session.post(
                endpoint,
                data=json.dumps(payload),
                headers={"Content-Type": "application/json"},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            # Ollama /api/chat 반환: {"message": {"role": "assistant", "content": "..."}}
            message = data.get("message", {})
            return message.get("content", "") or ""
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Ollama 서빙 엔진 호출 실패 ({endpoint}): {e}") from e

