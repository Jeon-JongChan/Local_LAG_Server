"""Ollama 서빙 어댑터.

Strategy 후보 중 하나이며, /api/chat 스키마를 BaseServingAdapter 로 맞추는 Adapter.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import requests

from .base import BaseServingAdapter
from settings import (
    LLAMA_NUM_PREDICT,
    LLAMA_NUM_PREDICT_THINK,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    REQUEST_TIMEOUT,
)


class OllamaServingAdapter(BaseServingAdapter):
    """Ollama 로컬 서빙 엔진 어댑터."""

    engine_type = "ollama"

    def __init__(
        self,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout: float = REQUEST_TIMEOUT,
    ) -> None:
        """초기화. URL/모델 기본값은 SettingsFacade 경유 상수."""
        super().__init__(
            base_url=(base_url or OLLAMA_BASE_URL).rstrip("/"),
            model_name=model_name or OLLAMA_MODEL,
            timeout=timeout,
        )
        self._session = requests.Session()

    def health_check(self) -> bool:
        """Ollama 서버 구동 여부 확인 (GET /api/tags)."""
        try:
            resp = self._session.get(f"{self.base_url}/api/tags", timeout=3.0)
            return resp.status_code == 200
        except Exception:
            return False

    def _call_engine(self, messages: List[Dict[str, str]], options: Dict[str, Any]) -> str:
        """Ollama /api/chat 호출."""
        endpoint = f"{self.base_url}/api/chat"
        use_think = options.get("use_think", False)
        temperature = options.get("temperature", 0.2)
        num_predict = LLAMA_NUM_PREDICT_THINK if use_think else LLAMA_NUM_PREDICT

        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "keep_alive": "30m",
            "options": {
                "temperature": temperature,
                "num_predict": num_predict,
                "num_ctx": 8192,
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
            message = data.get("message", {})
            return message.get("content", "") or ""
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Ollama 서빙 엔진 호출 실패 ({endpoint}): {e}") from e
