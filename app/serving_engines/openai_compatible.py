"""OpenAI 호환 Chat Completions REST API 서빙 어댑터.

표준 OpenAI 규격(/v1/chat/completions)을 지원하는 모든 엔진의 기본 베이스 클래스입니다.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
import requests

from .base import BaseServingAdapter

try:
    from settings import LLAMA_NUM_PREDICT, LLAMA_NUM_PREDICT_THINK
except ModuleNotFoundError:
    from app.settings import LLAMA_NUM_PREDICT, LLAMA_NUM_PREDICT_THINK


class OpenAICompatibleServingAdapter(BaseServingAdapter):
    """OpenAI 호환 REST API 어댑터."""

    def __init__(
        self,
        base_url: str,
        model_name: Optional[str] = None,
        api_key: str = "no-key",
        timeout: float = 600.0,
    ) -> None:
        super().__init__(base_url=base_url, model_name=model_name, timeout=timeout)
        self.api_key = api_key
        self._session = requests.Session()

    def health_check(self) -> bool:
        """/v1/models 엔드포인트를 통한 서버 헬스체크."""
        try:
            resp = self._session.get(
                f"{self.base_url}/v1/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=3.0,
            )
            return resp.status_code == 200
        except Exception:
            return False

    def _call_engine(self, messages: List[Dict[str, str]], options: Dict[str, Any]) -> str:
        """/v1/chat/completions 호출."""
        endpoint = f"{self.base_url}/v1/chat/completions"
        use_think = options.get("use_think", False)
        temperature = options.get("temperature", 0.2)
        max_tokens = LLAMA_NUM_PREDICT_THINK if use_think else LLAMA_NUM_PREDICT

        payload: Dict[str, Any] = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if self.model_name:
            payload["model"] = self.model_name

        # 서브클래스 엔진별 특화 옵션 병합 (cache_prompt, min_p, repetition_penalty 등)
        for opt_key, opt_val in options.items():
            if opt_key not in ("use_think", "mode_label", "temperature") and opt_val is not None:
                payload[opt_key] = opt_val

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        try:
            resp = self._session.post(
                endpoint,
                data=json.dumps(payload),
                headers=headers,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            choices = data.get("choices", [])
            if not choices:
                return ""
            msg = choices[0].get("message", {})
            return msg.get("content", "") or ""
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"서빙 엔진 호출 실패 ({endpoint}): {e}") from e

