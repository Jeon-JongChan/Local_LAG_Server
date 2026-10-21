"""OpenAI 호환 Chat Completions REST API 서빙 어댑터."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence

import requests

from .base import BaseServingAdapter
from settings import LLAMA_NUM_PREDICT, LLAMA_NUM_PREDICT_THINK


class OpenAICompatibleServingAdapter(BaseServingAdapter):
    """OpenAI 호환 REST API 어댑터.

    extra_options / health_* 로 엔진별 차이를 주입해 얇은 서브클래스 없이 쓴다.
    """

    engine_type: str = "openai"

    def __init__(
        self,
        base_url: str,
        model_name: Optional[str] = None,
        api_key: str = "no-key",
        timeout: float = 600.0,
        *,
        extra_options: Optional[Dict[str, Any]] = None,
        health_path: Optional[str] = "/health",
        health_ok_statuses: Optional[Sequence[str]] = None,
        engine_type: str = "openai",
    ) -> None:
        """초기화.

        Args:
            base_url: 서버 주소 (`.../v1` 포함 여부와 무관하게 정규화).
            model_name: 요청에 넣을 model 필드.
            api_key: Bearer 토큰.
            timeout: HTTP 타임아웃(초).
            extra_options: 매 요청 payload에 병합할 엔진 특화 옵션.
            health_path: 호스트 루트 기준 헬스 경로. None이면 /v1/models만 사용.
            health_ok_statuses: JSON status 허용값(llama.cpp 등). None이면 HTTP 200만 확인.
            engine_type: 팩토리/디버그용 엔진 식별자.
        """
        super().__init__(base_url=base_url, model_name=model_name, timeout=timeout)
        self.api_key = api_key
        self.extra_options = dict(extra_options or {})
        self.health_path = health_path
        self.health_ok_statuses = tuple(health_ok_statuses) if health_ok_statuses else None
        self.engine_type = engine_type
        self._session = requests.Session()

    def _api_root(self) -> str:
        """trailing slash 제거한 base_url."""
        return self.base_url.rstrip("/")

    def _host_root(self) -> str:
        """/v1 접미를 벗긴 서버 루트 (health용)."""
        root = self._api_root()
        if root.endswith("/v1"):
            return root[:-3].rstrip("/") or root
        return root

    def _chat_url(self) -> str:
        """base_url에 /v1이 있어도 이중 경로가 되지 않게 chat 엔드포인트 구성."""
        root = self._api_root()
        if root.endswith("/v1"):
            return f"{root}/chat/completions"
        return f"{root}/v1/chat/completions"

    def _models_url(self) -> str:
        """models 엔드포인트 URL."""
        root = self._api_root()
        if root.endswith("/v1"):
            return f"{root}/models"
        return f"{root}/v1/models"

    def health_check(self) -> bool:
        """헬스체크: 전용 path 우선, 실패 시 /v1/models."""
        if self.health_path:
            try:
                resp = self._session.get(
                    f"{self._host_root()}{self.health_path}",
                    timeout=3.0,
                )
                if resp.status_code == 200:
                    if self.health_ok_statuses is None:
                        return True
                    data = resp.json()
                    return data.get("status") in self.health_ok_statuses
            except Exception:
                pass
        try:
            resp = self._session.get(
                self._models_url(),
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=3.0,
            )
            return resp.status_code == 200
        except Exception:
            return False

    def _call_engine(self, messages: List[Dict[str, str]], options: Dict[str, Any]) -> str:
        """/v1/chat/completions 호출."""
        use_think = options.get("use_think", False)
        temperature = options.get("temperature", 0.2)
        max_tokens = LLAMA_NUM_PREDICT_THINK if use_think else LLAMA_NUM_PREDICT

        payload: Dict[str, Any] = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
            "stop": ["<|im_end|>", "<|endoftext|>", "</s>"],
        }
        if self.model_name:
            payload["model"] = self.model_name

        # 엔진 기본 옵션 먼저, 호출 옵션이 덮어씀
        for opt_key, opt_val in self.extra_options.items():
            if opt_val is not None:
                payload[opt_key] = opt_val
        for opt_key, opt_val in options.items():
            if opt_key not in ("use_think", "mode_label", "temperature") and opt_val is not None:
                payload[opt_key] = opt_val

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        endpoint = self._chat_url()

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
            content = msg.get("content") or ""
            reasoning = msg.get("reasoning_content") or ""

            if reasoning:
                if content:
                    return f"<think>\n{reasoning}\n</think>\n\n{content}"
                # max_tokens 제한 등으로 본문 생성 전 끊겼거나 reasoning에만 내용이 있는 경우 빈 응답 방지 폴백
                return reasoning
            return content
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"서빙 엔진 호출 실패 ({endpoint}): {e}") from e


def _demo_url_normalization() -> None:
    """base_url /v1 정규화 self-check."""
    with_v1 = OpenAICompatibleServingAdapter("http://localhost:8080/v1")
    assert with_v1._chat_url() == "http://localhost:8080/v1/chat/completions"
    assert with_v1._models_url() == "http://localhost:8080/v1/models"
    assert with_v1._host_root() == "http://localhost:8080"
    bare = OpenAICompatibleServingAdapter("http://localhost:8000")
    assert bare._chat_url() == "http://localhost:8000/v1/chat/completions"
    print("openai_compatible URL normalization: ok")


if __name__ == "__main__":
    # python -m serving_engines.openai_compatible  (app/ 가 path에 있을 때)
    _demo_url_normalization()
