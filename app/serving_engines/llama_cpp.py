"""llama.cpp (llama-server) 전용 서빙 어댑터."""

from __future__ import annotations

from typing import Optional
from .openai_compatible import OpenAICompatibleServingAdapter

try:
    from settings import (
        LLAMA_CPP_API_KEY,
        LLAMA_CPP_BASE_URL,
        LLAMA_CPP_MODEL,
        REQUEST_TIMEOUT,
    )
except ModuleNotFoundError:
    from app.settings import (
        LLAMA_CPP_API_KEY,
        LLAMA_CPP_BASE_URL,
        LLAMA_CPP_MODEL,
        REQUEST_TIMEOUT,
    )


class LlamaCppServingAdapter(OpenAICompatibleServingAdapter):
    """로컬 llama-server 서빙 엔진 어댑터."""

    def __init__(
        self,
        base_url: str = LLAMA_CPP_BASE_URL,
        model_name: Optional[str] = LLAMA_CPP_MODEL,
        api_key: str = LLAMA_CPP_API_KEY,
        timeout: float = REQUEST_TIMEOUT,
    ) -> None:
        super().__init__(
            base_url=base_url,
            model_name=model_name,
            api_key=api_key,
            timeout=timeout,
        )

    def health_check(self) -> bool:
        """llama-server 전용 /health 엔드포인트 점검."""
        try:
            resp = self._session.get(f"{self.base_url}/health", timeout=3.0)
            if resp.status_code == 200:
                data = resp.json()
                return data.get("status") in ("ok", "loading")
            return False
        except Exception:
            return super().health_check()

    def _call_engine(self, messages, options):
        """llama.cpp 최적화: cache_prompt 활성화로 반복 질의 및 컨텍스트 재사용 TTFT 가속."""
        llama_options = dict(options)
        llama_options["cache_prompt"] = True
        return super()._call_engine(messages, llama_options)

