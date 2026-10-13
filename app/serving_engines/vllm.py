"""vLLM 고성능 분산 서빙 엔진 어댑터.

vLLM의 OpenAI 호환 API 서버를 기반으로 구동됩니다.
"""

from __future__ import annotations

import os
from typing import Optional
from .openai_compatible import OpenAICompatibleServingAdapter

try:
    from settings import REQUEST_TIMEOUT
except ModuleNotFoundError:
    from app.settings import REQUEST_TIMEOUT


class VllmServingAdapter(OpenAICompatibleServingAdapter):
    """vLLM 서빙 엔진 어댑터."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = REQUEST_TIMEOUT,
    ) -> None:
        target_url = (base_url or os.getenv("VLLM_BASE_URL", "http://localhost:8000")).rstrip("/")
        target_model = model_name or os.getenv("VLLM_MODEL")
        target_key = api_key or os.getenv("VLLM_API_KEY", "EMPTY")
        super().__init__(
            base_url=target_url,
            model_name=target_model,
            api_key=target_key,
            timeout=timeout,
        )

    def health_check(self) -> bool:
        """vLLM /health 엔드포인트 헬스체크."""
        try:
            resp = self._session.get(f"{self.base_url}/health", timeout=3.0)
            return resp.status_code == 200
        except Exception:
            return super().health_check()

    def _call_engine(self, messages, options):
        """vLLM 최적화: PagedAttention prefix 캐싱 및 고속 디코딩 파라미터 설정."""
        vllm_options = dict(options)
        vllm_options["best_of"] = 1
        vllm_options["ignore_eos"] = False
        vllm_options["repetition_penalty"] = 1.05
        return super()._call_engine(messages, vllm_options)

