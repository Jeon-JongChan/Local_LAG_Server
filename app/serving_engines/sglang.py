"""SGLang (RadixAttention 기반) 고속 서빙 엔진 어댑터."""

from __future__ import annotations

import os
from typing import Optional
from .openai_compatible import OpenAICompatibleServingAdapter

try:
    from settings import REQUEST_TIMEOUT
except ModuleNotFoundError:
    from app.settings import REQUEST_TIMEOUT


class SglangServingAdapter(OpenAICompatibleServingAdapter):
    """SGLang 서빙 엔진 어댑터."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = REQUEST_TIMEOUT,
    ) -> None:
        target_url = (base_url or os.getenv("SGLANG_BASE_URL", "http://localhost:30000")).rstrip("/")
        target_model = model_name or os.getenv("SGLANG_MODEL")
        target_key = api_key or os.getenv("SGLANG_API_KEY", "EMPTY")
        super().__init__(
            base_url=target_url,
            model_name=target_model,
            api_key=target_key,
            timeout=timeout,
        )

    def health_check(self) -> bool:
        """SGLang /health 엔드포인트 헬스체크."""
        try:
            resp = self._session.get(f"{self.base_url}/health", timeout=3.0)
            return resp.status_code == 200
        except Exception:
            return super().health_check()

    def _call_engine(self, messages, options):
        """SGLang 최적화: RadixAttention KV 캐시 친화적 파라미터 주입."""
        sglang_options = dict(options)
        sglang_options["skip_special_tokens"] = True
        sglang_options["top_p"] = 0.9
        return super()._call_engine(messages, sglang_options)

