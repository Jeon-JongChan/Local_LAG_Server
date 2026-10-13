"""ExLlamaV3 (TabbyAPI / 로컬 EXL3 고속 런타임) 서빙 어댑터."""

from __future__ import annotations

import os
from typing import Optional
from .openai_compatible import OpenAICompatibleServingAdapter

try:
    from settings import REQUEST_TIMEOUT
except ModuleNotFoundError:
    from app.settings import REQUEST_TIMEOUT


class ExLlamaV3ServingAdapter(OpenAICompatibleServingAdapter):
    """ExLlamaV3 / TabbyAPI 서빙 엔진 어댑터."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = REQUEST_TIMEOUT,
    ) -> None:
        target_url = (base_url or os.getenv("EXLLAMA_BASE_URL", "http://localhost:5000")).rstrip("/")
        target_model = model_name or os.getenv("EXLLAMA_MODEL")
        target_key = api_key or os.getenv("EXLLAMA_API_KEY", "admin")
        super().__init__(
            base_url=target_url,
            model_name=target_model,
            api_key=target_key,
            timeout=timeout,
        )

    def health_check(self) -> bool:
        """TabbyAPI/ExLlama 헬스체크 (GET /health 또는 GET /v1/models)."""
        try:
            resp = self._session.get(f"{self.base_url}/health", timeout=3.0)
            if resp.status_code == 200:
                return True
        except Exception:
            pass
        return super().health_check()

    def _call_engine(self, messages, options):
        """ExLlamaV3/TabbyAPI 최적화: EXL3 양자화 모델 특화 min_p 샘플링 적용."""
        exl_options = dict(options)
        exl_options["min_p"] = 0.05
        exl_options["top_p"] = 0.9
        return super()._call_engine(messages, exl_options)

