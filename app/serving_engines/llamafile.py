"""Mozilla Llamafile (단일 포터블 실행 바이너리) 서빙 어댑터.

Llamafile의 내장 프롬프트 캐싱 및 슬롯 최적화 로직을 적용합니다.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional
from .openai_compatible import OpenAICompatibleServingAdapter

try:
    from settings import REQUEST_TIMEOUT
except ModuleNotFoundError:
    from app.settings import REQUEST_TIMEOUT


class LlamafileServingAdapter(OpenAICompatibleServingAdapter):
    """Mozilla Llamafile 엔진 어댑터."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = REQUEST_TIMEOUT,
    ) -> None:
        target_url = (base_url or os.getenv("LLAMAFILE_BASE_URL", "http://localhost:8080")).rstrip("/")
        target_model = model_name or os.getenv("LLAMAFILE_MODEL", "LLaMA_CPP")
        target_key = api_key or os.getenv("LLAMAFILE_API_KEY", "no-key")
        super().__init__(
            base_url=target_url,
            model_name=target_model,
            api_key=target_key,
            timeout=timeout,
        )

    def health_check(self) -> bool:
        """Llamafile /health 또는 /slots 헬스체크."""
        try:
            resp = self._session.get(f"{self.base_url}/health", timeout=3.0)
            if resp.status_code == 200:
                return True
        except Exception:
            pass
        return super().health_check()

    def _call_engine(self, messages: List[Dict[str, str]], options: Dict[str, Any]) -> str:
        """
        Llamafile 특화:
        내장 KV 프롬프트 캐시 재사용(cache_prompt) 및 슬롯 안정성 유지.
        """
        llamafile_options = dict(options)
        llamafile_options["cache_prompt"] = True
        return super()._call_engine(messages, llamafile_options)

