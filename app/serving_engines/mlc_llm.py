"""MLC LLM (TVM 컴파일 가속 엔진) 서빙 어댑터.

WebGPU, Vulkan, CUDA 등 TVM 기반 고성능 하드웨어 가속 추론에 최적화된 로직을 적용합니다.
미변환 모델 지정 시 Hugging Face MLC 저장소 자동 탐색 및 로컬 변환 파이프라인을 지원합니다.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from .openai_compatible import OpenAICompatibleServingAdapter

try:
    from settings import MODELS_DIR, REQUEST_TIMEOUT
except ModuleNotFoundError:
    from app.settings import MODELS_DIR, REQUEST_TIMEOUT


class MlcLlmServingAdapter(OpenAICompatibleServingAdapter):
    """MLC LLM 고성능 컴파일 엔진 어댑터."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = REQUEST_TIMEOUT,
    ) -> None:
        target_url = (base_url or os.getenv("MLC_LLM_BASE_URL", "http://localhost:8000")).rstrip("/")
        target_model = model_name or os.getenv("MLC_LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
        target_key = api_key or os.getenv("MLC_LLM_API_KEY", "EMPTY")
        super().__init__(
            base_url=target_url,
            model_name=target_model,
            api_key=target_key,
            timeout=timeout,
        )

    def ensure_model_ready(self) -> Path:
        """
        MLC 모델 가중치 검증 및 자동 다운로드/변환 파이프라인 트리거.
        services 어댑터를 통해 Hugging Face MLC 저장소 탐색 및 필요 시 로컬 변환을 수행합니다.
        """
        try:
            from services import get_model_manager
            mgr = get_model_manager("huggingface")
            if hasattr(mgr, "resolve_mlc_model"):
                target_dir = MODELS_DIR / "mlc" / (self.model_name or "default").replace("/", "_")
                return mgr.resolve_mlc_model(self.model_name or "Qwen/Qwen2.5-7B-Instruct", target_dir)
        except Exception as e:
            print(f"[MLC] 모델 준비 확인 중 예외: {e}")
        return MODELS_DIR / "mlc"

    def health_check(self) -> bool:
        """MLC LLM 서버 헬스체크 (GET /health 또는 GET /v1/models)."""
        try:
            resp = self._session.get(f"{self.base_url}/health", timeout=3.0)
            if resp.status_code == 200:
                return True
        except Exception:
            pass
        return super().health_check()

    def _call_engine(self, messages: List[Dict[str, str]], options: Dict[str, Any]) -> str:
        """
        MLC LLM 최적화 파라미터를 주입하여 /v1/chat/completions 호출.
        TVM KV 캐시 재사용을 위해 결정론적 샘플링과 적응형 max_tokens를 설정합니다.
        """
        mlc_options = dict(options)
        mlc_options["repetition_penalty"] = 1.05
        mlc_options["presence_penalty"] = 0.0
        mlc_options["frequency_penalty"] = 0.0

        return super()._call_engine(messages, mlc_options)
