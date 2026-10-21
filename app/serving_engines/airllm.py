"""AirLLM 기반 초저 VRAM 인메모리 레이어 스트리밍 서빙 엔진 어댑터 모듈.

local-rag의 5대 핵심 디자인 패턴(Adapter & Template Method)을 준수하며,
2GB~4GB 극소 VRAM 환경에서도 초대형 모델을 레이어 단위로 디스크/RAM에서 GPU로
스트리밍 추론할 수 있는 인메모리 어댑터를 제공합니다.
"""

from __future__ import annotations

import importlib.util
from typing import Any, Dict, List, Optional

from .base import BaseServingAdapter
from settings import (
    ACCELERATOR,
    AIRLLM_COMPRESSION,
    AIRLLM_DEVICE,
    AIRLLM_MAX_LENGTH,
    AIRLLM_MODEL,
    LLAMA_NUM_PREDICT,
    LLAMA_NUM_PREDICT_THINK,
    REQUEST_TIMEOUT,
)


class AirLLMServingAdapter(BaseServingAdapter):
    """AirLLM 기반 초저 VRAM 인메모리 레이어 스트리밍 서빙 어댑터 클래스."""

    engine_type: str = "airllm"

    def __init__(
        self,
        base_url: str = "in-memory",
        model_name: Optional[str] = None,
        timeout: float = REQUEST_TIMEOUT,
        device: Optional[str] = None,
        compression: Optional[str] = None,
        max_seq_len: Optional[int] = None,
    ) -> None:
        """어댑터 생성자. 무거운 가중치 로드는 지연(Lazy Loading)합니다.

        Args:
            base_url: 인메모리 엔진 식별자 (기본값: 'in-memory').
            model_name: HuggingFace 모델 식별자 (기본값: settings.AIRLLM_MODEL).
            timeout: 추론 타임아웃(초).
            device: 연산 디바이스 ('cuda' 또는 'cpu').
            compression: 양자화 압축 방식 (예: '4bit' 또는 None).
            max_seq_len: 컨텍스트 최대 시퀀스 길이.
        """
        super().__init__(
            base_url=base_url or "in-memory",
            model_name=AIRLLM_MODEL if model_name is None else model_name,
            timeout=timeout,
        )
        self.device = device or AIRLLM_DEVICE
        self.compression = compression or AIRLLM_COMPRESSION
        self.max_seq_len = max_seq_len or AIRLLM_MAX_LENGTH
        self._model: Any = None
        self._tokenizer: Any = None
        self._initialized: bool = False

    def is_installed(self) -> bool:
        """AirLLM 외부 라이브러리 설치 여부를 확인합니다."""
        return importlib.util.find_spec("airllm") is not None

    def health_check(self) -> bool:
        """어댑터 상태 점검: 라이브러리 설치 여부 및 대상 모델명 유효성 검사 (크래시 없음)."""
        if not self.is_installed():
            return False
        if not self.model_name or not str(self.model_name).strip():
            return False
        return True

    def is_ready(self) -> bool:
        """모델 및 토크나이저 인스턴스화 완료 여부를 반환합니다."""
        return self._initialized and self._model is not None

    def ensure_model(self) -> Any:
        """AirLLM 모델 및 토크나이저를 지연 적재(Lazy Loading)합니다.

        라이브러리 미설치 시 친절한 안내 메시지와 함께 RuntimeError를 발생시킵니다.
        """
        if self._model is not None:
            return self._model

        # 라이브러리 미설치 시 사용자 친화적 가이드 제공
        if not self.is_installed():
            raise RuntimeError(
                "AirLLM 라이브러리가 설치되지 않았습니다. "
                "초저 VRAM 레이어 스트리밍 추론을 사용하려면 'pip install airllm'을 실행하거나, "
                "SERVING_ENGINE을 'llama.cpp' 또는 'ollama'로 변경하십시오."
            )

        try:
            from airllm import AutoModel
            from transformers import AutoTokenizer

            # 토크나이저 로드
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
            )

            # 모델 옵션 구성 (양자화 및 최대 시퀀스 길이)
            kwargs: Dict[str, Any] = {}
            if self.compression:
                kwargs["compression"] = self.compression
            if self.max_seq_len:
                kwargs["max_seq_len"] = self.max_seq_len

            # AirLLM 레이어 스트리밍 모델 초기화
            self._model = AutoModel.from_pretrained(
                self.model_name,
                device=self.device,
                **kwargs,
            )
            self._initialized = True
            return self._model
        except Exception as e:
            self._model = None
            self._tokenizer = None
            self._initialized = False
            raise RuntimeError(f"AirLLM 모델({self.model_name}) 로드 실패: {e}") from e

    def _call_engine(self, messages: List[Dict[str, str]], options: Dict[str, Any]) -> str:
        """[템플릿 메서드 단계] 대화 메시지를 인코딩하여 AirLLM 스트리밍 추론을 수행합니다."""
        self.ensure_model()

        prompt = self._format_messages(messages)
        temperature = options.get("temperature", 0.2)
        use_think = options.get("use_think", False)
        num_predict = LLAMA_NUM_PREDICT_THINK if use_think else LLAMA_NUM_PREDICT

        try:
            inputs = self._tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_seq_len,
            )
            if self.device == "cuda":
                inputs = {k: v.to("cuda") for k, v in inputs.items()}

            gen_kwargs: Dict[str, Any] = {
                "max_new_tokens": num_predict,
                "use_cache": True,
            }
            if temperature > 0:
                gen_kwargs["temperature"] = temperature
                gen_kwargs["do_sample"] = True
            else:
                gen_kwargs["do_sample"] = False

            output_tokens = self._model.generate(
                inputs["input_ids"],
                **gen_kwargs,
            )
            prompt_len = inputs["input_ids"].shape[-1]
            generated_ids = output_tokens[0][prompt_len:]
            return str(self._tokenizer.decode(generated_ids, skip_special_tokens=True))
        except Exception as e:
            raise RuntimeError(f"AirLLM 추론 실패: {e}") from e

    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        """대화 메시지 리스트를 토크나이저 chat_template 또는 기본 포맷으로 결합합니다."""
        if self._tokenizer is not None and hasattr(self._tokenizer, "apply_chat_template"):
            try:
                formatted = self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                if formatted:
                    return str(formatted)
            except Exception:
                pass

        # 템플릿 미지원 시 기본 텍스트 포맷팅 폴백
        lines: List[str] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                lines.append(f"<|system|>\n{content}")
            elif role == "user":
                lines.append(f"<|user|>\n{content}")
            elif role == "assistant":
                lines.append(f"<|assistant|>\n{content}")
        lines.append("<|assistant|>\n")
        return "\n\n".join(lines)


__all__ = ["AirLLMServingAdapter"]
