"""서빙 엔진 추상 베이스 어댑터 모듈.

모든 서빙 엔진(llama.cpp, Ollama, vLLM, SGLang, ExLlamaV3 등)은
이 클래스를 상속받아 구현하며, 공통 템플릿 메서드를 통해
사고 모드(/think) 파싱, 프롬프트 빌드 및 출력을 정제합니다.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

_INLINE_THINK_RE = re.compile(r"^/think\b", re.IGNORECASE)
_INLINE_NO_THINK_RE = re.compile(r"^/no_think\b", re.IGNORECASE)

_THINK_KEYWORDS = (
    "왜", "어떻게", "원인", "비교", "분석", "차이", "장단점", "설계", "구현 방법",
    "이유", "해결책", "트러블슈팅", "코드 작성", "코드 리뷰", "장단점 비교",
)

_SYSTEM_PROMPT = """당신은 주어진 [참고 문서]를 바탕으로 사용자의 질문에 정확하게 답변하는 전문 AI 도우미입니다.
반드시 제공된 [참고 문서]의 내용에 근거하여 한국어로 명확하고 간결하게 답변하세요.
문서에 없는 내용은 추측하지 말고 솔직하게 모른다고 답하세요."""


class ThinkMode:
    """사고 모드 상수."""
    AUTO = "auto"
    THINK = "think"
    INSTANT = "instant"


class BaseServingAdapter(ABC):
    """서빙 엔진 추상 베이스 어댑터 클래스."""

    def __init__(self, base_url: str, model_name: Optional[str] = None, timeout: float = 600.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.timeout = timeout

    def resolve_think_mode(self, question: str, think_mode: str = "instant") -> Tuple[str, bool, str]:
        """
        질문 내 인라인 지시어(/think, /no_think) 또는 자동 키워드 기반 사고 모드 판별.

        Returns:
            (정제된 질문, 사고모드 활성화 여부, 결정 라벨)
        """
        q = question.strip()
        has_think = bool(_INLINE_THINK_RE.search(q))
        has_no_think = bool(_INLINE_NO_THINK_RE.search(q))
        cleaned = _INLINE_THINK_RE.sub("", q)
        cleaned = _INLINE_NO_THINK_RE.sub("", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        if has_think and not has_no_think:
            return cleaned, True, "inline->think"
        if has_no_think:
            return cleaned, False, "inline->instant"

        if think_mode == "think":
            return cleaned, True, "think"
        if think_mode == "instant":
            return cleaned, False, "instant"

        # auto 모드: 길이 및 분석 키워드 기반 판단
        if len(cleaned) >= 200 or any(kw in cleaned for kw in _THINK_KEYWORDS) or (cleaned.count("?") >= 2):
            return cleaned, True, "auto->think"
        return cleaned, False, "auto->instant"

    def build_messages(
        self,
        question: str,
        context: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> List[Dict[str, str]]:
        """시스템 프롬프트, 이전 대화 및 컨텍스트를 결합한 표준 메시지 리스트 구성."""
        user_content = (
            f"[참고 문서]\n{context}\n\n[질문]\n{question}\n\n"
            "위 [참고 문서]를 바탕으로 질문에 답변하세요."
        ) if context else question

        messages: List[Dict[str, str]] = [{"role": "system", "content": _SYSTEM_PROMPT}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_content})
        return messages

    def generate(
        self,
        question: str,
        context: str = "",
        *,
        think_mode: str = "instant",
        history: Optional[List[Dict[str, str]]] = None,
        temperature: float = 0.2,
    ) -> str:
        """
        [템플릿 메서드] 질문과 컨텍스트를 기반으로 최종 답변을 생성합니다.

        1. 지시어 파싱 -> 2. 메시지 빌드 -> 3. 서브클래스 호출 -> 4. 출력 정제
        """
        cleaned_q, use_think, mode_label = self.resolve_think_mode(question, think_mode)
        messages = self.build_messages(cleaned_q, context, history)

        options = {
            "use_think": use_think,
            "mode_label": mode_label,
            "temperature": temperature,
        }

        raw_output = self._call_engine(messages, options)
        return self._clean_output(raw_output)

    def _clean_output(self, text: str) -> str:
        """답변 텍스트 공통 정제 (태그 닫힘 등)."""
        out = text.strip()
        if "<think>" in out and "</think>" not in out:
            out = re.sub(r"<think>[\s\S]*$", "", out).strip()
        elif "</think>" in out:
            out = out.split("</think>")[-1].strip()
        return out

    @abstractmethod
    def _call_engine(self, messages: List[Dict[str, str]], options: Dict[str, Any]) -> str:
        """각 서빙 엔진별 실제 추론 호출 로직 (서브클래스 구현)."""
        pass

    @abstractmethod
    def health_check(self) -> bool:
        """서빙 엔진 헬스체크 (서브클래스 구현)."""
        pass

