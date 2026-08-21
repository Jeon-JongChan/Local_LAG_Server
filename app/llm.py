"""
LLM 호출 모듈 — llama-server (llama.cpp) OpenAI 호환 Chat Completions API.
"""

from __future__ import annotations

import json
import re
from typing import Any, Literal

import requests

from settings import (
    CHAT_TEMPERATURE,
    LLAMA_CPP_API_KEY,
    LLAMA_CPP_BASE_URL,
    LLAMA_CPP_MODEL,
    LLAMA_NUM_PREDICT,
    LLAMA_NUM_PREDICT_THINK,
    REQUEST_TIMEOUT,
    THINK_AUTO_MIN_CHARS,
)

ThinkMode = Literal["instant", "think", "auto"]

_THINK_KEYWORDS = (
    "자세히",
    "심층",
    "분석",
    "비교",
    "차이점",
    "정리해",
    "종합",
    "단계별",
    "근거",
    "설명해",
    "검토",
    "평가",
    "요약해",
    "정리해서",
    "비교해",
)

_INLINE_THINK_RE = re.compile(r"/think\b", re.IGNORECASE)
_INLINE_NO_THINK_RE = re.compile(r"/no_think\b", re.IGNORECASE)


def parse_question_directives(question: str) -> tuple[str, bool | None]:
    """
  질문 안의 /think · /no_think 지시어를 파싱합니다.
  Returns: (정리된 질문, True=think / False=instant / None=미지정)
    """
    q = question.strip()
    has_think = bool(_INLINE_THINK_RE.search(q))
    has_no_think = bool(_INLINE_NO_THINK_RE.search(q))
    cleaned = _INLINE_THINK_RE.sub("", q)
    cleaned = _INLINE_NO_THINK_RE.sub("", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if has_think and not has_no_think:
        return cleaned, True
    if has_no_think:
        return cleaned, False
    return cleaned, None


def should_auto_think(question: str) -> bool:
    """긴 질문·복잡한 질문이면 thinking 으로 자동 전환."""
    q = question.strip()
    if len(q) >= THINK_AUTO_MIN_CHARS:
        return True
    if any(kw in q for kw in _THINK_KEYWORDS):
        return True
    if q.count("?") + q.count("？") >= 2:
        return True
    return False


def resolve_think_mode(
    question: str,
    *,
    think_mode: ThinkMode = "instant",
) -> tuple[str, bool, str]:
    """
    최종 thinking 사용 여부를 결정합니다.

    Returns: (cleaned_question, use_think, resolved_label)
    resolved_label: instant | think | auto→instant | auto→think
    """
    cleaned, inline = parse_question_directives(question)

    if inline is not None:
        label = "think" if inline else "instant"
        return cleaned, inline, f"inline→{label}"

    if think_mode == "think":
        return cleaned, True, "think"
    if think_mode == "instant":
        return cleaned, False, "instant"

    if should_auto_think(cleaned):
        return cleaned, True, "auto→think"
    return cleaned, False, "auto→instant"


def _max_tokens(use_think: bool) -> int:
    return LLAMA_NUM_PREDICT_THINK if use_think else LLAMA_NUM_PREDICT


def _system_prompt(use_think: bool) -> str:
    prefix = "/think\n" if use_think else "/no_think\n"
    return (
        f"{prefix}"
        "너는 문서 질의응답 도우미다. "
        "최대한 제공된 문맥 안에서만 답하라. "
        "문맥에 없는 내용은 추측하지 말고 '문서에서 확인되지 않음'이라고 답하라. "
        "답변 마지막에는 참고한 파일명과 페이지를 간단히 적어라."
    )


def _user_prompt(question: str, context: str, use_think: bool) -> str:
    # instant 모드에서 /no_think 가 무시되는 경우를 줄이기 위해 user 쪽에도 명시
    head = "/think\n" if use_think else "/no_think\n"
    return f"""{head}[문맥]
{context}

[질문]
{question}
"""


def _server_root() -> str:
    """http://host:8080/v1 → http://host:8080"""
    base = LLAMA_CPP_BASE_URL.rstrip("/")
    if base.endswith("/v1"):
        return base[:-3]
    return base


def _estimate_tokens_from_chars(text: str) -> int:
    """토크나이저 실패 시 대략치 (한글·영문 혼합)."""
    if not text:
        return 0
    # 한글은 글자당 ~1.5~2토큰에 가깝고, 영문은 4자≈1토큰 — 보수적으로 chars*0.6
    return max(1, int(len(text) * 0.6))


def _tokenize_count(text: str) -> tuple[int, str]:
    """
    llama-server /tokenize 로 토큰 수 측정.
    Returns: (count, source)  source = tokenize | estimate
    """
    if not text:
        return 0, "tokenize"
    root = _server_root()
    try:
        resp = requests.post(
            f"{root}/tokenize",
            headers={
                "Authorization": f"Bearer {LLAMA_CPP_API_KEY}",
                "Content-Type": "application/json",
            },
            json={"content": text, "add_special": False},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        tokens = data.get("tokens")
        if isinstance(tokens, list):
            return len(tokens), "tokenize"
    except Exception as exc:
        print(f"[llm] tokenize 실패 → 추정 사용: {exc}", flush=True)
    return _estimate_tokens_from_chars(text), "estimate"


def _log_pre_request_tokens(
    *,
    system: str,
    history: list[dict[str, str]] | None,
    context: str,
    question: str,
    max_tokens: int,
    mode_label: str,
) -> None:
    """메인 모델 호출 직전 — 구간별 글자/토큰 로그."""
    hist = history or []
    hist_text = "\n".join(
        f"{m.get('role', '')}: {m.get('content', '')}" for m in hist
    )
    parts = {
        "system": system,
        "history": hist_text,
        "context": context,
        "question": question,
    }
    counts: dict[str, tuple[int, int, str]] = {}
    total_tok = 0
    source = "tokenize"
    for name, text in parts.items():
        n_tok, src = _tokenize_count(text)
        if src == "estimate":
            source = "estimate"
        counts[name] = (len(text), n_tok, src)
        total_tok += n_tok

    # 채팅 템플릿 오버헤드 대략 (+메시지당 수 토큰) — 참고용
    overhead = 8 + 4 * (1 + len(hist) + 1)
    approx_prompt = total_tok + overhead

    print(
        f"[llm] 요청 전 토큰 · mode={mode_label} · max_tokens={max_tokens} · "
        f"history={len(hist)}개 · 측정={source}",
        flush=True,
    )
    print(
        f"[llm]   system  chars={counts['system'][0]} tokens≈{counts['system'][1]}",
        flush=True,
    )
    print(
        f"[llm]   history chars={counts['history'][0]} tokens≈{counts['history'][1]}",
        flush=True,
    )
    print(
        f"[llm]   context chars={counts['context'][0]} tokens≈{counts['context'][1]}  ← RAG",
        flush=True,
    )
    print(
        f"[llm]   question chars={counts['question'][0]} tokens≈{counts['question'][1]}",
        flush=True,
    )
    print(
        f"[llm]   합계 tokens≈{total_tok} (+chat템플릿~{overhead}) "
        f"→ prompt≈{approx_prompt} · 생성한도={max_tokens}",
        flush=True,
    )


def _extract_assistant_text(choice: dict[str, Any], *, use_think: bool) -> str:
    message = choice.get("message") or {}
    content = (message.get("content") or "").strip()
    reasoning = (message.get("reasoning_content") or "").strip()
    finish = choice.get("finish_reason") or ""

    if content:
        if use_think and reasoning:
            print("[llm] thinking 완료 — content + reasoning_content", flush=True)
        return content

    if reasoning and finish == "length":
        budget = LLAMA_NUM_PREDICT_THINK if use_think else LLAMA_NUM_PREDICT
        raise RuntimeError(
            "생성 한도(max_tokens)가 thinking(reasoning_content)에만 소진되어 "
            "본문(content)이 비었습니다. "
            f"(mode={'think' if use_think else 'instant'}, max_tokens={budget}, "
            "finish_reason=length). "
            "instant 이면 enable_thinking=false 가 적용됐는지, "
            "think 이면 LLAMA_NUM_PREDICT_THINK 를 늘려 보세요."
        )

    if reasoning and use_think:
        print("[llm] 경고: content 없음 — reasoning_content 사용", flush=True)
        return reasoning

    if reasoning and not use_think:
        raise RuntimeError(
            "instant(enable_thinking=false)인데 thinking 만 생성되었습니다. "
            f"(finish_reason={finish}). "
            "llama-server 를 재시작해 --reasoning auto|--jinja 가 적용됐는지 확인하세요."
        )

    raise RuntimeError(
        "llama-server 응답 본문이 비어 있습니다. "
        f"(finish_reason={finish}, message_keys={list(message.keys())})"
    )


def _call_chat(
    question: str,
    context: str,
    *,
    use_think: bool,
    max_tokens: int,
    mode_label: str,
    history: list[dict[str, str]] | None = None,
) -> str:
    messages: list[dict[str, str]] = [
        {"role": "system", "content": _system_prompt(use_think)},
    ]
    if history:
        messages.extend(history)
    user_content = _user_prompt(question, context, use_think)
    messages.append({"role": "user", "content": user_content})

    payload: dict = {
        "messages": messages,
        "temperature": CHAT_TEMPERATURE,
        "max_tokens": max_tokens,
        "stream": False,
        # Qwen3/3.5: /no_think 문자열만으로는 thinking 이 안 꺼지는 경우가 많음
        "chat_template_kwargs": {"enable_thinking": use_think},
        "enable_thinking": use_think,
    }
    if LLAMA_CPP_MODEL:
        payload["model"] = LLAMA_CPP_MODEL

    print(
        f"[llm] chat · mode={mode_label} · enable_thinking={use_think} · "
        f"max_tokens={max_tokens}",
        flush=True,
    )

    _log_pre_request_tokens(
        system=messages[0]["content"],
        history=history,
        context=context,
        question=question,
        max_tokens=max_tokens,
        mode_label=mode_label,
    )

    response = requests.post(
        f"{LLAMA_CPP_BASE_URL}/chat/completions",
        headers={
            "Authorization": f"Bearer {LLAMA_CPP_API_KEY}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()

    data = response.json()
    usage = data.get("usage") or {}
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError(
            f"llama-server 응답에 choices 가 없습니다: "
            f"{json.dumps(data, ensure_ascii=False)[:500]}"
        )

    choice = choices[0]
    finish = choice.get("finish_reason")
    print(
        f"[llm] HTTP 200 · mode={mode_label} · finish_reason={finish} · "
        f"응답 후 tokens prompt={usage.get('prompt_tokens')} "
        f"completion={usage.get('completion_tokens')}",
        flush=True,
    )
    return _extract_assistant_text(choice, use_think=use_think)


def chat_answer(
    question: str,
    context: str,
    *,
    think_mode: ThinkMode = "instant",
    history: list[dict[str, str]] | None = None,
) -> str:
    """
    검색 문맥 + 질문으로 답변 생성.

    think_mode:
      - instant: 항상 /no_think (기본, 빠른 응답)
      - think:   항상 /think
      - auto:    짧고 단순하면 instant, 길거나 복잡하면 think
    """
    cleaned, use_think, mode_label = resolve_think_mode(
        question, think_mode=think_mode
    )
    max_tokens = _max_tokens(use_think)

    try:
        return _call_chat(
            cleaned,
            context,
            use_think=use_think,
            max_tokens=max_tokens,
            mode_label=mode_label,
            history=history,
        )
    except RuntimeError as exc:
        # instant 실패 시 토큰 2배 + /no_think 재시도 1회
        if not use_think:
            retry_tokens = min(max_tokens * 2, LLAMA_NUM_PREDICT_THINK)
            print(
                f"[llm] instant 실패 → 재시도 (max_tokens={retry_tokens}): {exc}",
                flush=True,
            )
            return _call_chat(
                cleaned,
                context,
                use_think=False,
                max_tokens=retry_tokens,
                mode_label="instant-retry",
                history=history,
            )
        raise
