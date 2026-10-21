"""질의(RAG) Application Service (오케스트레이터).

패턴 밖 파이프라인 조율자. Facade/Factory/Strategy/Template 을 조합한다.
구간별 소요 시간을 수집해 관리자 피드(/admin)에 제공한다.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional, Tuple

from chunker import format_citation
from embeddings import get_embedding_adapter
from rerankers import get_reranker_adapter
from serving_engines import get_serving_adapter
from storage import get_rag_storage, rrf_fuse
from settings import (
    ACCELERATOR,
    BM25_TOP_K,
    CHAT_HISTORY_MAX_MESSAGES,
    DOC_PARSER,
    EMBED_DEVICE,
    EMBED_MODEL,
    FAISS_TOP_K,
    HYBRID_BM25,
    HYBRID_RRF_K,
    PROFILE,
    RERANK_STRATEGY,
    active_rerank_model,
    RERANK_TOP_K,
    SERVING_ENGINE,
    MANAGED_SERVING_ENGINES,
)

_REF_MARKER = "\n\n[참고 문서]\n"
_TRACE_MAX = 80
_trace_lock = threading.Lock()
_traces: Deque[Dict[str, Any]] = deque(maxlen=_TRACE_MAX)
# ponytail: 단일 사용자. 동시 질의면 세션별 버킷으로 나눌 것.
_live_lock = threading.Lock()
_live_lines: List[str] = []
_live_active = False
_last_trace: Optional[Dict[str, Any]] = None


def _build_context(hits: List[dict]) -> tuple[str, str]:
    """히트 목록을 LLM 컨텍스트·참고 문서 문자열로 변환."""
    parts: List[str] = []
    refs: List[str] = []

    for i, hit in enumerate(hits, start=1):
        cite = format_citation(hit)
        refs.append(cite)
        parts.append(f"[문서 {i}] {cite}\n{hit['text']}")

    context = "\n\n".join(parts)
    ref_text = "\n".join(f"- {r}" for r in refs)
    return context, ref_text


def _qlog(step: str, detail: str) -> None:
    """콘솔 질의 로그. 같은 줄을 채팅 진행 표시에도 남긴다."""
    line = f"{step} - {detail}"
    print(f"[query] {step} - {detail}", flush=True)
    with _live_lock:
        _live_lines.append(line)


def begin_live_query() -> None:
    """진행 중 로그를 비우고 폴링에 노출한다."""
    global _live_active
    with _live_lock:
        _live_lines.clear()
        _live_active = True


def end_live_query() -> None:
    """진행 표시를 끈다. 마지막 줄은 다음 질의 전까지 유지."""
    global _live_active
    with _live_lock:
        _live_active = False


def get_live_query() -> Dict[str, Any]:
    """채팅 UI 폴링용 현재 단계."""
    with _live_lock:
        return {"active": _live_active, "lines": list(_live_lines)}


def _strip_refs(text: str) -> str:
    """답변에서 참고 문서 블록 제거."""
    idx = text.find(_REF_MARKER)
    return text[:idx].strip() if idx >= 0 else text.strip()


def _prepare_history(history: Optional[List[dict]]) -> List[dict[str, str]]:
    """LLM에 넣을 히스토리 정리."""
    if not history or CHAT_HISTORY_MAX_MESSAGES <= 0:
        return []
    trimmed = history[-CHAT_HISTORY_MAX_MESSAGES:]
    out: List[dict[str, str]] = []
    for msg in trimmed:
        role = msg.get("role")
        if role not in ("user", "assistant"):
            continue
        content = str(msg.get("content") or "").strip()
        if role == "assistant":
            content = _strip_refs(content)
        if content:
            out.append({"role": role, "content": content})
    return out


def _embed_query_text(question: str, history: Optional[List[dict]]) -> str:
    """짧은 후속 질문에 직전 user 맥락을 붙인 검색문."""
    q = question.strip()
    if not history or len(q) > 80:
        return q
    for msg in reversed(history):
        if msg.get("role") == "user":
            prev = str(msg.get("content") or "").strip()
            if prev:
                return f"{prev} - 후속: {q}"
            break
    return q


def _round_s(t0: float) -> float:
    """경과 초를 소수 2자리로."""
    return round(time.perf_counter() - t0, 2)


def _push_trace(trace: Dict[str, Any]) -> None:
    """관리자 피드용 트레이스 저장."""
    global _last_trace
    with _trace_lock:
        _last_trace = trace
        _traces.appendleft(trace)


def list_query_traces(limit: int = 40) -> List[Dict[str, Any]]:
    """최근 질의 트레이스 (새것 먼저)."""
    with _trace_lock:
        return list(_traces)[: max(0, limit)]


def get_runtime_config() -> Dict[str, Any]:
    """현재 선택된 프로필·엔진·모델 스냅샷."""
    return {
        "profile": PROFILE,
        "accelerator": ACCELERATOR,
        "serving_engine": SERVING_ENGINE,
        "embed_model": EMBED_MODEL,
        "embed_device": EMBED_DEVICE,
        "rerank_strategy": RERANK_STRATEGY,
        "rerank_model": active_rerank_model(),
        "doc_parser": DOC_PARSER,
        "faiss_top_k": FAISS_TOP_K,
        "rerank_top_k": RERANK_TOP_K,
        "hybrid_bm25": HYBRID_BM25,
    }


def ask_question(
    question: str,
    *,
    think_mode: str = "instant",
    history: Optional[List[dict]] = None,
) -> str:
    """질문에 대한 답변 문자열만 반환 (CLI 호환)."""
    answer, _trace = ask_question_detailed(
        question, think_mode=think_mode, history=history
    )
    return answer


def ask_question_detailed(
    question: str,
    *,
    think_mode: str = "instant",
    history: Optional[List[dict]] = None,
) -> Tuple[str, Dict[str, Any]]:
    """답변과 구간별 성능 트레이스를 함께 반환."""
    question = question.strip()
    t_all = time.perf_counter()
    stages: List[Dict[str, Any]] = []
    trace: Dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "question": question[:200],
        "think_mode": think_mode,
        "ok": True,
        "stages": stages,
        "config": get_runtime_config(),
        "total_s": 0.0,
    }

    if not question:
        answer = "질문이 비어 있습니다."
        trace["ok"] = False
        trace["total_s"] = _round_s(t_all)
        _push_trace(trace)
        return answer, trace

    store = get_rag_storage()
    begin_live_query()
    try:
        if store.count() == 0:
            answer = (
                "벡터 DB가 비어 있습니다. `data/docs`에 PDF를 넣고 "
                "`python app/main.py ingest --reset` 을 실행하세요."
            )
            trace["ok"] = False
            trace["total_s"] = _round_s(t_all)
            _push_trace(trace)
            return answer, trace

        llm_history = _prepare_history(history)
        if llm_history:
            _qlog("히스토리", f"이전 메시지 {len(llm_history)}개 LLM에 전달")

        search_text = _embed_query_text(question, history)
        _qlog("1/3 검색", f"임베딩={EMBED_MODEL} · 검색문 {len(search_text)}자")

        # --- embed ---
        t0 = time.perf_counter()
        embedder = get_embedding_adapter()
        query_vec = embedder.embed_query(search_text)
        embed_s = _round_s(t0)
        stages.append({
            "id": "embed",
            "label": "임베딩",
            "model": EMBED_MODEL,
            "detail": f"device={embedder.device}",
            "s": embed_s,
        })

        # --- search ---
        t1 = time.perf_counter()
        dense_hits = store.search(query_vec, top_k=FAISS_TOP_K)
        if HYBRID_BM25:
            bm25_hits = store.search_bm25(search_text, top_k=BM25_TOP_K)
            candidates = rrf_fuse(
                dense_hits,
                bm25_hits,
                k=HYBRID_RRF_K,
                top_n=FAISS_TOP_K,
            )
            search_detail = f"dense={len(dense_hits)} BM25={len(bm25_hits)} RRF={len(candidates)}"
        else:
            candidates = dense_hits
            search_detail = f"FAISS={len(candidates)} (BM25 off)"
        search_s = _round_s(t1)
        stages.append({
            "id": "search",
            "label": "검색",
            "model": "FAISS+BM25" if HYBRID_BM25 else "FAISS",
            "detail": search_detail,
            "s": search_s,
        })
        _qlog("1/3 검색", f"완료 embed={embed_s}s search={search_s}s · {search_detail}")

        if not candidates:
            answer = "검색 결과가 없습니다."
            trace["ok"] = False
            trace["total_s"] = _round_s(t_all)
            _push_trace(trace)
            return answer, trace

        # --- rerank ---
        if RERANK_TOP_K <= 0:
            hits = candidates
            stages.append({
                "id": "rerank",
                "label": "리랭크",
                "model": "(skip)",
                "detail": f"RERANK_TOP_K=0 · 후보 {len(hits)}",
                "s": 0.0,
            })
            _qlog("2/3 리랭크", f"skip · 후보 {len(hits)}개")
        else:
            _qlog("2/3 리랭크", f"전략={RERANK_STRATEGY} · {len(candidates)} -> top {RERANK_TOP_K}")
            t2 = time.perf_counter()
            try:
                reranker = get_reranker_adapter()
                hits = reranker.rerank(search_text, candidates, top_k=RERANK_TOP_K)
                backend = getattr(reranker, "backend_name", RERANK_STRATEGY)
            except Exception as exc:
                _qlog("2/3 리랭크", f"경고: 리랭커 실패 ({exc}) - 검색 후보군을 그대로 사용합니다.")
                hits = list(candidates[:RERANK_TOP_K])
                backend = f"{RERANK_STRATEGY}(fallback)"

            rerank_s = _round_s(t2)
            stages.append({
                "id": "rerank",
                "label": "리랭크",
                "model": f"{backend} / {active_rerank_model()}",
                "detail": f"{len(candidates)} → {len(hits)}",
                "s": rerank_s,
            })
            _qlog("2/3 리랭크", f"완료 {rerank_s}s · 선별 {len(hits)}개")

        context, ref_text = _build_context(hits)

        # --- llm ---
        serving_engine = get_serving_adapter()
        _, _, mode_label = serving_engine.resolve_think_mode(question, think_mode=think_mode)
        eng = getattr(serving_engine, "engine_type", SERVING_ENGINE)
        model_name = getattr(serving_engine, "model_name", None) or "(서버 기본)"
        _qlog("3/3 LLM", f"엔진={eng} ({serving_engine.base_url}) · mode={mode_label}")

        try:
            if eng in MANAGED_SERVING_ENGINES:
                from services import get_engine_manager

                if not get_engine_manager().ensure_running():
                    raise RuntimeError(f"llama-server 기동 실패 ({serving_engine.base_url})")
            t3 = time.perf_counter()
            answer = serving_engine.generate(
                question=question,
                context=context,
                think_mode=think_mode,
                history=llm_history,
            )
            llm_s = _round_s(t3)
            stages.append({
                "id": "llm",
                "label": "LLM",
                "model": str(model_name),
                "detail": f"engine={eng} · {serving_engine.base_url} · mode={mode_label}",
                "s": llm_s,
            })
            _qlog("3/3 LLM", f"완료 {llm_s}s · 답변 {len(answer)}자")
        except Exception as e:
            stages.append({
                "id": "llm",
                "label": "LLM",
                "model": str(model_name),
                "detail": f"engine={eng} · 실패 {type(e).__name__}",
                "s": 0.0,
            })
            _qlog("3/3 LLM", f"실패 - {type(e).__name__}: {e}")
            answer = (
                "답변 생성에 실패했습니다. 서빙 엔진이 실행 중인지 확인하세요.\n"
                f"(원인: {type(e).__name__}: {e})\n\n"
                f"[참고 문서]\n{ref_text}"
            )
            trace["ok"] = False
            trace["total_s"] = _round_s(t_all)
            _push_trace(trace)
            return answer, trace

        full = f"{answer}\n\n[참고 문서]\n{ref_text}"
        trace["total_s"] = _round_s(t_all)
        _push_trace(trace)
        return full, trace
    finally:
        end_live_query()
        store.close()
