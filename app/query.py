"""
질의(RAG) 오케스트레이션.

흐름:
  질문 → Qwen3 임베딩 → FAISS dense
       → (선택) BM25 FTS5
       → RRF 병합 → (선택) Qwen3 리랭크 → llama.cpp 답변 + 출처
"""

from __future__ import annotations

import time
from typing import List, Optional

from chunker import format_citation
from llm import chat_answer, resolve_think_mode
from models import embed_query, rerank
from settings import (
    BM25_TOP_K,
    CHAT_HISTORY_MAX_MESSAGES,
    EMBED_MODEL,
    FAISS_TOP_K,
    HYBRID_BM25,
    HYBRID_RRF_K,
    LLAMA_CPP_BASE_URL,
    RERANK_MODEL,
    RERANK_TOP_K,
)
from store import VectorStore, rrf_fuse

_REF_MARKER = "\n\n[참고 문서]\n"


def _build_context(hits: List[dict]) -> tuple[str, str]:
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
    print(f"[query] {step} — {detail}", flush=True)


def _strip_refs(text: str) -> str:
    idx = text.find(_REF_MARKER)
    return text[:idx].strip() if idx >= 0 else text.strip()


def _prepare_history(history: Optional[List[dict]]) -> List[dict[str, str]]:
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
    q = question.strip()
    if not history or len(q) > 80:
        return q
    for msg in reversed(history):
        if msg.get("role") == "user":
            prev = str(msg.get("content") or "").strip()
            if prev:
                return f"{prev} — 후속: {q}"
            break
    return q


def ask_question(
    question: str,
    *,
    think_mode: str = "instant",
    history: Optional[List[dict]] = None,
) -> str:
    question = question.strip()
    if not question:
        return "질문이 비어 있습니다."

    store = VectorStore()
    try:
        if store.count() == 0:
            return (
                "벡터 DB가 비어 있습니다. `data/docs`에 PDF를 넣고 "
                "`python app/main.py ingest --reset` 을 실행하세요."
            )

        llm_history = _prepare_history(history)
        if llm_history:
            _qlog("히스토리", f"이전 메시지 {len(llm_history)}개 LLM에 전달")

        # --- 1) dense + (선택) BM25 → RRF ---
        search_text = _embed_query_text(question, history)
        _qlog(
            "1/3 검색",
            f"임베딩 {EMBED_MODEL} · 검색문 {len(search_text)}자",
        )
        t0 = time.perf_counter()
        query_vec = embed_query(search_text)
        dense_hits = store.search(query_vec, top_k=FAISS_TOP_K)

        if HYBRID_BM25:
            bm25_hits = store.search_bm25(search_text, top_k=BM25_TOP_K)
            candidates = rrf_fuse(
                dense_hits,
                bm25_hits,
                k=HYBRID_RRF_K,
                top_n=FAISS_TOP_K,
            )
            _qlog(
                "1/3 검색",
                f"완료 {time.perf_counter() - t0:.2f}s · "
                f"dense={len(dense_hits)} BM25={len(bm25_hits)} "
                f"RRF={len(candidates)}",
            )
        else:
            candidates = dense_hits
            _qlog(
                "1/3 검색",
                f"완료 {time.perf_counter() - t0:.2f}s · "
                f"FAISS 후보 {len(candidates)}개 (BM25 off)",
            )

        if not candidates:
            return "검색 결과가 없습니다."

        # --- 2) 리랭크 (RERANK_TOP_K=0 이면 스킵 → RRF/dense 순위 그대로) ---
        if RERANK_TOP_K <= 0:
            hits = candidates
            _qlog(
                "2/3 리랭크",
                f"skip (RERANK_TOP_K=0) · 후보 {len(hits)}개 그대로 LLM",
            )
        else:
            _qlog(
                "2/3 리랭크",
                f"{RERANK_MODEL} · {len(candidates)} → top {RERANK_TOP_K}",
            )
            t1 = time.perf_counter()
            hits = rerank(search_text, candidates, top_k=RERANK_TOP_K)
            _qlog(
                "2/3 리랭크",
                f"완료 {time.perf_counter() - t1:.2f}s · 선별 {len(hits)}개",
            )

        context, ref_text = _build_context(hits)

        _, _, mode_label = resolve_think_mode(
            question, think_mode=think_mode  # type: ignore[arg-type]
        )
        _qlog(
            "3/3 LLM",
            f"llama-server {LLAMA_CPP_BASE_URL} · mode={mode_label}",
        )
        try:
            t2 = time.perf_counter()
            answer = chat_answer(
                question,
                context,
                think_mode=think_mode,  # type: ignore[arg-type]
                history=llm_history,
            )
            _qlog(
                "3/3 LLM",
                f"완료 {time.perf_counter() - t2:.2f}s · 답변 {len(answer)}자",
            )
        except Exception as e:
            _qlog("3/3 LLM", f"실패 — {type(e).__name__}: {e}")
            return (
                "답변 생성에 실패했습니다. llama-server 가 실행 중인지 확인하세요.\n"
                f"(원인: {type(e).__name__}: {e})\n\n"
                f"[참고 문서]\n{ref_text}"
            )

        return f"{answer}\n\n[참고 문서]\n{ref_text}"
    finally:
        store.close()
