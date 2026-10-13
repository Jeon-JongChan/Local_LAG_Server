"""
질의(RAG) 오케스트레이션 모듈.

어댑터 및 팩토리 패턴을 기반으로:
  1) 임베딩 어댑터 (embeddings.get_embedding_adapter)
  2) RAG 스토리지 어댑터 (storage.get_rag_storage)
  3) 리랭커 어댑터 (rerankers.get_reranker_adapter)
  4) 서빙 엔진 어댑터 (serving.get_serving_adapter)
를 결합하여 질의응답 파이프라인을 조율합니다.
"""

from __future__ import annotations

import time
from typing import List, Optional

from chunker import format_citation
from embeddings import get_embedding_adapter
from rerankers import get_reranker_adapter
from serving_engines import get_serving_adapter
from storage import get_rag_storage, rrf_fuse
from settings import (
    BM25_TOP_K,
    CHAT_HISTORY_MAX_MESSAGES,
    EMBED_MODEL,
    FAISS_TOP_K,
    HYBRID_BM25,
    HYBRID_RRF_K,
    RERANK_STRATEGY,
    RERANK_TOP_K,
)

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
    print(f"[query] {step} - {detail}", flush=True)


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
                return f"{prev} - 후속: {q}"
            break
    return q


def ask_question(
    question: str,
    *,
    think_mode: str = "instant",
    history: Optional[List[dict]] = None,
) -> str:
    """질문에 대해 하이브리드 검색, 선택형 리랭킹 및 LLM 추론을 거쳐 답변을 반환합니다."""
    question = question.strip()
    if not question:
        return "질문이 비어 있습니다."

    store = get_rag_storage()
    try:
        if store.count() == 0:
            return (
                "벡터 DB가 비어 있습니다. `data/docs`에 PDF를 넣고 "
                "`python app/main.py ingest --reset` 을 실행하세요."
            )

        llm_history = _prepare_history(history)
        if llm_history:
            _qlog("히스토리", f"이전 메시지 {len(llm_history)}개 LLM에 전달")

        # --- 1) dense + (선택) BM25 -> RRF ---
        search_text = _embed_query_text(question, history)
        _qlog("1/3 검색", f"임베딩={EMBED_MODEL} · 검색문 {len(search_text)}자")
        t0 = time.perf_counter()

        embedder = get_embedding_adapter()
        query_vec = embedder.embed_query(search_text)
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
                f"완료 {time.perf_counter() - t0:.2f}s · dense={len(dense_hits)} BM25={len(bm25_hits)} RRF={len(candidates)}",
            )
        else:
            candidates = dense_hits
            _qlog("1/3 검색", f"완료 {time.perf_counter() - t0:.2f}s · FAISS 후보 {len(candidates)}개 (BM25 off)")

        if not candidates:
            return "검색 결과가 없습니다."

        # --- 2) 리랭킹 (어댑터 팩토리 호출) ---
        if RERANK_TOP_K <= 0:
            hits = candidates
            _qlog("2/3 리랭크", f"skip (RERANK_TOP_K=0) · 후보 {len(hits)}개 그대로 LLM")
        else:
            _qlog("2/3 리랭크", f"전략={RERANK_STRATEGY} · {len(candidates)} -> top {RERANK_TOP_K}")
            t1 = time.perf_counter()
            reranker = get_reranker_adapter()
            hits = reranker.rerank(search_text, candidates, top_k=RERANK_TOP_K)
            _qlog("2/3 리랭크", f"완료 {time.perf_counter() - t1:.2f}s · 선별 {len(hits)}개")

        context, ref_text = _build_context(hits)

        # --- 3) 서빙 엔진 어댑터 호출 ---
        serving_engine = get_serving_adapter()
        _, _, mode_label = serving_engine.resolve_think_mode(question, think_mode=think_mode)
        _qlog("3/3 LLM", f"엔진={type(serving_engine).__name__} ({serving_engine.base_url}) · mode={mode_label}")

        try:
            t2 = time.perf_counter()
            answer = serving_engine.generate(
                question=question,
                context=context,
                think_mode=think_mode,
                history=llm_history,
            )
            _qlog("3/3 LLM", f"완료 {time.perf_counter() - t2:.2f}s · 답변 {len(answer)}자")
        except Exception as e:
            _qlog("3/3 LLM", f"실패 - {type(e).__name__}: {e}")
            return (
                "답변 생성에 실패했습니다. 서빙 엔진이 실행 중인지 확인하세요.\n"
                f"(원인: {type(e).__name__}: {e})\n\n"
                f"[참고 문서]\n{ref_text}"
            )

        return f"{answer}\n\n[참고 문서]\n{ref_text}"
    finally:
        store.close()
