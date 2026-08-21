"""
PDF 인제스트 오케스트레이션.

전체 흐름:
  discover PDF → (full/append/skip 결정) → OpenDataLoader → 청크 → embed → FAISS+SQLite+FTS5

CLI: python app/main.py ingest [--reset]
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

from chunker import elements_to_chunks
from models import current_embed_device, embed_texts
from pdf_parser import clear_parsed_cache, discover_pdf_files, parse_pdfs_batch
from progress import IngestProgress
from settings import DOCS_DIR, EMBED_MODEL, REJECTED_DIR
from store import VectorStore


def _embed_label() -> str:
    """진행률용 짧은 모델명 (Qwen/Qwen3-Embedding-0.6B → Qwen3-Embedding-0.6B)."""
    return EMBED_MODEL.rsplit("/", 1)[-1] or EMBED_MODEL


def _file_sig(path: Path) -> Tuple[int, float]:
    """
    파일 변경 감지용 시그니처: (바이트 크기, 수정 시각 mtime).

    ingest_manifest 와 비교해 PDF가 바뀌었는지 판단합니다.
    """
    st = path.stat()
    return int(st.st_size), float(st.st_mtime)


def _rel_key(path: Path) -> str:
    """
    data/docs 기준 상대 경로 (POSIX 슬래시).

    예: Windows docs\\a\\b.pdf → "a/b.pdf"
    """
    return path.relative_to(DOCS_DIR).as_posix().replace("\\", "/")


def _current_manifest(pdfs: List[Path]) -> Dict[str, Tuple[int, float]]:
    """현재 docs 폴더에 있는 모든 PDF의 rel_path → 시그니처 맵."""
    return {_rel_key(p): _file_sig(p) for p in pdfs}


def _plan_ingest(
    pdfs: List[Path], store: VectorStore, *, reset: bool
) -> Tuple[str, List[Path]]:
    """
    이번 인제스트를 full / append / skip 중 무엇으로 할지 결정합니다.

    Returns:
        (mode, targets)
        - mode: "full" | "append" | "skip"
        - targets: 실제로 parse_pdfs_batch 에 넘길 PDF 목록

    규칙:
        - reset=True → 무조건 full, 전체 PDF
        - manifest 와 현재 파일 목록·시그니처 동일 → skip
        - 파일 삭제·크기/mtime 변경 → full (FAISS 개별 삭제 미지원)
        - 신규 파일만 추가 → append (신규 PDF만 parse, FAISS에 add)
    """
    if reset:
        return "full", pdfs
    if not pdfs:
        return "skip", []
    if store.count() == 0:
        return "full", pdfs

    current = _current_manifest(pdfs)
    prev = store.get_manifest()

    # 완전 동일하면 재인제스트 불필요
    if current == prev:
        return "skip", []

    prev_keys = set(prev)
    curr_keys = set(current)
    added = curr_keys - prev_keys       # 새로 생긴 PDF
    removed = prev_keys - curr_keys     # 삭제된 PDF
    changed = {
        k for k in curr_keys & prev_keys if current[k] != prev[k]
    }  # 크기·mtime 변경

    if removed or changed:
        return "full", pdfs
    return "append", [p for p in pdfs if _rel_key(p) in added]


def ingest_documents(*, reset: bool = False) -> None:
    """
    data/docs PDF를 색인하는 메인 진입 함수.

    Args:
        reset: True 이면 FAISS·SQLite·parsed 캐시 전부 지우고 처음부터 구축
    """
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    pdfs = discover_pdf_files(DOCS_DIR)
    if not pdfs:
        raise SystemExit(
            f"PDF가 없습니다. `{DOCS_DIR}` 에 .pdf 파일을 넣으세요."
        )

    store = VectorStore()
    try:
        # 1) full / append / skip 결정
        mode, targets = _plan_ingest(pdfs, store, reset=reset)
        if mode == "skip":
            print(f"[OK] 변경 없음 — 청크 {store.count()}개 유지")
            return

        # 2) full 이면 기존 인덱스·파싱 캐시 삭제
        if mode == "full":
            print("[ingest] 전체 재구축")
            store.reset()
            clear_parsed_cache()
            targets = pdfs

        # 3) OpenDataLoader 파싱 → 요소 리스트 (실패 PDF 는 data/rejected 로 분리)
        progress = IngestProgress()
        progress.start_stage(1, "PDF 파싱 (OpenDataLoader)", len(targets))

        def _pdf_progress(done: int, total: int, detail: str) -> None:
            progress.update(done, total, detail)

        parse_result = parse_pdfs_batch(targets, on_progress=_pdf_progress)
        progress.finish_stage(
            f"요소 {len(parse_result.elements)}개, "
            f"분리 {len(parse_result.quarantined)}개"
        )
        if parse_result.quarantined:
            print(
                f"[ingest] 문제 PDF {len(parse_result.quarantined)}개 분리 → {REJECTED_DIR}"
            )
        elements = parse_result.elements
        if not elements:
            if parse_result.quarantined:
                raise SystemExit(
                    f"처리 가능한 PDF가 없습니다. 분리된 파일: {REJECTED_DIR}"
                )
            raise SystemExit("추출된 텍스트 요소가 없습니다.")

        # 4) 요소 → RAG 청크
        progress.start_stage(2, "청크 생성", len(elements))
        chunks = elements_to_chunks(elements)
        progress.finish_stage(f"청크 {len(chunks)}개")
        if not chunks:
            raise SystemExit("생성된 청크가 없습니다.")

        # 5) 임베딩 (EMBED_MODEL)
        progress.start_stage(
            3, f"임베딩 ({_embed_label()}, {current_embed_device()})", len(chunks)
        )
        texts = [c["text"] for c in chunks]

        def _embed_progress(done: int, total: int, detail: str) -> None:
            progress.update(done, total, detail)

        embeddings = embed_texts(texts, on_progress=_embed_progress)
        progress.finish_stage(f"벡터 {len(embeddings)}개")

        # 6) FAISS + SQLite 저장
        progress.start_stage(4, "인덱스 저장", len(chunks))
        store.add_chunks(chunks, embeddings)
        progress.finish_stage(f"FAISS·SQLite 반영 완료")

        # 7) manifest 갱신 (격리된 PDF 제외, docs 에 남은 파일만)
        pdfs = discover_pdf_files(DOCS_DIR)
        current = _current_manifest(pdfs)
        for rel, sig in current.items():
            store.update_manifest(rel, sig[0], sig[1])

        print(
            f"[OK] ingest 완료 — 청크 {store.count()}개 "
            f"(모드={mode}, PDF={len(targets)})"
        )
    finally:
        store.close()
