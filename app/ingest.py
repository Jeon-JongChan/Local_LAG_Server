"""인제스트 Application Service (오케스트레이터).

패턴 밖 파이프라인 조율자. 하위 도메인은 Factory로 어댑터를 받아 쓴다.

흐름:
  discover → 변경 감지 → Parser(Strategy) → 청킹 → Embedding → RagStorage(Facade)

CLI: python app/main.py ingest [--reset] [--parser opendataloader|pypdf]
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from chunker import elements_to_chunks
from embeddings import get_embedding_adapter
from parsers import clear_parsed_cache, discover_pdf_files, get_document_parser
from progress import IngestProgress
from settings import DOCS_DIR, EMBED_MODEL, INDEX_DIR, REJECTED_DIR
from storage import RagStorage, get_rag_storage

CHUNKS_CACHE_FILE = INDEX_DIR / "chunks_cache.json"
EMBED_CHECKPOINT_FILE = INDEX_DIR / "embedding_checkpoint.npy"


def _embed_label() -> str:
    """진행률용 짧은 모델명."""
    return EMBED_MODEL.rsplit("/", 1)[-1] or EMBED_MODEL


def _file_sig(path: Path) -> Tuple[int, float]:
    """파일 변경 감지용 시그니처: (바이트 크기, 수정 시각 mtime)."""
    st = path.stat()
    return int(st.st_size), float(st.st_mtime)


def _rel_key(path: Path) -> str:
    """data/docs 기준 상대 경로 (POSIX 슬래시)."""
    return path.relative_to(DOCS_DIR).as_posix().replace("\\", "/")


def _plan_ingest(
    pdfs: List[Path], store: RagStorage, *, reset: bool
) -> Tuple[str, List[Path]]:
    """인제스트 모드 결정: full | append | skip."""
    if reset:
        return "full", pdfs

    if store.count() == 0:
        return "full", pdfs

    prev = store.get_manifest()
    targets: List[Path] = []
    for p in pdfs:
        rel = _rel_key(p)
        sz, mt = _file_sig(p)
        old = prev.get(rel)
        if old is None or old[0] != sz or abs(old[1] - mt) > 1e-4:
            targets.append(p)

    if not targets:
        return "skip", []
    return "append", targets


def ingest_documents(
    reset: bool = False,
    parser_type: Optional[str] = None,
    fresh: bool = False,
) -> None:
    """
    data/docs 아래 문서를 탐색하여 변경된 파일만 증분 인제스트합니다.

    Args:
        reset: True 시 기존 인덱스를 완전히 비우고 전체 재구축
        parser_type: 'opendataloader' | 'pypdf' (기본: 설정값 또는 opendataloader)
        fresh: True 시 기존 청크 캐시 및 임베딩 체크포인트를 초기화하고 처음부터 강제 실행
    """
    # 1) 파일 탐색
    pdfs = discover_pdf_files()
    if not pdfs:
        print("[ingest] data/docs 에 PDF 가 없습니다.")
        return

    store = get_rag_storage()
    try:
        # 2) 인제스트 모드 판정
        mode, targets = _plan_ingest(pdfs, store, reset=reset)
        if mode == "skip":
            print(f"[OK] 변경 없음 - 청크 {store.count()}개 유지")
            return

        if fresh:
            clear_parsed_cache()
            if CHUNKS_CACHE_FILE.is_file():
                CHUNKS_CACHE_FILE.unlink(missing_ok=True)
            if EMBED_CHECKPOINT_FILE.is_file():
                EMBED_CHECKPOINT_FILE.unlink(missing_ok=True)

        if mode == "append":
            print(f"[ingest] 증분 인제스트 - 대상 {len(targets)}/{len(pdfs)}개")
        else:
            print("[ingest] 전체 재구축")
            targets = pdfs

        target_keys = sorted([_rel_key(p) for p in targets])

        # 3) 청크 캐시 확인 (1단계 파싱, 2단계 청킹 건너뛰기 지원)
        chunks: Optional[List[Dict[str, Any]]] = None
        if not fresh and CHUNKS_CACHE_FILE.is_file():
            try:
                with open(CHUNKS_CACHE_FILE, "r", encoding="utf-8") as f:
                    cache_payload = json.load(f)
                if (
                    cache_payload.get("target_keys") == target_keys
                    and isinstance(cache_payload.get("chunks"), list)
                    and len(cache_payload["chunks"]) > 0
                ):
                    chunks = cache_payload["chunks"]
                    print(
                        f"[ingest] 청크 캐시 발견 (총 {len(chunks)}개) - "
                        "1단계(파싱) 및 2단계(청킹)를 건너뛰고 3단계(임베딩)부터 재개합니다."
                    )
            except Exception:
                chunks = None

        if chunks is None:
            # 3-1) 문서 파서 어댑터를 통한 파싱 실행
            doc_parser = get_document_parser(parser_type)
            progress = IngestProgress()
            progress.start_stage(1, f"문서 파싱 ({doc_parser.name})", len(targets))

            def _pdf_progress(done: int, total: int, detail: str) -> None:
                progress.update(done, total, detail)

            parse_result = doc_parser.parse_batch(targets, on_progress=_pdf_progress)
            progress.finish_stage(
                f"요소 {len(parse_result.elements)}개, "
                f"분리 {len(parse_result.quarantined)}개"
            )
            if parse_result.quarantined:
                print(
                    f"[ingest] 문제 문서 {len(parse_result.quarantined)}개 분리 -> {REJECTED_DIR}"
                )
            elements = parse_result.elements
            if not elements:
                if parse_result.quarantined:
                    raise SystemExit(
                        f"처리 가능한 문서가 없습니다. 분리된 파일: {REJECTED_DIR}"
                    )
                print("[ingest] 추출된 텍스트 요소가 없습니다.")
                return

            # 4) 청킹
            progress.start_stage(2, "청크 생성", len(elements))
            chunks = elements_to_chunks(elements)
            progress.finish_stage(f"청크 {len(chunks)}개")
            if not chunks:
                print("[ingest] 생성된 청크가 없습니다.")
                return

            # 청크 캐시 파일 저장 (추후 중단 시 3단계부터 즉시 재개 가능)
            try:
                INDEX_DIR.mkdir(parents=True, exist_ok=True)
                with open(CHUNKS_CACHE_FILE, "w", encoding="utf-8") as f:
                    json.dump(
                        {"target_keys": target_keys, "chunks": chunks},
                        f,
                        ensure_ascii=False,
                    )
            except Exception:
                pass

        # 5) 임베딩 생성 (어댑터 템플릿 호출 + 체크포인트 재개)
        embedder = get_embedding_adapter()
        texts = [c["text"] for c in chunks]
        progress = IngestProgress()
        progress.start_stage(
            3,
            f"임베딩 ({_embed_label()} · {embedder.device})",
            len(texts),
        )

        def _embed_progress(done: int, total: int, detail: str) -> None:
            progress.update(done, total, detail)

        embeddings = embedder.embed_documents(
            texts,
            checkpoint_path=EMBED_CHECKPOINT_FILE,
            on_progress=_embed_progress,
        )
        progress.finish_stage(f"벡터 {len(embeddings)}개")

        # 6) 스토리지 저장 (FAISS + SQLite + FTS5)
        if mode != "append":
            store.reset()

        progress.start_stage(4, "인덱스 저장", len(chunks))
        store.add_chunks(chunks, embeddings)
        for p in targets:
            rel = _rel_key(p)
            sig = _file_sig(p)
            store.update_manifest(rel, sig[0], sig[1])
        progress.finish_stage("FAISS·SQLite 반영 완료")

        # 인제스트 성공 시 임시 캐시 및 체크포인트 정리
        try:
            if EMBED_CHECKPOINT_FILE.is_file():
                EMBED_CHECKPOINT_FILE.unlink(missing_ok=True)
            if CHUNKS_CACHE_FILE.is_file():
                CHUNKS_CACHE_FILE.unlink(missing_ok=True)
        except Exception:
            pass

        print(
            f"\n[OK] ingest 완료 - 청크 {store.count()}개 "
            f"(이번 추가: {len(chunks)}개)"
        )
    finally:
        store.close()
