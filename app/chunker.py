"""
청크 분할 모듈 — OpenDataLoader 요소 → RAG 검색 단위(청크).

OpenDataLoader는 문단·제목 등 작은 단위로 텍스트를 뽑습니다.
이 모듈은 그것을:
  - 너무 작으면 병합 (CHUNK_MIN_CHARS)
  - 너무 크면 분할 (CHUNK_MAX_CHARS + overlap)
하고 page/bbox 메타데이터를 청크에 붙입니다.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from settings import CHUNK_MAX_CHARS, CHUNK_MIN_CHARS, CHUNK_OVERLAP_CHARS


def _split_long_text(text: str, max_chars: int, overlap: int) -> List[str]:
    """
    max_chars 를 넘는 긴 문자열을 겹침(overlap)을 두고 여러 조각으로 자릅니다.

    RAG에서 한 문단이 800자를 넘을 때 검색·임베딩 단위를 나누기 위함.
    """
    if len(text) <= max_chars:
        return [text]

    parts: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        parts.append(text[start:end].strip())
        if end >= len(text):
            break
        # 다음 조각은 overlap 만큼 앞에서부터 다시 시작 (문맥 연결)
        start = max(0, end - overlap)

    return [p for p in parts if p]


def _merge_meta(
    base: Dict[str, Any], pages: List[int | None], bboxes: List[List[float] | None]
) -> Dict[str, Any]:
    """
    버퍼에 모인 여러 요소의 page/bbox 를 청크 하나의 메타로 합칩니다.

    - page: 첫 페이지를 대표값으로, 여러 페이지면 pages 리스트 추가
    - bbox: 첫 bbox 를 대표값으로, 여러 개면 bboxes 리스트 추가
    """
    meta = dict(base)

    unique_pages = sorted({p for p in pages if p is not None})
    if unique_pages:
        meta["page"] = unique_pages[0]
        if len(unique_pages) > 1:
            meta["pages"] = unique_pages

    valid_bboxes = [b for b in bboxes if b is not None]
    if valid_bboxes:
        meta["bbox"] = valid_bboxes[0]
        if len(valid_bboxes) > 1:
            meta["bboxes"] = valid_bboxes

    return meta


def elements_to_chunks(elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    OpenDataLoader 요소 리스트 → 최종 청크 리스트.

    알고리즘:
      1. 요소 텍스트를 buffer 에 누적
      2. buffer 길이 >= CHUNK_MIN_CHARS 이면 flush (청크 확정)
      3. 누적 시 CHUNK_MAX_CHARS 초과 예상이면 먼저 flush 후 새 buffer
      4. 마지막 남은 buffer 도 flush

    반환 각 항목: { text, file_name, source_path, page, bbox, element_type, ... }
    """
    chunks: List[Dict[str, Any]] = []

    # 아직 확정되지 않은 텍스트·메타를 모아 두는 버퍼
    buffer_text = ""
    buffer_pages: List[int | None] = []
    buffer_bboxes: List[List[float] | None] = []
    buffer_meta: Dict[str, Any] = {}

    def flush_buffer() -> None:
        """버퍼 내용을 하나 이상의 청크로 만들고 chunks 에 append."""
        nonlocal buffer_text, buffer_pages, buffer_bboxes, buffer_meta

        text = buffer_text.strip()
        if not text:
            # 빈 버퍼면 초기화만
            buffer_text = ""
            buffer_pages = []
            buffer_bboxes = []
            buffer_meta = {}
            return

        # page/bbox 등 버퍼에 쌓인 메타 병합
        meta = _merge_meta(buffer_meta, buffer_pages, buffer_bboxes)

        # 800자 초과분은 overlap 분할
        for piece in _split_long_text(text, CHUNK_MAX_CHARS, CHUNK_OVERLAP_CHARS):
            chunks.append({"text": piece, **meta})

        # 버퍼 비우기
        buffer_text = ""
        buffer_pages = []
        buffer_bboxes = []
        buffer_meta = {}

    for el in elements:
        text = (el.get("text") or "").strip()
        if not text:
            continue

        page = el.get("page")
        bbox = el.get("bbox")

        # 버퍼가 비어 있으면 이 요소의 파일 정보로 메타 시작
        if not buffer_text:
            buffer_meta = {
                "file_name": el.get("file_name", "unknown"),
                "source_path": el.get("source_path", ""),
                "element_type": el.get("element_type"),
            }

        # 기존 buffer + 새 텍스트 합친 길이
        candidate = f"{buffer_text}\n{text}".strip() if buffer_text else text

        # 합치면 max 초과 → 지금 buffer 먼저 flush, 새 텍스트로 새 buffer
        if buffer_text and len(candidate) > CHUNK_MAX_CHARS:
            flush_buffer()
            buffer_text = text
            buffer_pages = [page]
            buffer_bboxes = [bbox]
            buffer_meta = {
                "file_name": el.get("file_name", "unknown"),
                "source_path": el.get("source_path", ""),
                "element_type": el.get("element_type"),
            }
        else:
            # 아직 max 이하면 buffer 에 계속 누적
            buffer_text = candidate
            buffer_pages.append(page)
            buffer_bboxes.append(bbox)

        # min 길이에 도달하면 청크로 확정
        if len(buffer_text) >= CHUNK_MIN_CHARS:
            flush_buffer()

    # 루프 끝 후 남은 buffer 처리 (min 미만이어도 버림 없이 저장)
    flush_buffer()
    return chunks


def format_citation(meta: Dict[str, Any]) -> str:
    """
    검색 결과·답변 하단에 표시할 출처 한 줄 문자열을 만듭니다.

    예: "report.pdf · p.3~4 · bbox=[72.0, 450.0, 500.0, 480.0] · chunks=200~202"
    """
    parts = [meta.get("file_name", "unknown")]

    page_str = meta.get("page_str")
    if page_str:
        parts.append(page_str)
    else:
        page = meta.get("page")
        if page is not None:
            parts.append(f"p.{page}")

    bbox = meta.get("bbox")
    if bbox:
        parts.append(f"bbox={json.dumps(bbox)}")

    chunk_ids = meta.get("chunk_ids")
    if chunk_ids and len(chunk_ids) > 1:
        parts.append(f"chunks={chunk_ids[0]}~{chunk_ids[-1]}")
    elif meta.get("id") is not None:
        parts.append(f"chunk={meta['id']}")

    return " · ".join(parts)
