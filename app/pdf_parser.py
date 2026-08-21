"""
PDF 파싱 모듈 — OpenDataLoader PDF → 구조화 JSON → 텍스트 요소 목록.

역할:
  1. data/docs 아래 PDF 탐색
  2. opendataloader_pdf.convert() 로 JSON 생성 (Java 엔진)
  3. JSON 트리에서 문단·제목·표 등을 순회하며 text + page + bbox 추출
  4. 손상·파싱 실패 PDF 는 data/rejected 로 이동
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional

from settings import DOCS_DIR, OPENDATALOADER_FORMAT, PARSED_DIR, REJECTED_DIR

# OpenDataLoader JSON 에서 RAG에 쓸 텍스트 요소 타입 (표는 별도 처리)
TEXT_ELEMENT_TYPES = frozenset(
    {"paragraph", "heading", "list", "list_item", "caption", "footnote"}
)

# docs 탐색 시 건너뛸 하위 폴더명
_SKIP_DOC_DIR_NAMES = frozenset({"_rejected"})

# OpenDataLoader stderr 에서 파일명을 추출할 때 쓰는 패턴
_INVALID_PDF_RE = re.compile(
    r"Error:\s*'([^']+)'\s+is not a valid PDF file",
    re.IGNORECASE,
)


@dataclass
class ParseBatchResult:
    elements: List[Dict[str, Any]] = field(default_factory=list)
    quarantined: List[tuple[str, str]] = field(default_factory=list)


def _element_text(element: Dict[str, Any]) -> str:
    """
    JSON 요소 하나에서 임베딩·검색에 쓸 순수 텍스트를 꺼냅니다.

    - 일반 요소: "content" 문자열
    - 표(table): rows/kids 를 순회해 셀을 " | " 로 이어 한 줄씩 만듦
    """
    content = element.get("content")
    if isinstance(content, str) and content.strip():
        return content.strip()

    if element.get("type") == "table":
        rows = element.get("rows") or element.get("kids") or []
        lines: List[str] = []
        for row in rows:
            if isinstance(row, dict):
                cell = row.get("content") or row.get("text")
                if isinstance(cell, str) and cell.strip():
                    lines.append(cell.strip())
            elif isinstance(row, list):
                cells = [str(c).strip() for c in row if str(c).strip()]
                if cells:
                    lines.append(" | ".join(cells))
        if lines:
            return "\n".join(lines)

    return ""


def _iter_elements(node: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    """JSON 문서 트리를 깊이 우선으로 순회하며 텍스트가 있는 요소만 yield."""
    kids = node.get("kids")
    if isinstance(kids, list):
        for kid in kids:
            if isinstance(kid, dict):
                yield from _iter_elements(kid)

    etype = node.get("type")
    if etype in TEXT_ELEMENT_TYPES or etype == "table":
        text = _element_text(node)
        if text:
            yield node


def _page_number(element: Dict[str, Any]) -> int | None:
    for key in ("page number", "page_number", "pageNumber", "page"):
        val = element.get(key)
        if isinstance(val, int):
            return val
        if isinstance(val, str) and val.isdigit():
            return int(val)
    return None


def _bounding_box(element: Dict[str, Any]) -> List[float] | None:
    for key in ("bounding box", "bounding_box", "bbox"):
        val = element.get(key)
        if isinstance(val, list) and len(val) >= 4:
            try:
                return [float(v) for v in val[:4]]
            except (TypeError, ValueError):
                return None
    return None


def elements_from_json(doc: Dict[str, Any], source_path: Path) -> List[Dict[str, Any]]:
    """OpenDataLoader JSON 문서 전체를 RAG용 요소 dict 리스트로 변환합니다."""
    file_name = doc.get("file name") or doc.get("file_name") or source_path.name
    out: List[Dict[str, Any]] = []

    for element in _iter_elements(doc):
        text = _element_text(element)
        if not text:
            continue
        out.append(
            {
                "text": text,
                "file_name": file_name,
                "source_path": str(source_path),
                "page": _page_number(element),
                "bbox": _bounding_box(element),
                "element_type": element.get("type"),
            }
        )
    return out


def _load_json_doc(json_path: Path) -> Dict[str, Any]:
    with json_path.open(encoding="utf-8") as f:
        return json.load(f)


def _rel_key(path: Path, docs_dir: Path = DOCS_DIR) -> str:
    return path.resolve().relative_to(docs_dir.resolve()).as_posix()


def is_valid_pdf_header(path: Path) -> tuple[bool, str]:
    """PDF 매직 바이트(%PDF-) 확인. (ok, reason) 반환."""
    try:
        with path.open("rb") as f:
            header = f.read(8)
    except OSError as e:
        return False, f"파일 읽기 실패: {e}"

    if header.startswith(b"%PDF-"):
        return True, ""
    return False, "유효한 PDF 헤더가 없습니다 (%PDF- 누락)"


def quarantine_pdf(
    pdf_path: Path,
    reason: str,
    *,
    docs_dir: Path = DOCS_DIR,
) -> Path:
    """
    문제 PDF 를 data/rejected/<docs 상대경로> 로 이동하고 사유 파일을 남깁니다.
    """
    pdf_path = pdf_path.resolve()
    docs_dir = docs_dir.resolve()
    rel = _rel_key(pdf_path, docs_dir)
    dest = REJECTED_DIR / rel
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        stem = dest.stem
        suffix = dest.suffix
        parent = dest.parent
        n = 2
        while dest.exists():
            dest = parent / f"{stem}_{n}{suffix}"
            n += 1

    shutil.move(str(pdf_path), str(dest))

    reason_path = dest.with_suffix(dest.suffix + ".reason.txt")
    reason_path.write_text(
        "\n".join(
            [
                f"path: {rel}",
                f"time: {datetime.now(timezone.utc).isoformat()}",
                f"reason: {reason.strip()}",
            ]
        ),
        encoding="utf-8",
    )
    print(f"[reject] {rel} → {dest.relative_to(REJECTED_DIR)}")
    return dest


def _safe_rmtree(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)


def _find_json_for_pdf(work_dir: Path, pdf_path: Path) -> Path | None:
    json_path = work_dir / f"{pdf_path.stem}.json"
    if json_path.is_file():
        return json_path
    matches = sorted(work_dir.glob(f"{pdf_path.stem}*.json"))
    return matches[0] if matches else None


def _convert_pdfs_to_work_dir(pdf_paths: List[Path], work_dir: Path) -> None:
    import opendataloader_pdf

    work_dir.mkdir(parents=True, exist_ok=True)
    opendataloader_pdf.convert(
        input_path=[str(p) for p in pdf_paths],
        output_dir=str(work_dir),
        format=OPENDATALOADER_FORMAT,
        quiet=True,
    )


def _elements_from_work_dir(pdf_path: Path, work_dir: Path) -> List[Dict[str, Any]]:
    json_path = _find_json_for_pdf(work_dir, pdf_path)
    if json_path is None:
        raise FileNotFoundError(
            f"OpenDataLoader JSON 출력을 찾을 수 없습니다: {pdf_path.name}"
        )
    doc = _load_json_doc(json_path)
    return elements_from_json(doc, pdf_path)


def _parse_error_reason(exc: BaseException) -> str:
    parts = [str(exc)]
    for attr in ("stderr", "output", "stdout"):
        val = getattr(exc, attr, None)
        if val:
            parts.append(str(val))
    text = "\n".join(parts)
    match = _INVALID_PDF_RE.search(text)
    if match:
        return f"유효한 PDF가 아님: {match.group(1)}"
    return text.strip() or exc.__class__.__name__


def _parse_single_pdf(pdf_path: Path) -> List[Dict[str, Any]]:
    work_dir = PARSED_DIR / "odl-work" / pdf_path.stem
    _safe_rmtree(work_dir)
    try:
        _convert_pdfs_to_work_dir([pdf_path], work_dir)
        return _elements_from_work_dir(pdf_path, work_dir)
    finally:
        _safe_rmtree(work_dir)


def parse_pdf(pdf_path: Path, *, work_dir: Path | None = None) -> List[Dict[str, Any]]:
    """단일 PDF 1개를 OpenDataLoader로 변환합니다."""
    pdf_path = pdf_path.resolve()
    if work_dir is None:
        return _parse_single_pdf(pdf_path)

    work_dir.mkdir(parents=True, exist_ok=True)
    _convert_pdfs_to_work_dir([pdf_path], work_dir)
    return _elements_from_work_dir(pdf_path, work_dir)


def parse_pdfs_batch(
    pdf_paths: List[Path],
    *,
    on_progress: Optional[Callable[[int, int, str], None]] = None,
) -> ParseBatchResult:
    """
    여러 PDF를 파싱합니다. 실패 파일은 data/rejected 로 분리하고 나머지는 계속 처리합니다.

    on_progress(done, total, detail) — 파일 단위 진행률 콜백.
    """
    result = ParseBatchResult()
    if not pdf_paths:
        return result

    pdf_paths = [p.resolve() for p in pdf_paths]
    pending: List[Path] = []

    for pdf_path in pdf_paths:
        ok, reason = is_valid_pdf_header(pdf_path)
        if ok:
            pending.append(pdf_path)
            continue
        quarantine_pdf(pdf_path, reason)
        result.quarantined.append((_rel_key(pdf_path, DOCS_DIR), reason))

    if not pending:
        return result

    total = len(pending)
    for idx, pdf_path in enumerate(pending, 1):
        if on_progress:
            on_progress(idx - 1, total, pdf_path.name)
        try:
            result.elements.extend(_parse_single_pdf(pdf_path))
        except Exception as exc:
            reason = _parse_error_reason(exc)
            quarantine_pdf(pdf_path, reason)
            result.quarantined.append((_rel_key(pdf_path, DOCS_DIR), reason))

    if on_progress:
        on_progress(total, total, "완료")

    return result


def discover_pdf_files(docs_dir: Path) -> List[Path]:
    """
    docs_dir 이하 모든 .pdf/.PDF 파일을 재귀 탐색합니다.

    - 숨김 파일(.으로 시작) 제외
    - _rejected 하위 폴더 제외
    """
    if not docs_dir.is_dir():
        return []
    found: List[Path] = []
    for path in docs_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.name.startswith("."):
            continue
        if path.suffix.lower() != ".pdf":
            continue
        rel_parts = path.relative_to(docs_dir).parts
        if any(part in _SKIP_DOC_DIR_NAMES for part in rel_parts[:-1]):
            continue
        found.append(path)
    return sorted(found)


def clear_parsed_cache() -> None:
    """data/parsed 캐시 폴더를 비웁니다."""
    if PARSED_DIR.is_dir():
        shutil.rmtree(PARSED_DIR, ignore_errors=True)
    PARSED_DIR.mkdir(parents=True, exist_ok=True)
