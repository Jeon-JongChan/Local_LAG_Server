"""OpenDataLoader PDF 전용 문서 파서 어댑터 구현체.

opendataloader_pdf (Java 기반 레이아웃 분석 엔진)를 사용하여
제목, 문단, 표(HTML/Markdown 변환) 및 바운딩 박스를 추출합니다.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from .base import BaseDocumentParserAdapter
from settings import OPENDATALOADER_FORMAT, PARSED_DIR

TEXT_ELEMENT_TYPES = frozenset(
    {"paragraph", "heading", "list", "list_item", "caption", "footnote"}
)


def _extract_text_from_node(node: Any) -> str:
    """노드 또는 하위 노드에서 텍스트를 재귀적으로 수집."""
    if isinstance(node, str):
        return node.strip()
    if isinstance(node, dict):
        texts: List[str] = []
        if "content" in node and isinstance(node["content"], str) and node["content"].strip():
            texts.append(node["content"].strip())
        elif "text" in node and isinstance(node["text"], str) and node["text"].strip():
            texts.append(node["text"].strip())

        for child in node.get("kids") or node.get("children") or []:
            child_text = _extract_text_from_node(child)
            if child_text:
                texts.append(child_text)
        return " ".join(texts).strip()
    if isinstance(node, list):
        return " ".join(filter(None, (_extract_text_from_node(item) for item in node))).strip()
    return ""


def _element_text(element: Dict[str, Any]) -> str:
    """JSON 요소에서 검색·임베딩용 순수 텍스트 추출 (표 포함)."""
    if element.get("type") == "table":
        rows = element.get("rows") or element.get("kids") or []
        lines: List[str] = []
        for row in rows:
            if isinstance(row, dict):
                cells = row.get("cells") or row.get("kids") or []
                if cells:
                    row_cells = []
                    for cell in cells:
                        cell_str = _extract_text_from_node(cell).replace("\n", " ")
                        row_cells.append(cell_str)
                    if any(row_cells):
                        lines.append(" | ".join(row_cells))
                else:
                    cell = row.get("content") or row.get("text")
                    if cell:
                        lines.append(str(cell).strip())
            elif isinstance(row, list):
                row_cells = [_extract_text_from_node(c).replace("\n", " ") for c in row]
                if any(row_cells):
                    lines.append(" | ".join(row_cells))
        return "\n".join(lines).strip()

    content = element.get("content")
    if isinstance(content, str) and content.strip():
        return content.strip()

    return ""


def _iter_elements(node: Any) -> Iterator[Dict[str, Any]]:
    """OpenDataLoader 결과 트리를 재귀 순회하여 유의미한 요소 추출."""
    if isinstance(node, dict):
        etype = node.get("type", "")
        if etype in TEXT_ELEMENT_TYPES or etype == "table":
            yield node
        # table의 경우 내부 셀/문단은 _element_text에서 한꺼번에 표 텍스트로 처리하므로 중복 순회 방지
        if etype != "table":
            for child in node.get("kids") or node.get("children") or []:
                yield from _iter_elements(child)
    elif isinstance(node, list):
        for item in node:
            yield from _iter_elements(item)


class OpenDataLoaderParserAdapter(BaseDocumentParserAdapter):
    """OpenDataLoader PDF 엔진 어댑터."""

    name: str = "opendataloader"

    def __init__(self, output_dir: Path = PARSED_DIR, output_format: str = OPENDATALOADER_FORMAT) -> None:
        self.output_dir = output_dir
        self.output_format = output_format
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def parse_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """단일 PDF 파싱 및 텍스트 요소 추출."""
        stem = file_path.stem
        candidates = [
            self.output_dir / f"{stem}.json",
            self.output_dir / stem / f"{stem}.json",
            self.output_dir / stem / "output.json",
        ]
        json_path: Optional[Path] = None
        for cand in candidates:
            if cand.is_file():
                json_path = cand
                break

        if not json_path:
            for f in self.output_dir.glob(f"{stem}*.json"):
                json_path = f
                break

        # JSON 캐시가 없으면 OpenDataLoader 변환 실행
        if not json_path or not json_path.is_file():
            try:
                import opendataloader_pdf
            except ImportError as e:
                raise RuntimeError(
                    "opendataloader_pdf 패키지가 설치되지 않았습니다. "
                    "(pip install opendataloader-pdf)"
                ) from e

            opendataloader_pdf.convert(
                input_path=str(file_path),
                output_dir=str(self.output_dir),
                format=self.output_format,
                quiet=True,
            )

            for cand in candidates:
                if cand.is_file():
                    json_path = cand
                    break
            if not json_path:
                for f in self.output_dir.glob(f"{stem}*.json"):
                    json_path = f
                    break

        if not json_path or not json_path.is_file():
            raise FileNotFoundError(f"OpenDataLoader 출력 JSON을 찾을 수 없습니다: {stem}")

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 텍스트 요소 순회 및 정제
        elements: List[Dict[str, Any]] = []
        for el in _iter_elements(data):
            text = _element_text(el)
            if not text:
                continue

            raw_page = el.get("page number") if "page number" in el else el.get("page_number")
            if raw_page is not None:
                try:
                    page = int(raw_page)
                except (ValueError, TypeError):
                    page = None
            elif el.get("page") is not None:
                try:
                    page = int(el["page"]) + 1  # 0-indexed -> 1-indexed
                except (ValueError, TypeError):
                    page = None
            else:
                page = None

            bbox = el.get("bounding box") if "bounding box" in el else el.get("bbox")
            bbox_str = json.dumps(bbox) if bbox is not None else None

            elements.append({
                "file_name": file_path.name,
                "source_path": str(file_path),
                "page": page,
                "bbox": bbox_str,
                "type": el.get("type", "paragraph"),
                "text": text,
            })

        return elements
