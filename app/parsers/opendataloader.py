"""OpenDataLoader PDF 전용 문서 파서 어댑터 구현체.

opendataloader_pdf (Java 기반 레이아웃 분석 엔진)를 사용하여
제목, 문단, 표(HTML/Markdown 변환) 및 바운딩 박스를 추출합니다.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from .base import BaseDocumentParserAdapter

try:
    from settings import OPENDATALOADER_FORMAT, PARSED_DIR
except ModuleNotFoundError:
    from app.settings import OPENDATALOADER_FORMAT, PARSED_DIR

TEXT_ELEMENT_TYPES = frozenset(
    {"paragraph", "heading", "list", "list_item", "caption", "footnote"}
)


def _element_text(element: Dict[str, Any]) -> str:
    """JSON 요소에서 검색·임베딩용 순수 텍스트 추출."""
    content = element.get("content")
    if isinstance(content, str) and content.strip():
        return content.strip()

    if element.get("type") == "table":
        rows = element.get("rows") or element.get("kids") or []
        lines: List[str] = []
        for row in rows:
            if isinstance(row, dict):
                cell = row.get("content") or row.get("text")
                if cell:
                    lines.append(str(cell).strip())
                elif "kids" in row:
                    sub = [
                        str(k.get("content") or k.get("text") or "").strip()
                        for k in row["kids"]
                        if isinstance(k, dict)
                    ]
                    lines.append(" | ".join(filter(None, sub)))
            elif isinstance(row, list):
                sub = [str(c).strip() for c in row if str(c).strip()]
                lines.append(" | ".join(sub))
        return "\n".join(lines).strip()

    return ""


def _iter_elements(node: Any) -> Iterator[Dict[str, Any]]:
    """OpenDataLoader 결과 트리를 재귀 순회하여 유의미한 요소 추출."""
    if isinstance(node, dict):
        etype = node.get("type", "")
        if etype in TEXT_ELEMENT_TYPES or etype == "table":
            yield node
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
        try:
            import opendataloader_pdf
        except ImportError as e:
            raise RuntimeError(
                "opendataloader_pdf 패키지가 설치되지 않았습니다. "
                "(pip install opendataloader-pdf)"
            ) from e

        # 1. OpenDataLoader 변환 실행
        opendataloader_pdf.convert(
            input_path=str(file_path),
            output_dir=str(self.output_dir),
            format=self.output_format,
            quiet=True,
        )

        # 2. 결과 JSON 로드
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

        if not json_path:
            raise FileNotFoundError(f"OpenDataLoader 출력 JSON을 찾을 수 없습니다: {stem}")

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 3. 텍스트 요소 순회 및 정제
        elements: List[Dict[str, Any]] = []
        for el in _iter_elements(data):
            text = _element_text(el)
            if not text:
                continue

            page = el.get("page")
            if page is not None:
                try:
                    page = int(page) + 1  # 0-indexed -> 1-indexed
                except (ValueError, TypeError):
                    page = None

            bbox = el.get("bbox")
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

