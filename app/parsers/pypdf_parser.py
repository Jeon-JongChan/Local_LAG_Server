"""PyPDF 기반 경량 문서 파서 어댑터 구현체.

Java 런타임 없이 순수 파이썬(pypdf)으로 PDF 페이지별 텍스트를 초고속 추출합니다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from .base import BaseDocumentParserAdapter


class PyPdfParserAdapter(BaseDocumentParserAdapter):
    """pypdf 기반 경량 PDF 파서 어댑터."""

    name: str = "pypdf"

    def parse_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """pypdf를 통해 페이지별 텍스트를 추출."""
        try:
            from pypdf import PdfReader
        except ImportError as e:
            raise RuntimeError(
                "pypdf 패키지가 설치되지 않았습니다. (pip install pypdf)"
            ) from e

        reader = PdfReader(str(file_path))
        elements: List[Dict[str, Any]] = []

        for page_idx, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            text = text.strip()
            if not text:
                continue

            elements.append({
                "file_name": file_path.name,
                "source_path": str(file_path),
                "page": page_idx,
                "bbox": None,
                "type": "page_text",
                "text": text,
            })

        return elements

