"""문서 파서 팩토리.

Strategy: DOC_PARSER(또는 인자)로 opendataloader | pypdf 선택.
Adapter: 각 구현이 외부 파서 라이브러리를 공통 element 규약으로 맞춤.
캐시 없음 — 파서는 상태 없어 매 호출 신규 인스턴스(serving/reranker 와 정책이 다름).
"""

from __future__ import annotations

from typing import Optional

from .base import BaseDocumentParserAdapter
from .opendataloader import OpenDataLoaderParserAdapter
from .pypdf_parser import PyPdfParserAdapter
from settings import DOC_PARSER


def get_document_parser(parser_type: Optional[str] = None) -> BaseDocumentParserAdapter:
    """문서 파서 팩토리. 미지정 시 settings.DOC_PARSER."""
    target_type = (parser_type or DOC_PARSER or "opendataloader").strip().lower()

    if target_type in ("pypdf", "pdf"):
        return PyPdfParserAdapter()
    return OpenDataLoaderParserAdapter()
