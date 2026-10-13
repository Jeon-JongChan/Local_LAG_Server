"""문서 파서 어댑터 팩토리 모듈."""

from __future__ import annotations

import os
from typing import Optional
from .base import BaseDocumentParserAdapter
from .opendataloader import OpenDataLoaderParserAdapter
from .pypdf_parser import PyPdfParserAdapter

_active_parser: Optional[BaseDocumentParserAdapter] = None


def get_document_parser(parser_type: Optional[str] = None) -> BaseDocumentParserAdapter:
    """
    문서 파서 어댑터 팩토리 함수.

    Args:
        parser_type: 'opendataloader' | 'pypdf' (기본: 환경변수 DOC_PARSER 또는 'opendataloader')

    Returns:
        BaseDocumentParserAdapter 서브클래스 인스턴스
    """
    global _active_parser

    target_type = (parser_type or os.getenv("DOC_PARSER", "opendataloader")).strip().lower()

    if target_type in ("opendataloader", "open_data_loader"):
        return OpenDataLoaderParserAdapter()
    elif target_type in ("pypdf", "pdf"):
        return PyPdfParserAdapter()
    else:
        # 기본 opendataloader로 fallback
        return OpenDataLoaderParserAdapter()

