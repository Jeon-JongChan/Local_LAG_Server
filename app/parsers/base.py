"""문서 파서 추상 베이스 어댑터 모듈.

다양한 PDF/문서 파서(OpenDataLoader, PyMuPDF, PDFPlumber, Unstructured 등)를
단일한 규약(ParsedElement, ParseBatchResult)으로 추상화합니다.
"""

from __future__ import annotations

import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from settings import DOCS_DIR, PARSED_DIR, REJECTED_DIR

ProgressCallback = Callable[[int, int, str], None]
_SKIP_DOC_DIR_NAMES = frozenset({"_rejected", "rejected"})


@dataclass
class ParseBatchResult:
    """배치 파싱 결과 컨테이너."""
    elements: List[Dict[str, Any]] = field(default_factory=list)
    quarantined: List[Tuple[str, str]] = field(default_factory=list)


def discover_pdf_files(docs_dir: Path = DOCS_DIR) -> List[Path]:
    """
    docs_dir 이하 모든 .pdf/.PDF 파일을 재귀 탐색합니다.

    - 숨김 파일(.으로 시작) 제외
    - rejected 하위 폴더 제외
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


def clear_parsed_cache(parsed_dir: Path = PARSED_DIR) -> None:
    """data/parsed 임시 캐시 디렉터리를 비웁니다."""
    if parsed_dir.is_dir():
        shutil.rmtree(parsed_dir, ignore_errors=True)
    parsed_dir.mkdir(parents=True, exist_ok=True)


class BaseDocumentParserAdapter(ABC):
    """문서 파서 추상 베이스 어댑터."""

    name: str = "base"

    @abstractmethod
    def parse_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """단일 문서 파일을 파싱하여 텍스트 요소 목록(element)을 반환합니다."""
        pass

    def quarantine_file(self, bad_path: Path, reason: str) -> None:
        """손상되거나 파싱할 수 없는 문서를 격리 디렉토리로 이동."""
        REJECTED_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        dest = REJECTED_DIR / f"{bad_path.stem}_{ts}{bad_path.suffix}"
        try:
            shutil.move(str(bad_path), str(dest))
            print(f"[격리] {bad_path.name} → {dest.name} (사유: {reason})")
        except Exception as exc:
            print(f"[경고] 파일 격리 이동 실패 ({bad_path.name}): {exc}")

    def parse_batch(
        self,
        file_paths: List[Path],
        on_progress: Optional[ProgressCallback] = None,
    ) -> ParseBatchResult:
        """
        [템플릿 메서드] 여러 문서 파일을 배치로 파싱합니다.
        진행률 콜백 호출 및 파싱 실패 파일 자동 격리를 공통으로 수행합니다.
        """
        result = ParseBatchResult()
        total = len(file_paths)

        for idx, file_path in enumerate(file_paths, start=1):
            if on_progress:
                on_progress(idx - 1, total, file_path.name)

            try:
                elements = self.parse_file(file_path)
                result.elements.extend(elements)
            except Exception as exc:
                reason = f"파싱 실패: {exc}"
                result.quarantined.append((file_path.name, reason))
                self.quarantine_file(file_path, reason)

        if on_progress:
            on_progress(total, total, "완료")

        return result
