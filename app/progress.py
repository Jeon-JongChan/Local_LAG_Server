"""CLI 진행률 표시 유틸 (ingest 등).

main ↔ ingest 순환 의존을 피하기 위해 분리한다.
"""

from __future__ import annotations

import sys
from typing import Optional


def _bar(pct: int, width: int = 24) -> str:
    """텍스트 기반 프로그레스 바 생성."""
    pct = max(0, min(100, pct))
    filled = int(width * pct / 100)
    return "[" + "#" * filled + "-" * (width - filled) + "]"


class IngestProgress:
    """4단계 인제스트 진행률: 문서 파싱 → 청킹 → 임베딩 → 저장."""

    def __init__(self, total_stages: int = 4) -> None:
        """단계 수 초기화."""
        self.total_stages = total_stages
        self._stage = 0
        self._title = ""
        self._current = 0
        self._total = 1
        self._last_len = 0

    def start_stage(self, stage: int, title: str, total: int) -> None:
        """새 단계 시작."""
        self._stage = stage
        self._title = title
        self._total = max(total, 1)
        self._current = 0
        print(f"\n[ingest] 단계 {stage}/{self.total_stages} - {title} (총 {total}개)")

    def update(self, current: int, total: Optional[int] = None, detail: str = "") -> None:
        """현재 단계 진행률 갱신."""
        if total is not None:
            self._total = max(total, 1)
        self._current = min(current, self._total)
        pct = int(100 * self._current / self._total)
        line = (
            f"[ingest] 단계 {self._stage}/{self.total_stages} - {self._title} "
            f"{_bar(pct)} {pct:3d}% ({self._current}/{self._total})"
        )
        if detail:
            short = detail if len(detail) <= 48 else detail[:45] + "..."
            line += f"  {short}"
        pad = max(0, self._last_len - len(line))
        sys.stdout.write("\r" + line + " " * pad)
        sys.stdout.flush()
        self._last_len = len(line)

    def finish_stage(self, summary: str) -> None:
        """단계 완료 한 줄 요약."""
        self.update(self._total, detail="완료")
        print()
        print(f"  → {summary}")
