"""
인제스트 진행률 출력 (Windows cmd / PowerShell 터미널용).
"""

from __future__ import annotations

import sys
from typing import Callable, Optional

ProgressCallback = Callable[[int, int, str], None]


def _bar(pct: int, width: int = 24) -> str:
    pct = max(0, min(100, pct))
    filled = int(width * pct / 100)
    return "[" + "#" * filled + "-" * (width - filled) + "]"


class IngestProgress:
    """4단계 인제스트 진행률: PDF 파싱 → 청킹 → 임베딩 → 저장."""

    def __init__(self, total_stages: int = 4) -> None:
        self.total_stages = total_stages
        self._stage = 0
        self._title = ""
        self._current = 0
        self._total = 1
        self._last_len = 0

    def start_stage(self, stage: int, title: str, total: int) -> None:
        self._stage = stage
        self._title = title
        self._total = max(total, 1)
        self._current = 0
        print(
            f"\n[ingest] 단계 {stage}/{self.total_stages} — {title} "
            f"(총 {total}개)"
        )

    def update(self, current: int, total: Optional[int] = None, detail: str = "") -> None:
        if total is not None:
            self._total = max(total, 1)
        self._current = min(current, self._total)
        pct = int(100 * self._current / self._total)
        line = (
            f"[ingest] 단계 {self._stage}/{self.total_stages} — {self._title} "
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
        self.update(self._total, detail="완료")
        print()
        print(f"  → {summary}")
