"""
SQLite 기반 청크 본문, 메타데이터 및 FTS5 BM25 전문 검색 저장소 어댑터 구현체.
"""

from __future__ import annotations

import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import BaseChunkStore

try:
    from settings import INDEX_DIR, SQLITE_PATH
except ModuleNotFoundError:
    from app.settings import INDEX_DIR, SQLITE_PATH

_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/\-]*|[가-힣]+")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _prepare_fts_text(text: str) -> str:
    """FTS5 색인용 공백 구분 토큰 문자열."""
    return " ".join(_TOKEN_RE.findall((text or "").lower()))


def _fts_match_query(question: str) -> str:
    """FTS5 MATCH 절용 OR 결합 질의어 생성."""
    tokens = _TOKEN_RE.findall((question or "").lower())
    if not tokens:
        return ""
    parts: List[str] = []
    for t in tokens:
        esc = t.replace('"', '""')
        parts.append(f'"{esc}"')
    return " OR ".join(parts)


class SqliteChunkStore(BaseChunkStore):
    """SQLite 청크 본문 & FTS5 BM25 키워드 검색 어댑터."""

    def __init__(self, db_path: Path = SQLITE_PATH) -> None:
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._init_schema()

    def _init_schema(self) -> None:
        with self._conn:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY,
                    file_name TEXT NOT NULL,
                    source_path TEXT NOT NULL,
                    page INTEGER,
                    bbox TEXT,
                    element_type TEXT,
                    text TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS ingest_manifest (
                    rel_path TEXT PRIMARY KEY,
                    size INTEGER NOT NULL,
                    mtime REAL NOT NULL
                );
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                    text,
                    tokenize = 'unicode61'
                );
                """
            )

    def insert_chunk(
        self,
        chunk_id: int,
        file_name: str,
        source_path: str,
        page: Optional[int],
        bbox: Optional[str],
        element_type: Optional[str],
        text: str,
    ) -> None:
        now = _utc_now()
        fts_body = _prepare_fts_text(text)
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO chunks (id, file_name, source_path, page, bbox, element_type, text, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (chunk_id, file_name, source_path, page, bbox, element_type, text, now),
            )
            self._conn.execute(
                "INSERT INTO chunks_fts (rowid, text) VALUES (?, ?)",
                (chunk_id, fts_body),
            )

    def get_chunks(self, ids: List[int]) -> Dict[int, Dict[str, Any]]:
        if not ids:
            return {}
        placeholders = ",".join("?" for _ in ids)
        cur = self._conn.execute(
            f"SELECT * FROM chunks WHERE id IN ({placeholders})",
            ids,
        )
        return {int(row["id"]): dict(row) for row in cur.fetchall()}

    def search_bm25(self, query: str, top_k: int = 16) -> List[Dict[str, Any]]:
        fts_q = _fts_match_query(query)
        if not fts_q:
            return []

        sql = """
            SELECT c.*, bm25(chunks_fts) AS bm25_score
            FROM chunks_fts f
            JOIN chunks c ON c.id = f.rowid
            WHERE chunks_fts MATCH ?
            ORDER BY bm25_score ASC
            LIMIT ?
        """
        cur = self._conn.execute(sql, (fts_q, top_k))
        out: List[Dict[str, Any]] = []
        for r in cur.fetchall():
            d = dict(r)
            d["score"] = -float(d.pop("bm25_score", 0.0))
            out.append(d)
        return out

    def is_ingested(self, rel_path: str, size: int, mtime: float) -> bool:
        cur = self._conn.execute(
            "SELECT size, mtime FROM ingest_manifest WHERE rel_path = ?",
            (rel_path,),
        )
        row = cur.fetchone()
        if not row:
            return False
        return int(row["size"]) == int(size) and abs(float(row["mtime"]) - float(mtime)) < 1e-4

    def update_manifest(self, rel_path: str, size: int, mtime: float) -> None:
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO ingest_manifest (rel_path, size, mtime)
                VALUES (?, ?, ?)
                ON CONFLICT(rel_path) DO UPDATE SET size = excluded.size, mtime = excluded.mtime
                """,
                (rel_path, size, mtime),
            )

    def count(self) -> int:
        cur = self._conn.execute("SELECT COUNT(*) FROM chunks")
        return int(cur.fetchone()[0])

    def reset(self) -> None:
        with self._conn:
            self._conn.execute("DELETE FROM chunks")
            self._conn.execute("DELETE FROM ingest_manifest")
            self._conn.execute("DELETE FROM chunks_fts")

    def close(self) -> None:
        self._conn.close()
