"""
벡터 저장소 — FAISS(dense) + SQLite(청크·BM25 FTS5).

설계:
  - FAISS: L2 정규화 벡터 + IndexFlatIP (내적 = 코사인)
  - SQLite chunks: id = FAISS 인덱스와 1:1
  - chunks_fts: BM25용 FTS5 (한글·영문 토큰 전처리 후 저장)
"""

from __future__ import annotations

import json
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

import faiss
import numpy as np

from settings import (
    EMBED_MODEL,
    FAISS_PATH,
    INDEX_DIR,
    INDEX_META_PATH,
    RERANK_MODEL,
    SQLITE_PATH,
)

# 영숫자 토큰 + 한글 연속 구간 (형태소 분석기 없이 BM25용)
_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/\-]*|[가-힣]+")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def prepare_fts_text(text: str) -> str:
    """FTS5에 넣을 공백 구분 토큰 문자열."""
    return " ".join(_TOKEN_RE.findall((text or "").lower()))


def fts_match_query(question: str) -> str:
    """
    FTS5 MATCH 절용 질의.
    토큰을 OR 로 연결 (약어·키워드 부분 매칭에 유리).
    """
    tokens = _TOKEN_RE.findall((question or "").lower())
    if not tokens:
        return ""
    parts: List[str] = []
    for t in tokens:
        esc = t.replace('"', '""')
        parts.append(f'"{esc}"')
    return " OR ".join(parts)


class VectorStore:
    """FAISS + SQLite(+FTS5 BM25) 저장소."""

    def __init__(self) -> None:
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(SQLITE_PATH))
        self._conn.row_factory = sqlite3.Row
        self._ensure_schema()
        self._index: faiss.Index | None = None
        self._dim: int | None = None
        self._load_faiss_if_exists()

    def _ensure_schema(self) -> None:
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
            """
        )
        self._conn.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                text,
                tokenize = 'unicode61'
            )
            """
        )
        self._conn.commit()
        self._sync_fts_if_needed()

    def _fts_count(self) -> int:
        try:
            row = self._conn.execute(
                "SELECT COUNT(*) AS c FROM chunks_fts"
            ).fetchone()
            return int(row["c"]) if row else 0
        except sqlite3.Error:
            return 0

    def _sync_fts_if_needed(self) -> None:
        """chunks 는 있는데 FTS 가 비어 있으면 재구축 (마이그레이션)."""
        if self.count() > 0 and self._fts_count() == 0:
            self._rebuild_fts()

    def _rebuild_fts(self) -> None:
        self._conn.execute("DELETE FROM chunks_fts")
        rows = self._conn.execute("SELECT id, text FROM chunks").fetchall()
        for row in rows:
            prepared = prepare_fts_text(row["text"])
            if not prepared:
                continue
            self._conn.execute(
                "INSERT INTO chunks_fts(rowid, text) VALUES (?, ?)",
                (int(row["id"]), prepared),
            )
        self._conn.commit()

    def _load_faiss_if_exists(self) -> None:
        if not FAISS_PATH.is_file() or not INDEX_META_PATH.is_file():
            return
        meta = json.loads(INDEX_META_PATH.read_text(encoding="utf-8"))
        self._dim = int(meta["dimension"])
        self._index = faiss.read_index(str(FAISS_PATH))
        indexed_model = (meta.get("embed_model") or "").strip()
        if indexed_model and indexed_model != EMBED_MODEL:
            print(
                f"[store] 경고: 인덱스 embed_model={indexed_model} "
                f"≠ 현재 EMBED_MODEL={EMBED_MODEL}. "
                f"`ingest --reset` 으로 재색인하세요.",
                flush=True,
            )

    def reset(self) -> None:
        self._conn.executescript(
            "DELETE FROM chunks; DELETE FROM ingest_manifest;"
        )
        try:
            self._conn.execute("DELETE FROM chunks_fts")
        except sqlite3.Error:
            pass
        self._conn.commit()
        self._index = None
        self._dim = None
        for path in (FAISS_PATH, INDEX_META_PATH):
            if path.is_file():
                path.unlink()

    def count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) AS c FROM chunks").fetchone()
        return int(row["c"]) if row else 0

    @contextmanager
    def _transaction(self):
        try:
            yield
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    def add_chunks(
        self, chunks: List[Dict[str, Any]], embeddings: np.ndarray
    ) -> None:
        if len(chunks) != embeddings.shape[0]:
            raise ValueError("chunks 수와 embedding 행 수가 일치해야 합니다.")
        if embeddings.ndim != 2:
            raise ValueError("embeddings 는 2차원 배열이어야 합니다.")

        dim = embeddings.shape[1]
        vectors = np.ascontiguousarray(embeddings.astype(np.float32))

        with self._transaction():
            start_id = self.count()

            for i, chunk in enumerate(chunks):
                chunk_id = start_id + i
                bbox = chunk.get("bbox")
                text = chunk["text"]
                self._conn.execute(
                    """
                    INSERT INTO chunks (id, file_name, source_path, page, bbox, element_type, text, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chunk_id,
                        chunk.get("file_name", "unknown"),
                        chunk.get("source_path", ""),
                        chunk.get("page"),
                        json.dumps(bbox) if bbox is not None else None,
                        chunk.get("element_type"),
                        text,
                        _utc_now(),
                    ),
                )
                prepared = prepare_fts_text(text)
                if prepared:
                    self._conn.execute(
                        "INSERT INTO chunks_fts(rowid, text) VALUES (?, ?)",
                        (chunk_id, prepared),
                    )

            if self._index is None:
                self._index = faiss.IndexFlatIP(dim)
                self._dim = dim
            elif self._dim != dim:
                raise ValueError(
                    f"임베딩 차원 불일치: index={self._dim}, new={dim}"
                )

            self._index.add(vectors)

            faiss.write_index(self._index, str(FAISS_PATH))
            INDEX_META_PATH.write_text(
                json.dumps(
                    {
                        "dimension": dim,
                        "embed_model": EMBED_MODEL,
                        "rerank_model": RERANK_MODEL,
                        "count": self.count(),
                        "updated_at": _utc_now(),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

    def _row_to_hit(self, row: sqlite3.Row, score: float) -> Dict[str, Any]:
        bbox_raw = row["bbox"]
        bbox = json.loads(bbox_raw) if bbox_raw else None
        return {
            "id": int(row["id"]),
            "score": float(score),
            "text": row["text"],
            "file_name": row["file_name"],
            "source_path": row["source_path"],
            "page": row["page"],
            "bbox": bbox,
            "element_type": row["element_type"],
        }

    def search(self, query_vector: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        """Dense(FAISS) 검색."""
        if self._index is None or self.count() == 0:
            return []
        if self._dim is None:
            raise RuntimeError("FAISS 인덱스 차원이 설정되지 않았습니다.")

        q = np.ascontiguousarray(
            query_vector.astype(np.float32).reshape(1, -1)
        )
        k = min(top_k, self.count())
        scores, ids = self._index.search(q, k)

        results: List[Dict[str, Any]] = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0:
                continue
            row = self._conn.execute(
                "SELECT * FROM chunks WHERE id = ?", (int(idx),)
            ).fetchone()
            if row is None:
                continue
            results.append(self._row_to_hit(row, float(score)))
        return results

    def search_bm25(self, question: str, top_k: int) -> List[Dict[str, Any]]:
        """
        SQLite FTS5 BM25 검색.
        bm25() 는 더 관련 있을수록 더 작은(음수) 값 → ASC 정렬.
        """
        if self.count() == 0 or top_k <= 0:
            return []
        match = fts_match_query(question)
        if not match:
            return []

        try:
            rows = self._conn.execute(
                """
                SELECT c.*, bm25(chunks_fts) AS bm25_score
                FROM chunks_fts
                JOIN chunks c ON c.id = chunks_fts.rowid
                WHERE chunks_fts MATCH ?
                ORDER BY bm25_score
                LIMIT ?
                """,
                (match, top_k),
            ).fetchall()
        except sqlite3.OperationalError:
            # 잘못된 MATCH 문법 등
            return []

        results: List[Dict[str, Any]] = []
        for row in rows:
            hit = self._row_to_hit(row, float(row["bm25_score"]))
            hit["bm25_score"] = float(row["bm25_score"])
            results.append(hit)
        return results

    def get_manifest(self) -> Dict[str, Tuple[int, float]]:
        rows = self._conn.execute(
            "SELECT rel_path, size, mtime FROM ingest_manifest"
        ).fetchall()
        return {r["rel_path"]: (int(r["size"]), float(r["mtime"])) for r in rows}

    def update_manifest(self, rel_path: str, size: int, mtime: float) -> None:
        self._conn.execute(
            """
            INSERT INTO ingest_manifest (rel_path, size, mtime)
            VALUES (?, ?, ?)
            ON CONFLICT(rel_path) DO UPDATE SET size=excluded.size, mtime=excluded.mtime
            """,
            (rel_path, size, mtime),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()


def rrf_fuse(
    *hit_lists: List[Dict[str, Any]],
    k: int = 60,
    top_n: int = 20,
) -> List[Dict[str, Any]]:
    """
    Reciprocal Rank Fusion.
    score = Σ 1/(k + rank) ; rank 는 0-based.
    """
    scores: Dict[int, float] = {}
    docs: Dict[int, Dict[str, Any]] = {}

    for hits in hit_lists:
        for rank, hit in enumerate(hits):
            cid = int(hit["id"])
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
            if cid not in docs:
                docs[cid] = dict(hit)
            else:
                # dense score / bm25_score 등 메타 보존
                for key, val in hit.items():
                    if key == "score":
                        continue
                    if key not in docs[cid] or docs[cid].get(key) is None:
                        docs[cid][key] = val

    ranked_ids = sorted(scores.keys(), key=lambda i: scores[i], reverse=True)
    out: List[Dict[str, Any]] = []
    for cid in ranked_ids[:top_n]:
        item = dict(docs[cid])
        item["score"] = scores[cid]
        item["rrf_score"] = scores[cid]
        out.append(item)
    return out


def get_store() -> VectorStore:
    return VectorStore()
