"""
SQLite 기반 대화 세션 저장소 어댑터 구현체.
기존 chat_db.py의 세션/메시지 관리 규약과 100% 호환됩니다.
"""

from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import BaseChatStore

try:
    from settings import INDEX_DIR, SQLITE_PATH
except ModuleNotFoundError:
    from app.settings import INDEX_DIR, SQLITE_PATH


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _title_from_text(text: str, max_len: int = 40) -> str:
    one_line = " ".join(text.strip().split())
    if len(one_line) <= max_len:
        return one_line or "새 대화"
    return one_line[: max_len - 1] + "…"


class SqliteChatStore(BaseChatStore):
    """SQLite 대화 세션 및 메시지 저장소 어댑터."""

    def __init__(self, db_path: Path | str = SQLITE_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON;")
        self._ensure_schema()

    def __enter__(self) -> SqliteChatStore:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def _ensure_schema(self) -> None:
        with self._conn:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES chat_sessions(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_chat_messages_session
                    ON chat_messages(session_id, id);
                """
            )

    def list_sessions(self, limit: int = 50) -> List[Dict[str, Any]]:
        rows = self._conn.execute(
            """
            SELECT id, title, created_at, updated_at
            FROM chat_sessions
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def create_session(self, title: str = "새 대화", model: str = "") -> Dict[str, Any]:
        sid = str(uuid.uuid4())
        now = _utc_now()
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO chat_sessions (id, title, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                (sid, title.strip() or "새 대화", now, now),
            )
        return {
            "id": sid,
            "title": title.strip() or "새 대화",
            "created_at": now,
            "updated_at": now,
        }

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        row = self._conn.execute(
            "SELECT id, title, created_at, updated_at FROM chat_sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
        return dict(row) if row else None

    def list_messages(self, session_id: str) -> List[Dict[str, Any]]:
        rows = self._conn.execute(
            """
            SELECT id, session_id, role, content, created_at
            FROM chat_messages
            WHERE session_id = ?
            ORDER BY id ASC
            """,
            (session_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def delete_session(self, session_id: str) -> bool:
        cur = self._conn.execute("SELECT id FROM chat_sessions WHERE id = ?", (session_id,))
        if not cur.fetchone():
            return False
        with self._conn:
            self._conn.execute("DELETE FROM chat_messages WHERE session_id = ?", (session_id,))
            self._conn.execute("DELETE FROM chat_sessions WHERE id = ?", (session_id,))
        return True

    def add_message(self, session_id: str, role: str, content: str) -> Dict[str, Any]:
        if role not in ("user", "assistant"):
            raise ValueError("role must be user or assistant")
        now = _utc_now()
        with self._conn:
            cur = self._conn.execute(
                """
                INSERT INTO chat_messages (session_id, role, content, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (session_id, role, content, now),
            )
            if role == "user":
                count = self._conn.execute(
                    "SELECT COUNT(*) AS c FROM chat_messages WHERE session_id = ? AND role = 'user'",
                    (session_id,),
                ).fetchone()
                if count and int(count["c"]) == 1:
                    self._conn.execute(
                        "UPDATE chat_sessions SET title = ? WHERE id = ?",
                        (_title_from_text(content), session_id),
                    )
            self._conn.execute(
                "UPDATE chat_sessions SET updated_at = ? WHERE id = ?",
                (now, session_id),
            )
        return {
            "id": int(cur.lastrowid),
            "session_id": session_id,
            "role": role,
            "content": content,
            "created_at": now,
        }

    def close(self) -> None:
        self._conn.close()
