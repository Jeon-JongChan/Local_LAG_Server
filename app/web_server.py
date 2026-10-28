"""
웹 UI HTTP 서버 — 표준 라이브러리만 사용 (FastAPI/Flask 없음).

제공:
  - GET  /, /static/*           → Vue 채팅 UI
  - GET  /admin                 → 관리자(모델·질의 구간 시간)
  - GET  /api/admin             → 설정 스냅샷 + 최근 질의 트레이스
  - GET  /api/sessions          → 대화 목록
  - POST /api/sessions          → 새 대화
  - GET  /api/sessions/{id}     → 대화 + 메시지
  - DELETE /api/sessions/{id}   → 대화 삭제
  - POST /api/sessions/{id}/messages → 질문 전송 (RAG)
  - GET  /api/query/live        → 진행 중 단계 로그
"""

from __future__ import annotations

import json
import re
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import unquote

from query import (
    ask_question,
    ask_question_detailed,
    get_live_query,
    get_runtime_config,
    list_query_traces,
)
from settings import WEB_HOST, WEB_PORT, PARSED_DIR
from storage import get_chat_store

STATIC_ROOT = Path(__file__).resolve().parent / "static"

MIME = {
    ".html": "text/html; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
}

_SESSION_PATH_RE = re.compile(
    r"^/api/sessions/([^/]+)(/messages)?$"
)

# 백그라운드 지속 생성 관리자 (클라이언트 접속이 끊겨도 답변 생성을 끝까지 완수하여 DB에 저장)
_session_tasks: Dict[str, threading.Event] = {}
_session_task_results: Dict[str, Dict[str, Any]] = {}
_tasks_lock = threading.Lock()


def _parse_json_body(handler: BaseHTTPRequestHandler) -> tuple[Optional[dict], Optional[str]]:
    length = int(handler.headers.get("Content-Length", 0))
    raw = handler.rfile.read(length) if length else b"{}"
    try:
        return json.loads(raw.decode("utf-8")), None
    except json.JSONDecodeError:
        return None, "invalid json"


class Handler(BaseHTTPRequestHandler):
    """HTTP 요청 1건을 처리하는 핸들러 (스레드마다 인스턴스 생성)."""

    def log_message(self, fmt: str, *args) -> None:
        """액세스 로그. 진행 표시 폴링은 질문 1건에 수십 번이라 생략."""
        line = fmt % args
        if "/api/query/live" in line:
            return
        print(f"[web] {self.address_string()} {line}")

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, OSError):
            # 클라이언트가 브라우저 탭을 닫거나 이동하여 소켓이 끊긴 경우 무시
            pass

    def _send_file(self, path: Path) -> None:
        data = path.read_bytes()
        ext = path.suffix.lower()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", MIME.get(ext, "application/octet-stream"))
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_json(self) -> tuple[Optional[dict], Optional[str]]:
        return _parse_json_body(self)

    def do_GET(self) -> None:
        path = unquote(self.path.split("?", 1)[0])

        if path in ("/", "/index.html"):
            return self._send_file(STATIC_ROOT / "index.html")

        if path in ("/admin", "/admin.html"):
            return self._send_file(STATIC_ROOT / "admin.html")

        if path == "/api/admin":
            return self._api_admin()

        if path == "/api/query/live":
            return self._send_json(HTTPStatus.OK, get_live_query())

        if path == "/api/sessions":
            return self._api_list_sessions()

        m = _SESSION_PATH_RE.match(path)
        if m and not m.group(2):
            return self._api_get_session(m.group(1))

        if path.startswith("/api/images/"):
            rel = path[len("/api/images/") :].lstrip("/")
            if ".." in rel:
                self.send_error(HTTPStatus.FORBIDDEN)
                return
            full = (PARSED_DIR / rel).resolve()
            try:
                full.relative_to(PARSED_DIR.resolve())
            except ValueError:
                self.send_error(HTTPStatus.FORBIDDEN)
                return
            if full.is_file():
                return self._send_file(full)
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        if path.startswith("/static/"):
            rel = path[len("/static/") :].lstrip("/")
            if ".." in rel:
                self.send_error(HTTPStatus.FORBIDDEN)
                return
            full = (STATIC_ROOT / rel).resolve()
            try:
                full.relative_to(STATIC_ROOT.resolve())
            except ValueError:
                self.send_error(HTTPStatus.FORBIDDEN)
                return
            if full.is_file():
                return self._send_file(full)

        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        path = unquote(self.path.split("?", 1)[0])

        if path == "/api/sessions":
            return self._api_create_session()

        m = _SESSION_PATH_RE.match(path)
        if m and m.group(2):
            return self._api_post_message(m.group(1))

        # 하위 호환
        if path == "/api/ask":
            return self._api_ask_legacy()

        self.send_error(HTTPStatus.NOT_FOUND)

    def do_DELETE(self) -> None:
        path = unquote(self.path.split("?", 1)[0])
        m = _SESSION_PATH_RE.match(path)
        if m and not m.group(2):
            return self._api_delete_session(m.group(1))
        self.send_error(HTTPStatus.NOT_FOUND)

    def _api_admin(self) -> None:
        """현재 모델/엔진 설정 + 최근 질의 구간 시간 피드."""
        self._send_json(
            HTTPStatus.OK,
            {
                "config": get_runtime_config(),
                "traces": list_query_traces(60),
            },
        )

    def _api_list_sessions(self) -> None:
        db = get_chat_store()
        try:
            self._send_json(HTTPStatus.OK, {"sessions": db.list_sessions()})
        finally:
            db.close()

    def _api_create_session(self) -> None:
        data, err = self._read_json()
        if err:
            return self._send_json(HTTPStatus.BAD_REQUEST, {"error": err})
        title = ((data or {}).get("title") or "새 대화").strip() or "새 대화"
        db = get_chat_store()
        try:
            session = db.create_session(title=title)
            self._send_json(HTTPStatus.CREATED, {"session": session})
        finally:
            db.close()

    def _api_get_session(self, session_id: str) -> None:
        db = get_chat_store()
        try:
            session = db.get_session(session_id)
            if not session:
                return self._send_json(HTTPStatus.NOT_FOUND, {"error": "session not found"})
            messages = db.list_messages(session_id)
            with _tasks_lock:
                generating = session_id in _session_tasks
            self._send_json(
                HTTPStatus.OK,
                {"session": session, "messages": messages, "generating": generating},
            )
        finally:
            db.close()

    def _api_delete_session(self, session_id: str) -> None:
        db = get_chat_store()
        try:
            if not db.delete_session(session_id):
                return self._send_json(HTTPStatus.NOT_FOUND, {"error": "session not found"})
            self._send_json(HTTPStatus.OK, {"ok": True})
        finally:
            db.close()

    def _api_post_message(self, session_id: str) -> None:
        data, err = self._read_json()
        if err:
            return self._send_json(HTTPStatus.BAD_REQUEST, {"error": err})

        content = ((data or {}).get("content") or "").strip()
        if not content:
            return self._send_json(HTTPStatus.BAD_REQUEST, {"error": "content required"})

        think_mode = ((data or {}).get("think_mode") or "instant").strip().lower()
        if think_mode not in ("instant", "think", "auto"):
            think_mode = "instant"

        # 기본 OFF — '추론'을 명시적으로 켤 때만 LLM 호출, 아니면 청크 원문만 표시
        reason = bool((data or {}).get("reason"))

        # 기본 OFF — UI/요청에서 명시적으로 켤 때만 이전 대화 반영
        use_history = bool((data or {}).get("use_history"))

        db = get_chat_store()
        try:
            if not db.get_session(session_id):
                return self._send_json(HTTPStatus.NOT_FOUND, {"error": "session not found"})

            prior = db.list_messages(session_id) if use_history else None
            user_msg = db.add_message(session_id, "user", content)
        finally:
            db.close()

        # 백그라운드 지속 생성 워커 시작
        event = threading.Event()
        with _tasks_lock:
            _session_tasks[session_id] = event

        def _bg_worker():
            try:
                worker_db = get_chat_store()
                try:
                    answer, trace = ask_question_detailed(
                        content,
                        think_mode=think_mode,
                        history=prior,
                        reason=reason,
                    )
                    images = trace.get("images", [])
                    assistant_msg = worker_db.add_message(session_id, "assistant", answer, images=images)
                    session = worker_db.get_session(session_id)
                    with _tasks_lock:
                        _session_task_results[session_id] = {
                            "session": session,
                            "user_message": user_msg,
                            "assistant_message": assistant_msg,
                            "trace": trace,
                            "error": None,
                        }
                finally:
                    worker_db.close()
            except Exception as exc:
                with _tasks_lock:
                    _session_task_results[session_id] = {
                        "error": str(exc),
                        "user_message": user_msg,
                    }
            finally:
                with _tasks_lock:
                    if _session_tasks.get(session_id) == event:
                        del _session_tasks[session_id]
                event.set()

        worker_thread = threading.Thread(target=_bg_worker, daemon=True)
        worker_thread.start()

        # 클라이언트 연결이 유지되는 동안 생성 완료 대기
        from settings import REQUEST_TIMEOUT
        completed = event.wait(timeout=REQUEST_TIMEOUT)

        with _tasks_lock:
            res = _session_task_results.pop(session_id, None)

        try:
            if not completed or not res:
                return self._send_json(HTTPStatus.ACCEPTED, {"generating": True, "user_message": user_msg})

            if res.get("error"):
                return self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": res["error"]})

            self._send_json(HTTPStatus.OK, res)
        except (BrokenPipeError, ConnectionResetError, OSError):
            # 클라이언트 연결이 끊겼더라도 백그라운드에서 DB 저장은 이미 완료되었으므로 안전하게 종료
            pass

    def _api_ask_legacy(self) -> None:
        data, err = self._read_json()
        if err:
            return self._send_json(HTTPStatus.BAD_REQUEST, {"error": err})
        question = ((data or {}).get("question") or "").strip()
        if not question:
            return self._send_json(HTTPStatus.BAD_REQUEST, {"error": "question required"})
        answer, trace = ask_question_detailed(question)
        self._send_json(HTTPStatus.OK, {"answer": answer, "trace": trace})


def run_server(host: str | None = None, port: int | None = None) -> None:
    """웹 서버 기동. 배너 값은 SettingsFacade 경유 상수를 쓴다."""
    from settings import EMBED_MODEL, LLAMA_GGUF_FILE, rerank_label, serving_label

    h = host or WEB_HOST
    p = port or WEB_PORT
    chat = LLAMA_GGUF_FILE or "(LLAMA_GGUF_FILE 미설정)"
    print(f"[web] http://{h}:{p}")
    print(f"[web] CHAT={chat}")
    print(f"[web] EMBED={EMBED_MODEL}")
    print(f"[web] RERANK={rerank_label()}")
    print(f"[web] SERVING={serving_label()}")
    server = ThreadingHTTPServer((h, p), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[web] 종료")
    finally:
        server.server_close()


if __name__ == "__main__":
    run_server()
