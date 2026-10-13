"""CLI 진입점 — ingest / ask / serve 서브커맨드.

사용 예:
  python app/main.py ingest --reset
  python app/main.py ask --question "..."
  python app/main.py serve
"""

from __future__ import annotations

import argparse
import sys
from typing import Callable, Optional

from ingest import ingest_documents
from query import ask_question
from settings import WEB_HOST, WEB_PORT

ProgressCallback = Callable[[int, int, str], None]


def _bar(pct: int, width: int = 24) -> str:
    """텍스트 기반 프로그레스 바 생성."""
    pct = max(0, min(100, pct))
    filled = int(width * pct / 100)
    return "[" + "#" * filled + "-" * (width - filled) + "]"


class IngestProgress:
    """4단계 인제스트 진행률: 문서 파싱 → 청킹 → 임베딩 → 저장."""

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
        print(f"\n[ingest] 단계 {stage}/{self.total_stages} — {title} (총 {total}개)")

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


def build_parser() -> argparse.ArgumentParser:
    """argparse 서브커맨드 트리를 구성합니다."""
    parser = argparse.ArgumentParser(
        description="Local RAG — Document Parsers + FAISS/SQLite + Embeddings + Rerankers + Serving Engines"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- ingest: PDF/문서 → 벡터 DB ---
    ingest_parser = sub.add_parser("ingest", help="data/docs 문서 인제스트")
    ingest_parser.add_argument(
        "--reset",
        action="store_true",
        help="FAISS·SQLite 인덱스 전체 재구축",
    )
    ingest_parser.add_argument(
        "--parser",
        default=None,
        help="문서 파서 선택 (opendataloader | pypdf)",
    )

    # --- ask: 터미널에서 질문 1회 ---
    ask_parser = sub.add_parser("ask", help="질문하기")
    ask_parser.add_argument("--question", required=True, help="질문")

    # --- serve: 웹 UI HTTP 서버 ---
    serve_parser = sub.add_parser("serve", help="웹 UI")
    serve_parser.add_argument("--host", default=None)
    serve_parser.add_argument("--port", type=int, default=None)

    # --- setup: 모델 및 서빙 엔진 사전 준비 ---
    setup_parser = sub.add_parser("setup", help="모델 및 서빙 엔진 사전 준비")
    setup_parser.add_argument("--models", action="store_true", help="임베딩·리랭커 모델 다운로드")
    setup_parser.add_argument("--embed-only", action="store_true", help="임베딩 모델만 다운로드")
    setup_parser.add_argument("--engine", action="store_true", help="서빙 엔진 바이너리 및 모델 준비/기동")

    return parser


def main() -> None:
    """CLI 인자 파싱 후 해당 서브커맨드 핸들러 실행."""
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "ingest":
        ingest_documents(reset=args.reset, parser_type=args.parser)
    elif args.command == "ask":
        answer = ask_question(args.question)
        print("\n=== Answer ===\n")
        print(answer)
    elif args.command == "serve":
        from web_server import run_server
        run_server(host=args.host, port=args.port)
    elif args.command == "setup":
        from services import get_engine_manager, get_model_manager
        from settings import (
            EMBED_MODEL,
            EMBED_MODELS_DIR,
            RERANK_MODEL,
            RERANK_MODELS_DIR,
            RERANK_STRATEGY,
            hf_model_dir,
        )

        model_mgr = get_model_manager("huggingface")
        if args.embed_only:
            dest = hf_model_dir(EMBED_MODEL, EMBED_MODELS_DIR)
            model_mgr.ensure_model(EMBED_MODEL, dest)
            print(f"[setup] 임베딩 모델 준비 완료: {EMBED_MODEL}")
            return

        if args.models or (not args.engine and not args.embed_only):
            dest_embed = hf_model_dir(EMBED_MODEL, EMBED_MODELS_DIR)
            model_mgr.ensure_model(EMBED_MODEL, dest_embed)
            if RERANK_STRATEGY == "origin":
                dest_rerank = hf_model_dir(RERANK_MODEL, RERANK_MODELS_DIR)
                model_mgr.ensure_model(RERANK_MODEL, dest_rerank)
            print(f"[setup] ML 모델 준비 완료 (EMBED={EMBED_MODEL}, RERANK={RERANK_STRATEGY})")

        if args.engine or (not args.models and not args.embed_only):
            engine_mgr = get_engine_manager("llama_cpp")
            engine_mgr.start()
            print(f"[setup] 서빙 엔진 준비 완료")


if __name__ == "__main__":
    main()
