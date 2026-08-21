"""
CLI 진입점 — ingest / ask / serve 서브커맨드.

사용 예:
  python app/main.py ingest --reset
  python app/main.py ask --question "..."
  python app/main.py serve
"""

import argparse

from ingest import ingest_documents
from query import ask_question
from settings import WEB_HOST, WEB_PORT


def build_parser() -> argparse.ArgumentParser:
    """
    argparse 서브커맨드 트리를 구성합니다.

    command: ingest | ask | serve (필수)
    """
    parser = argparse.ArgumentParser(
        description="Local RAG — OpenDataLoader PDF + FAISS/SQLite + Qwen embed/rerank + llama.cpp"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- ingest: PDF → 벡터 DB ---
    ingest_parser = sub.add_parser("ingest", help="data/docs PDF 인제스트")
    ingest_parser.add_argument(
        "--reset",
        action="store_true",
        help="FAISS·SQLite 인덱스 전체 재구축",
    )

    # --- ask: 터미널에서 질문 1회 ---
    ask_parser = sub.add_parser("ask", help="질문하기")
    ask_parser.add_argument("--question", required=True, help="질문")

    # --- serve: 웹 UI HTTP 서버 ---
    serve_parser = sub.add_parser("serve", help="웹 UI")
    serve_parser.add_argument("--host", default=None)
    serve_parser.add_argument("--port", type=int, default=None)

    return parser


def main() -> None:
    """CLI 인자 파싱 후 해당 서브커맨드 핸들러 실행."""
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "ingest":
        ingest_documents(reset=args.reset)
    elif args.command == "ask":
        answer = ask_question(args.question)
        print("\n=== Answer ===\n")
        print(answer)
    elif args.command == "serve":
        # 웹 서버는 의존성 최소화를 위해 serve 시점에 import
        from web_server import run_server

        run_server(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
