"""CLI 진입점 — ingest / ask / serve / setup 서브커맨드.

사용 예:
  python app/main.py ingest --reset
  python app/main.py ask --question "..."
  python app/main.py serve
"""

from __future__ import annotations

import argparse
import sys

if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from ingest import ingest_documents
from query import ask_question
from settings import WEB_HOST, WEB_PORT


def build_parser() -> argparse.ArgumentParser:
    """argparse 서브커맨드 트리를 구성합니다."""
    parser = argparse.ArgumentParser(
        description="Local RAG — Document Parsers + FAISS/SQLite + Embeddings + Rerankers + Serving Engines"
    )
    sub = parser.add_subparsers(dest="command", required=True)

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
    ingest_parser.add_argument(
        "--fresh",
        action="store_true",
        help="청크 캐시 및 체크포인트를 초기화하고 1단계(파싱)부터 강제 실행",
    )

    ask_parser = sub.add_parser("ask", help="질문하기")
    ask_parser.add_argument("--question", required=True, help="질문")

    serve_parser = sub.add_parser("serve", help="웹 UI")
    serve_parser.add_argument("--host", default=None)
    serve_parser.add_argument("--port", type=int, default=None)

    setup_parser = sub.add_parser("setup", help="모델 및 서빙 엔진 사전 준비")
    setup_parser.add_argument("--models", action="store_true", help="임베딩·리랭커 모델 다운로드")
    setup_parser.add_argument("--embed-only", action="store_true", help="임베딩 모델만 다운로드")
    setup_parser.add_argument("--engine", action="store_true", help="서빙 엔진 바이너리 및 모델 준비/기동")
    setup_parser.add_argument("--gpu", action="store_true", help="CUDA PyTorch 환경 확인 및 강제 설치")

    return parser


def main() -> None:
    """CLI 인자 파싱 후 해당 서브커맨드 핸들러 실행."""
    parser = build_parser()
    args = parser.parse_args()

    # ACCELERATOR 설정에 따른 GPU(CUDA) 가속 환경 점검 및 자동 설치
    from settings import ACCELERATOR
    if ACCELERATOR in ("auto", "cuda", "gpu"):
        from services import ensure_gpu_torch
        ensure_gpu_torch(force=bool(getattr(args, "gpu", False)))

    if args.command == "ingest":
        ingest_documents(reset=args.reset, parser_type=args.parser, fresh=args.fresh)
    elif args.command == "ask":
        answer = ask_question(args.question)
        print("\n=== Answer ===\n")
        print(answer)
    elif args.command == "serve":
        from web_server import run_server
        run_server(host=args.host, port=args.port)
    elif args.command == "setup":
        from services import get_engine_manager, get_model_manager
        from serving_engines import canonical_engine
        from settings import (
            EMBED_MODEL,
            EMBED_MODELS_DIR,
            MANAGED_SERVING_ENGINES,
            RERANK_MODEL,
            RERANK_MODELS_DIR,
            RERANK_STRATEGY,
            SERVING_ENGINE,
            hf_model_dir,
            rerank_label,
            serving_label,
        )

        model_mgr = get_model_manager()
        from embeddings import ensure_gguf_model, is_gguf_model

        if args.embed_only:
            if is_gguf_model(EMBED_MODEL):
                ensure_gguf_model()
                print(f"[setup] GGUF 임베딩 모델 준비 완료: {EMBED_MODEL}")
            else:
                dest = hf_model_dir(EMBED_MODEL, EMBED_MODELS_DIR)
                model_mgr.ensure_model(EMBED_MODEL, dest)
                print(f"[setup] 임베딩 모델 준비 완료: {EMBED_MODEL}")
            return

        if args.models or (not args.engine and not args.embed_only):
            if is_gguf_model(EMBED_MODEL):
                ensure_gguf_model()
            else:
                dest_embed = hf_model_dir(EMBED_MODEL, EMBED_MODELS_DIR)
                model_mgr.ensure_model(EMBED_MODEL, dest_embed)
            if RERANK_STRATEGY == "origin":
                dest_rerank = hf_model_dir(RERANK_MODEL, RERANK_MODELS_DIR)
                model_mgr.ensure_model(RERANK_MODEL, dest_rerank)
            if RERANK_STRATEGY == "origin":
                print(f"[setup] 임베딩·origin 리랭커 준비 완료 ({rerank_label()})")
            else:
                print(
                    f"[setup] 임베딩 준비 완료 ({EMBED_MODEL}). "
                    f"리랭크={rerank_label()} 는 질의 때 로드 (Qwen 리랭커는 받지 않음)"
                )

        if args.engine or (not args.models and not args.embed_only):
            engine = canonical_engine(SERVING_ENGINE)
            # EngineManager 는 llama.cpp 바이너리만 관리 — 전략과 수명주기 정합
            if engine not in MANAGED_SERVING_ENGINES:
                print(
                    f"[setup] {serving_label()}. "
                    f"이 프로젝트는 llama.cpp만 data/serving_engines 에 받아 띄웁니다."
                )
                return
            engine_mgr = get_engine_manager()
            ok = engine_mgr.ensure_running()
            print(f"[setup] 서빙 엔진 준비 {'완료' if ok else '실패'} ({serving_label()})")


if __name__ == "__main__":
    main()
