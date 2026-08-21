"""
임베딩·리랭커를 data/models/ 에 미리 받아 둡니다.

local-rag.bat 가 ingest / ask / serve 전에 호출합니다.
app/models.py 도 없으면 자동 다운로드합니다 (이중 안전장치).

환경 변수: EMBED_MODEL, RERANK_MODEL (기본 Qwen3-Embedding / Qwen3-Reranker)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))

from models import ensure_model_downloaded  # noqa: E402
from settings import (  # noqa: E402
    EMBED_MODEL,
    EMBED_MODELS_DIR,
    RERANK_MODEL,
    RERANK_MODELS_DIR,
    hf_model_dir,
)


def ensure_embed() -> Path:
    path = hf_model_dir(EMBED_MODEL, EMBED_MODELS_DIR)
    return ensure_model_downloaded(EMBED_MODEL, path)


def ensure_rerank() -> Path:
    path = hf_model_dir(RERANK_MODEL, RERANK_MODELS_DIR)
    return ensure_model_downloaded(RERANK_MODEL, path)


def main() -> int:
    parser = argparse.ArgumentParser(description="임베딩·리랭커 모델 로컬 다운로드")
    parser.add_argument(
        "--embed-only",
        action="store_true",
        help="임베딩 모델만 받기 (ingest 용)",
    )
    args = parser.parse_args()

    try:
        print(f"[ml] EMBED={EMBED_MODEL}")
        ensure_embed()
        if not args.embed_only:
            print(f"[ml] RERANK={RERANK_MODEL}")
            ensure_rerank()
    except Exception as e:
        print(f"[오류] ML 모델 준비 실패: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
