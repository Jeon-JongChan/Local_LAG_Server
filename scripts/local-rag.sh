#!/usr/bin/env bash
# local-rag Linux 런처 (Windows: scripts/local-rag.bat)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# ============================================================
#  사용자 설정
# ============================================================
export EMBED_MODEL="${EMBED_MODEL:-Qwen/Qwen3-Embedding-0.6B}"
export RERANK_MODEL="${RERANK_MODEL:-Qwen/Qwen3-Reranker-0.6B}"
export FAISS_TOP_K="${FAISS_TOP_K:-8}"
export RERANK_TOP_K="${RERANK_TOP_K:-4}"
export HYBRID_BM25="${HYBRID_BM25:-1}"
export BM25_TOP_K="${BM25_TOP_K:-20}"
# 리랭크 끄기: RERANK_TOP_K=0

export CHUNK_MIN_CHARS="${CHUNK_MIN_CHARS:-200}"
export CHUNK_MAX_CHARS="${CHUNK_MAX_CHARS:-800}"
export CHAT_TEMPERATURE="${CHAT_TEMPERATURE:-0.2}"

export LLAMA_CPP_BASE_URL="${LLAMA_CPP_BASE_URL:-http://localhost:8080/v1}"
export LLAMA_NUM_PREDICT="${LLAMA_NUM_PREDICT:-4096}"
export LLAMA_NUM_PREDICT_THINK="${LLAMA_NUM_PREDICT_THINK:-16384}"
export THINK_AUTO_MIN_CHARS="${THINK_AUTO_MIN_CHARS:-200}"
export WEB_PORT="${WEB_PORT:-8787}"

# llama.cpp: cpu | vulkan | cuda (비우면 자동 — Linux는 NVIDIA면 vulkan, 아니면 cpu)
export LLAMA_CPP_VARIANT="${LLAMA_CPP_VARIANT:-}"
# export LLAMA_CPP_RELEASE_TAG=b9565

export LLAMA_CTX_SIZE="${LLAMA_CTX_SIZE:-4096}"
export LLAMA_NGL="${LLAMA_NGL:-999}"
export LLAMA_THREADS="${LLAMA_THREADS:-8}"
export LLAMA_THREADS_BATCH="${LLAMA_THREADS_BATCH:-16}"
export LLAMA_FLASH_ATTN="${LLAMA_FLASH_ATTN:-on}"
export LLAMA_BATCH_SIZE="${LLAMA_BATCH_SIZE:-1024}"
export LLAMA_UBATCH_SIZE="${LLAMA_UBATCH_SIZE:-256}"
export LLAMA_REASONING="${LLAMA_REASONING:-auto}"

# 채팅 GGUF
export LLAMA_GGUF_REPO="${LLAMA_GGUF_REPO:-bartowski/Qwen_Qwen3.5-9B-GGUF}"
export LLAMA_GGUF_FILE="${LLAMA_GGUF_FILE:-Qwen_Qwen3.5-9B-Q4_K_M.gguf}"
# 4B 예:
# export LLAMA_GGUF_REPO=jc-builds/Qwen3.5-4B-Q4_K_M-GGUF
# export LLAMA_GGUF_FILE=Qwen3.5-4B-Q4_K_M.gguf

py() {
  if [[ -x "$ROOT/.venv/bin/python" ]]; then
    "$ROOT/.venv/bin/python" "$@"
  else
    python3 "$@"
  fi
}

ensure_ml_embed() {
  py scripts/ensure_ml_models.py --embed-only
}

ensure_ml_ready() {
  py scripts/ensure_ml_models.py
}

ensure_llama_ready() {
  py scripts/ensure_llama_server.py
}

usage() {
  cat <<EOF

  ./scripts/local-rag.sh [serve|ingest|ask|llama|models]

    serve   웹 UI (기본) — models + llama-server 자동 준비
    ingest  PDF 인제스트 — embed 모델만 (Java 11+)
    ask     터미널 질문 — models + llama-server
    llama   llama-server 설치·GGUF·기동만
    models  임베딩+리랭커 다운로드만

  data/models/embeddings  data/models/reranker  data/models/chat
  EMBED=${EMBED_MODEL}  RERANK=${RERANK_MODEL}

EOF
}

ACTION="${1:-serve}"

case "$ACTION" in
  serve)
    ensure_ml_ready || echo "[경고] 임베딩·리랭커 준비 실패 — 질의가 실패할 수 있습니다."
    ensure_llama_ready || echo "[경고] llama-server 준비 실패 — 웹 UI만 띄웁니다. 질의는 실패할 수 있습니다."
    echo "[local-rag] 웹 UI  port=${WEB_PORT}  llama=${LLAMA_CPP_BASE_URL}"
    echo "[local-rag] CHAT=${LLAMA_GGUF_FILE}"
    echo "[local-rag] EMBED=${EMBED_MODEL}  RERANK=${RERANK_MODEL}"
    py app/main.py serve
    ;;
  ingest)
    if ! ensure_ml_embed; then
      echo "[오류] 임베딩 모델 ${EMBED_MODEL} 준비 실패"
      exit 1
    fi
    echo "[local-rag] PDF 인제스트  EMBED=${EMBED_MODEL}  Java 11+ 필요"
    py app/main.py ingest --reset
    ;;
  ask)
    if ! ensure_ml_ready; then
      echo "[오류] 임베딩·리랭커 모델 준비 실패"
      exit 1
    fi
    if ! ensure_llama_ready; then
      echo "[오류] llama-server 가 필요합니다."
      exit 1
    fi
    LR_Q="${2:-}"
    if [[ -z "$LR_Q" ]]; then
      read -r -p "질문: " LR_Q
    fi
    py app/main.py ask --question "$LR_Q"
    ;;
  llama)
    ensure_llama_ready
    ;;
  models)
    ensure_ml_ready
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    echo "[오류] 알 수 없는 명령: $ACTION"
    usage
    exit 1
    ;;
esac
