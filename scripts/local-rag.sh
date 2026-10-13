#!/usr/bin/env bash
# local-rag Linux 런처 (Windows: scripts/local-rag.bat)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [ -f ".env" ]; then
  set -a
  source <(grep -v '^#' .env | sed -e 's/\r$//')
  set +a
fi

# ============================================================
#  1. 모델 및 리랭커 설정
# ============================================================
export EMBED_MODEL="${EMBED_MODEL:-Qwen/Qwen3-Embedding-0.6B}"
# 리랭커 전략: origin (기본, Qwen) | laya (Laya Multilingual) | clm (Contrastive-LM)
export RERANK_STRATEGY="${RERANK_STRATEGY:-origin}"
export RERANK_MODEL="${RERANK_MODEL:-Qwen/Qwen3-Reranker-0.6B}"
# 최종 선별 청크 수 (0으로 설정 시 리랭킹 스킵)
export RERANK_TOP_K="${RERANK_TOP_K:-4}"
export FAISS_TOP_K="${FAISS_TOP_K:-8}"
# 문서 파서: opendataloader (고정밀) | pypdf (초고속 경량)
export DOC_PARSER="${DOC_PARSER:-opendataloader}"

# ============================================================
#  2. LLM 서빙 엔진 설정
# ============================================================
# 엔진 선택: llama.cpp (기본) | ollama | vllm | sglang | exllamav3 | mlc_llm | llamafile
export SERVING_ENGINE="${SERVING_ENGINE:-llama.cpp}"
export LLAMA_GGUF_REPO="${LLAMA_GGUF_REPO:-bartowski/Qwen_Qwen3.5-9B-GGUF}"
export LLAMA_GGUF_FILE="${LLAMA_GGUF_FILE:-Qwen_Qwen3.5-9B-Q4_K_M.gguf}"
# 가속 엔진 선택: vulkan (AMD/Intel iGPU 범용) | cuda (NVIDIA) | cpu (CPU 전용)
export LLAMA_CPP_VARIANT="${LLAMA_CPP_VARIANT:-}"
export LLAMA_CTX_SIZE="${LLAMA_CTX_SIZE:-4096}"
# GPU 오프로드 레이어 수: 999 (GPU 전량) | 0 (CPU 전용)
export LLAMA_NGL="${LLAMA_NGL:-999}"
export WEB_PORT="${WEB_PORT:-8787}"

py() {
  if [[ -x "$ROOT/.venv/bin/python" ]]; then
    "$ROOT/.venv/bin/python" "$@"
  else
    python3 "$@"
  fi
}

ensure_ml_embed() { py app/main.py setup --embed-only; }
ensure_ml_ready() { py app/main.py setup --models; }
ensure_llama_ready() { py app/main.py setup --engine; }

usage() {
  cat <<EOF

  ./scripts/local-rag.sh [serve|ingest|ask|llama|models|benchmark|test]
    serve      웹 UI 실행 (기본값)
    ingest     PDF 문서 인제스트 및 벡터 색인
    ask        터미널에서 직접 질문 질의
    llama      llama-server 설치 및 기동만 실행
    models     임베딩 및 리랭커 모델 다운로드만 실행
    benchmark  리랭커 3종(origin/laya/clm) 성능 및 가용성 측정
    test       단위 테스트 실행 (scripts/tests)

  [주요 옵션 환경 변수]
    RERANK_STRATEGY   : origin (기본, Qwen) | laya (Multilingual) | clm (Contrastive-LM)
    RERANK_TOP_K      : 4 (최종 선별 청크 수, 0=리랭킹 끄기)
    LLAMA_CPP_VARIANT : vulkan (iGPU 범용) | cuda (NVIDIA) | cpu (CPU 전용)
    LLAMA_NGL         : 999 (GPU 전량 오프로드) | 0 (CPU 전용)

  현재 설정: EMBED=${EMBED_MODEL}  RERANK=${RERANK_MODEL} [${RERANK_STRATEGY}]  ENGINE=${LLAMA_CPP_VARIANT:-auto}

EOF
}

ACTION="${1:-serve}"

case "$ACTION" in
  serve)
    ensure_ml_ready || echo "[경고] 임베딩·리랭커 준비 실패 - 질의가 실패할 수 있습니다."
    ensure_llama_ready || echo "[경고] llama-server 준비 실패 - 웹 UI만 기동합니다."
    echo "[local-rag] 웹 UI: port=${WEB_PORT}  llama=${LLAMA_CPP_BASE_URL:-http://localhost:8080/v1}"
    echo "[local-rag] 모델: CHAT=${LLAMA_GGUF_FILE}  EMBED=${EMBED_MODEL}  RERANK=${RERANK_MODEL} [${RERANK_STRATEGY}]"
    py app/main.py serve
    ;;
  benchmark)
    echo "[local-rag] 리랭커 백엔드(origin, laya, clm) 벤치마크 실행..."
    py scripts/tests/benchmark/benchmark_reranker.py
    ;;
  test)
    echo "[local-rag] 단위 테스트 실행 (scripts/tests)..."
    py -m unittest discover -s scripts/tests
    ;;
  ingest)
    if ! ensure_ml_embed; then
      echo "[오류] 임베딩 모델 ${EMBED_MODEL} 준비 실패"
      exit 1
    fi
    echo "[local-rag] PDF 인제스트 시작 (EMBED=${EMBED_MODEL}, PARSER=${DOC_PARSER})"
    py app/main.py ingest --reset --parser "${DOC_PARSER}"
    ;;
  ask)
    if ! ensure_ml_ready; then echo "[오류] 모델 준비 실패"; exit 1; fi
    if ! ensure_llama_ready; then echo "[오류] llama-server 필요"; exit 1; fi
    LR_Q="${2:-}"
    if [[ -z "$LR_Q" ]]; then read -r -p "질문: " LR_Q; fi
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
