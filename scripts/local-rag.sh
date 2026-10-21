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
# 리랭커: origin=Qwen 로컬 | laya=laya 패키지 | clm=외부 clm-serve
export RERANK_STRATEGY="${RERANK_STRATEGY:-laya}"
export RERANK_MODEL="${RERANK_MODEL:-Qwen/Qwen3-Reranker-0.6B}"
export RERANK_LAYA_MODEL="${RERANK_LAYA_MODEL:-convaiinnovations/laya-multilingual}"
export RERANK_CLM_BASE_URL="${RERANK_CLM_BASE_URL:-http://127.0.0.1:8700}"
case "$RERANK_STRATEGY" in
  laya) RERANK_LABEL="laya (${RERANK_LAYA_MODEL})" ;;
  clm) RERANK_LABEL="clm (${RERANK_CLM_BASE_URL})" ;;
  *) RERANK_LABEL="origin (${RERANK_MODEL})" ;;
esac
export LLAMA_CPP_BASE_URL="${LLAMA_CPP_BASE_URL:-http://localhost:8080/v1}"
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
  if [[ -n "${PYTHON_EXE:-}" ]] && "$PYTHON_EXE" -c "import sys" >/dev/null 2>&1; then
    "$PYTHON_EXE" "$@"
    return $?
  fi
  if [[ -x "$ROOT/.venv/bin/python" ]] && "$ROOT/.venv/bin/python" -c "import sys" >/dev/null 2>&1; then
    "$ROOT/.venv/bin/python" "$@"
    return $?
  fi
  for cand in "$ROOT"/tools/python/cpython-*/bin/python3 "$ROOT/tools/python/bin/python3" "$ROOT/tools/python/python3"; do
    if [[ -x "$cand" ]] && "$cand" -c "import sys" >/dev/null 2>&1; then
      "$cand" "$@"
      return $?
    fi
  done
  if command -v python3 >/dev/null 2>&1 && python3 -c "import sys" >/dev/null 2>&1; then
    python3 "$@"
    return $?
  elif command -v python >/dev/null 2>&1 && python -c "import sys" >/dev/null 2>&1; then
    python "$@"
    return $?
  fi
  echo "[local-rag] [ERROR] 실행 가능한 Python 인터프리터를 찾을 수 없습니다 (tools/python, .venv, PATH)." >&2
  exit 1
}

ensure_ml_embed() { py app/main.py setup --embed-only; }
ensure_ml_ready() { py app/main.py setup --models; }
ensure_llama_ready() { py app/main.py setup --engine; }

usage() {
  cat <<EOF

  ./scripts/local-rag.sh [serve|ingest|ask|llama|models|benchmark|test]
    serve      웹 UI 실행 (기본값)
    ingest     PDF 문서 인제스트 및 벡터 색인 (옵션: --fresh 캐시 초기화)
    ask        터미널에서 직접 질문 질의
    llama      llama-server 설치 및 기동만 실행
    models     임베딩 다운로드 (origin이면 Qwen 리랭커도)
    benchmark  리랭커 3종(origin/laya/clm) 성능 및 가용성 측정
    test       단위 테스트 실행 (scripts/tests)

  [주요 옵션 환경 변수]
    RERANK_STRATEGY   : origin (Qwen 로컬) | laya (파이썬 패키지, 프로세스 안) | clm (외부 clm-serve)
    RERANK_TOP_K      : 4 (최종 선별 청크 수, 0=리랭킹 끄기)
    LLAMA_CPP_VARIANT : auto (ACCELERATOR 추종) | cuda (NVIDIA) | vulkan (iGPU 범용) | cpu (CPU 전용)
    LLAMA_NGL         : 999 (GPU 전량 오프로드) | 0 (CPU 전용)

  현재 설정: EMBED=${EMBED_MODEL}  RERANK=${RERANK_LABEL}  ENGINE=${SERVING_ENGINE}

EOF
}

ACTION="${1:-serve}"

case "$ACTION" in
  serve)
    ensure_ml_ready || echo "[경고] 임베딩·리랭커 준비 실패 - 질의가 실패할 수 있습니다."
    ensure_llama_ready || echo "[경고] llama-server 준비 실패 - 웹 UI만 기동합니다."
    if [ "$SERVING_ENGINE" = "llama.cpp" ]; then
      echo "[local-rag] 웹 UI: port=${WEB_PORT}  SERVING=llama.cpp (${LLAMA_CPP_BASE_URL}) · data/serving_engines"
    else
      echo "[local-rag] 웹 UI: port=${WEB_PORT}  SERVING=${SERVING_ENGINE} · 외부 서버"
    fi
    echo "[local-rag] 모델: CHAT=${LLAMA_GGUF_FILE}  EMBED=${EMBED_MODEL}  RERANK=${RERANK_LABEL}"
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
    shift
    py app/main.py ingest --reset --parser "${DOC_PARSER}" "$@"
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
