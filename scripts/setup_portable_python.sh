#!/usr/bin/env bash
# tools/python 포터블 Python 준비 스크립트 (Linux / macOS)
# Windows: scripts/setup_portable_python.bat
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "============================================================"
echo " [local-rag] tools/python 포터블 Python 준비 스크립트"
echo "============================================================"

TOOLS_PY_DIR="$ROOT/tools/python"
PORTABLE_PY=""

# 1. 이미 존재하는 포터블 Python 확인
for cand in "$TOOLS_PY_DIR"/cpython-*/bin/python3 "$TOOLS_PY_DIR/bin/python3" "$TOOLS_PY_DIR/python3"; do
  if [[ -x "$cand" ]] && "$cand" -c "import sys" >/dev/null 2>&1; then
    PORTABLE_PY="$cand"
    break
  fi
done

if [[ -n "$PORTABLE_PY" ]]; then
  echo "[setup_portable_python] 이미 포터블 Python이 준비되어 있습니다:"
  echo "  > $PORTABLE_PY"
else
  echo "[setup_portable_python] uv 를 통해 Python 3.13 포터블 설치를 시도합니다..."
  UV_BIN="$(command -v uv || true)"
  if [[ -z "$UV_BIN" ]]; then
    echo "[setup_portable_python] uv 를 내려받아 설치합니다..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    UV_BIN="$(command -v uv || true)"
  fi

  if [[ -z "$UV_BIN" ]]; then
    echo "[setup_portable_python] ERROR: uv 설치 실패. 직접 Python 3.11~3.13을 시스템에 설치하세요."
    exit 1
  fi

  mkdir -p "$TOOLS_PY_DIR"
  "$UV_BIN" python install 3.13 --install-dir "$TOOLS_PY_DIR"

  for cand in "$TOOLS_PY_DIR"/cpython-*/bin/python3 "$TOOLS_PY_DIR/bin/python3"; do
    if [[ -x "$cand" ]]; then
      PORTABLE_PY="$cand"
      break
    fi
  done
fi

if [[ -n "$PORTABLE_PY" ]]; then
  echo "[setup_portable_python] 포터블 Python 준비 완료: $PORTABLE_PY"
else
  echo "[setup_portable_python] WARNING: 포터블 바이너리를 특정하지 못했습니다."
fi

exit 0

