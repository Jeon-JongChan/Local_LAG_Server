#!/usr/bin/env bash
# 라벨링 도구 Linux 런처 (Windows: scripts/labeler.bat)
# VL llama-server 가 없으면 scripts/ensure_vlm_server.py 로 기동
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="$ROOT/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "[labeler] .venv 가 없습니다. 먼저 다음을 실행하세요:"
  echo "  python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
  echo "  또는: ./scripts/reinstall_venv.sh"
  exit 1
fi

if ! "$PY" -c "import pypdfium2, PIL, easyocr" >/dev/null 2>&1; then
  echo "[labeler] 의존성 설치 중 (pypdfium2, pillow, easyocr)..."
  "$PY" -m pip install -r tools/labeler/requirements.txt
fi

echo "[labeler] VL llama-server 확인..."
if ! "$PY" scripts/ensure_vlm_server.py; then
  echo "[경고] VL 서버 준비 실패 — 라벨러는 띄웁니다. VLM 설명 생성은 실패할 수 있습니다."
fi

echo "[labeler] http://localhost:8788 시작"
if command -v xdg-open >/dev/null 2>&1; then
  xdg-open "http://localhost:8788" >/dev/null 2>&1 || true
elif command -v open >/dev/null 2>&1; then
  open "http://localhost:8788" >/dev/null 2>&1 || true
fi

exec "$PY" tools/labeler/server.py
