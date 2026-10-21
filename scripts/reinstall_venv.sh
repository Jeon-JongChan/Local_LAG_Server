#!/usr/bin/env bash
# .venv 재구성 — lib(site-packages)만 보존, bin/pyvenv.cfg 는 현재 Python 으로 새로 생성
# Windows: scripts/reinstall_venv.bat
#
# 용도: 다른 Linux 머신으로 복사했거나 Python 경로가 달라 .venv/bin/python 이 깨졌을 때
# 주의: major.minor 가 다르면 바이너리 휠(.so)이 깨질 수 있음 → pip --force-reinstall
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VENV_DIR=".venv"
LIB_BACKUP=".venv_lib_backup"
REQ="requirements.txt"

echo "[reinstall_venv] 프로젝트: $ROOT"

# --- Python 찾기: tools/python 우선, 없으면 PATH ---
PY=""
for cand in "$ROOT"/tools/python/cpython-*/bin/python3 "$ROOT/tools/python/bin/python3" "$ROOT/tools/python/python3"; do
  if [[ -x "$cand" ]] && "$cand" -c "import sys" >/dev/null 2>&1; then
    PY="$cand"
    break
  fi
done
if [[ -z "$PY" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PY="$(python3 -c 'import sys; print(sys.executable)')"
  elif command -v python >/dev/null 2>&1; then
    PY="$(python -c 'import sys; print(sys.executable)')"
  else
    echo "[reinstall_venv] ERROR: tools/python 또는 PATH 에 python3 / python 이 없습니다."
    exit 1
  fi
fi

NEW_VER="$("$PY" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
echo "[reinstall_venv] 사용할 Python: $PY  ($NEW_VER)"

# Unix: lib / Windows 복사본: Lib
LIB_SRC=""
if [[ -d "$VENV_DIR/lib" ]]; then
  LIB_SRC="$VENV_DIR/lib"
elif [[ -d "$VENV_DIR/Lib" ]]; then
  LIB_SRC="$VENV_DIR/Lib"
fi

if [[ -z "$LIB_SRC" ]]; then
  echo "[reinstall_venv] 보존할 lib 가 없습니다. 새 venv 만 생성합니다."
else
  if [[ -f "$VENV_DIR/pyvenv.cfg" ]]; then
    echo "[reinstall_venv] 이전 pyvenv.cfg:"
    grep -E '^(version|version_info|home)' "$VENV_DIR/pyvenv.cfg" || true
  fi
  echo "[reinstall_venv] lib 백업 → $LIB_BACKUP"
  rm -rf "$LIB_BACKUP"
  mv "$LIB_SRC" "$LIB_BACKUP"
  echo "[reinstall_venv] 기존 $VENV_DIR 삭제..."
  rm -rf "$VENV_DIR"
fi

echo "[reinstall_venv] 새 venv 생성..."
if ! "$PY" -m venv "$VENV_DIR"; then
  echo "[reinstall_venv] ERROR: venv 생성 실패"
  if [[ -d "$LIB_BACKUP" ]]; then
    echo "[reinstall_venv] lib 백업은 $LIB_BACKUP 에 남아 있습니다."
  fi
  exit 1
fi

if [[ -d "$LIB_BACKUP" ]]; then
  echo "[reinstall_venv] 새 lib 제거 후 백업 복원..."
  rm -rf "$VENV_DIR/lib" "$VENV_DIR/Lib"
  # Unix venv 는 소문자 lib
  mv "$LIB_BACKUP" "$VENV_DIR/lib"
fi

VPY="$VENV_DIR/bin/python"
if [[ ! -x "$VPY" ]]; then
  echo "[reinstall_venv] ERROR: $VPY 없음"
  exit 1
fi

echo "[reinstall_venv] pip 확인/복구..."
"$VPY" -m ensurepip --upgrade >/dev/null 2>&1 || true
"$VPY" -m pip install --upgrade pip setuptools wheel || \
  echo "[reinstall_venv] WARNING: pip 업그레이드 실패 — 수동 확인 필요"

echo
echo "[reinstall_venv] 스모크 테스트..."
"$VPY" -c 'import sys; print("OK", sys.version)'

echo
echo "[reinstall_venv] 완료."
echo "  - 인터프리터: $VPY"
echo "  - lib 는 이전 패키지를 유지했습니다."
echo
echo "  major.minor 가 바뀌었거나 torch/numpy 등이 깨지면:"
echo "    $VPY -m pip install -r $REQ --force-reinstall"
echo "  라벨러 추가 의존성:"
echo "    $VPY -m pip install -r tools/labeler/requirements.txt"
echo
