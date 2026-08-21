@echo off
chcp 65001 >nul 2>&1
setlocal
cd /d "%~dp0.."

set "PY=.venv\Scripts\python.exe"

if not exist "%PY%" (
  echo [labeler] .venv 가 없습니다. 먼저 local-rag.bat 으로 환경을 만드세요.
  pause
  exit /b 1
)

rem 라벨링 도구 의존성 확인 (pypdfium2 / pillow / easyocr)
"%PY%" -c "import pypdfium2, PIL, easyocr" >nul 2>&1
if errorlevel 1 (
  echo [labeler] 의존성 설치 중 (pypdfium2, pillow, easyocr)...
  "%PY%" -m pip install -r tools\labeler\requirements.txt
)

rem VL llama-server (config.py VLM_* 기준, 이미 떠 있으면 스킵)
echo [labeler] VL llama-server 확인...
"%PY%" "scripts\ensure_vlm_server.py"
if errorlevel 1 (
  echo [경고] VL 서버 준비 실패 — 라벨러는 띄웁니다. VLM 설명 생성은 실패할 수 있습니다.
)

echo [labeler] http://localhost:8788 시작
start "" http://localhost:8788
"%PY%" tools\labeler\server.py
