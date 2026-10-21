@echo off
chcp 65001 >nul 2>&1
setlocal EnableDelayedExpansion
cd /d "%~dp0.."

echo.
echo ============================================================
echo  [local-rag] tools\python 포터블 Python 준비 스크립트
echo ============================================================
echo.

set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "TOOLS_PY_DIR=%CD%\tools\python"
set "PORTABLE_PY="

rem --- 1. 이미 존재하는 포터블 Python 확인 ---
for /d %%I in ("%TOOLS_PY_DIR%\cpython-*") do (
  if not defined PORTABLE_PY if exist "%%~fI\python.exe" (
    "%%~fI\python.exe" -c "import sys" >nul 2>&1
    if not errorlevel 1 set "PORTABLE_PY=%%~fI\python.exe"
  )
)
if not defined PORTABLE_PY if exist "%TOOLS_PY_DIR%\python.exe" (
  "%TOOLS_PY_DIR%\python.exe" -c "import sys" >nul 2>&1
  if not errorlevel 1 set "PORTABLE_PY=%TOOLS_PY_DIR%\python.exe"
)

if defined PORTABLE_PY (
  echo [setup_portable_python] 이미 포터블 Python이 준비되어 있습니다:
  echo   ^> !PORTABLE_PY!
  goto :after_python_ready
)

rem --- 2. uv 도구 탐색 및 다운로드 ---
echo [setup_portable_python] tools\python 에 포터블 Python (CPython 3.13) 다운로드를 시작합니다...
set "UV_EXE="
where uv >nul 2>&1
if not errorlevel 1 (
  for /f "delims=" %%i in ('where uv 2^>nul') do if not defined UV_EXE set "UV_EXE=%%i"
)
if not defined UV_EXE if exist "D:\Projects\.tools\Graphify\portable\uv\uv.exe" (
  set "UV_EXE=D:\Projects\.tools\Graphify\portable\uv\uv.exe"
)
if not defined UV_EXE if exist "%CD%\tools\uv\uv.exe" (
  set "UV_EXE=%CD%\tools\uv\uv.exe"
)

if not defined UV_EXE (
  echo [setup_portable_python] uv 바이너리 다운로드 중 (tools\uv)...
  if not exist "tools\uv" mkdir "tools\uv"
  powershell -NoProfile -Command ^
    "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; " ^
    "$zip = 'tools\uv\uv.zip'; " ^
    "Invoke-WebRequest -Uri 'https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip' -OutFile $zip; " ^
    "Expand-Archive -Path $zip -DestinationPath 'tools\uv' -Force; " ^
    "if (Test-Path $zip) { Remove-Item $zip -Force }"
  if exist "tools\uv\uv.exe" set "UV_EXE=%CD%\tools\uv\uv.exe"
)

if not defined UV_EXE (
  echo [setup_portable_python] [오류] uv를 찾거나 다운로드할 수 없습니다.
  echo 네트워크 연결을 확인하거나 직접 Python 3.13을 tools\python 에 배치하세요.
  pause
  exit /b 1
)

rem --- 3. uv를 통한 Python 3.13 다운로드 ---
echo [setup_portable_python] uv (!UV_EXE!) 로 Python 3.13 설치 중...
if not exist "%TOOLS_PY_DIR%" mkdir "%TOOLS_PY_DIR%"
"%UV_EXE%" python install 3.13 --install-dir "%TOOLS_PY_DIR%"
if errorlevel 1 (
  echo [setup_portable_python] [오류] 포터블 Python 설치 실패
  pause
  exit /b 1
)

for /d %%I in ("%TOOLS_PY_DIR%\cpython-*") do (
  if not defined PORTABLE_PY if exist "%%~fI\python.exe" set "PORTABLE_PY=%%~fI\python.exe"
)
if not defined PORTABLE_PY (
  echo [setup_portable_python] [오류] 설치 완료 후 python.exe를 찾을 수 없습니다.
  pause
  exit /b 1
)

echo [setup_portable_python] 포터블 Python 준비 완료: !PORTABLE_PY!

:after_python_ready
rem --- 4. .venv 연결 상태 점검 및 복구 ---
if exist ".venv\pyvenv.cfg" (
  echo [setup_portable_python] 기존 .venv 가 발견되었습니다. 유효성 검사 중...
  ".venv\Scripts\python.exe" -c "import sys" >nul 2>&1
  if errorlevel 1 (
    echo [setup_portable_python] .venv 인터프리터 바인딩이 깨져 있어 복구합니다...
    for %%P in ("!PORTABLE_PY!") do set "PY_HOME=%%~dpP"
    if "!PY_HOME:~-1!"=="\" set "PY_HOME=!PY_HOME:~0,-1!"
    (
      echo home = !PY_HOME!
      echo include-system-site-packages = false
      echo version = 3.13.16
      echo executable = !PORTABLE_PY!
      echo command = !PORTABLE_PY! -m venv %CD%\.venv
    ) > ".venv\pyvenv.cfg"
    echo [setup_portable_python] .venv\pyvenv.cfg 가 업데이트되었습니다.
  ) else (
    echo [setup_portable_python] .venv 가 정상 동작 중입니다.
  )
)

rem --- 5. labeler .venv 도 있으면 함께 점검 ---
if exist "tools\labeler\.venv\pyvenv.cfg" (
  "tools\labeler\.venv\Scripts\python.exe" -c "import sys" >nul 2>&1
  if errorlevel 1 (
    echo [setup_portable_python] tools\labeler\.venv 인터프리터 바인딩 복구 중...
    for %%P in ("!PORTABLE_PY!") do set "PY_HOME=%%~dpP"
    if "!PY_HOME:~-1!"=="\" set "PY_HOME=!PY_HOME:~0,-1!"
    (
      echo home = !PY_HOME!
      echo include-system-site-packages = false
      echo version = 3.13.16
      echo executable = !PORTABLE_PY!
      echo command = !PORTABLE_PY! -m venv %CD%\tools\labeler\.venv
    ) > "tools\labeler\.venv\pyvenv.cfg"
    echo [setup_portable_python] tools\labeler\.venv\pyvenv.cfg 가 업데이트되었습니다.
  )
)
echo [setup_portable_python] GPU(CUDA) 가속 환경 확인 중...
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -c "import sys; sys.path.insert(0, 'app'); from services.gpu_manager import ensure_gpu_torch; ensure_gpu_torch()"
)

echo.
echo [setup_portable_python] 모든 포터블 환경 설정이 완료되었습니다.
echo 이제 scripts\local-rag.bat 을 자유롭게 실행하실 수 있습니다.
echo.
pause
exit /b 0

