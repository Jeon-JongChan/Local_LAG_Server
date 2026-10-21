@echo off
chcp 65001 >nul 2>&1
setlocal EnableDelayedExpansion
cd /d "%~dp0.."

rem ============================================================
rem  .venv 재구성 — Lib 만 보존, Scripts/pyvenv.cfg 등은 현재 Python 으로 새로 생성
rem
rem  용도: 다른 PC로 복사했거나 Python 경로/마이너 버전이 달라
rem        .venv\Scripts\python.exe 가 깨졌을 때, 이미 받아 둔 site-packages(Lib)를
rem        다시 받지 않고 인터프리터 바인딩만 고친다.
rem
rem  주의: major.minor 가 다르면(예: 3.11 → 3.12) 바이너리 휠(.pyd)이
rem        깨질 수 있다. 그 경우 스크립트 끝의 pip 재설치를 권장한다.
rem ============================================================

set "VENV_DIR=.venv"
set "LIB_BACKUP=.venv_Lib_backup"
set "REQ=requirements.txt"

echo [reinstall_venv] 프로젝트: %CD%

rem --- Python 찾기: tools\python 우선 탐색 후 시스템 PATH ---
set "PY="
for /d %%I in ("tools\python\cpython-*") do (
  if not defined PY if exist "%%~fI\python.exe" (
    "%%~fI\python.exe" -c "import sys" >nul 2>&1
    if not errorlevel 1 set "PY=%%~fI\python.exe"
  )
)
if not defined PY if exist "tools\python\python.exe" (
  "tools\python\python.exe" -c "import sys" >nul 2>&1
  if not errorlevel 1 set "PY=%CD%\tools\python\python.exe"
)
if not defined PY (
  where py >nul 2>&1
  if not errorlevel 1 (
    for /f "delims=" %%i in ('py -3 -c "import sys; print(sys.executable)" 2^>nul') do (
      "%%i" -c "import sys" >nul 2>&1
      if not errorlevel 1 set "PY=%%i"
    )
  )
)
if not defined PY (
  where python >nul 2>&1
  if not errorlevel 1 (
    for /f "delims=" %%i in ('where python 2^>nul') do (
      if not defined PY (
        "%%i" -c "import sys" >nul 2>&1
        if not errorlevel 1 set "PY=%%i"
      )
    )
  )
)
if not defined PY (
  echo [reinstall_venv] ERROR: tools\python 또는 PATH 에 유효한 python / py 가 없습니다.
  exit /b 1
)

for /f "delims=" %%i in ('"%PY%" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"') do set "NEW_VER=%%i"
echo [reinstall_venv] 사용할 Python: %PY%  (%NEW_VER%)

if not exist "%VENV_DIR%\Lib" (
  echo [reinstall_venv] 보존할 %VENV_DIR%\Lib 가 없습니다. 새 venv 만 생성합니다.
  goto :create_fresh
)

rem --- 이전 venv Python 버전 힌트 (있으면 경고) ---
if exist "%VENV_DIR%\pyvenv.cfg" (
  for /f "tokens=2 delims== " %%a in ('findstr /b /c:"version_info" /c:"version" "%VENV_DIR%\pyvenv.cfg"') do (
    echo [reinstall_venv] 이전 pyvenv.cfg version 힌트: %%a
  )
)

echo [reinstall_venv] Lib 백업 → %LIB_BACKUP%
if exist "%LIB_BACKUP%" (
  echo [reinstall_venv] 기존 백업 폴더 삭제...
  rmdir /s /q "%LIB_BACKUP%"
)
move "%VENV_DIR%\Lib" "%LIB_BACKUP%" >nul
if errorlevel 1 (
  echo [reinstall_venv] ERROR: Lib 이동 실패
  exit /b 1
)

echo [reinstall_venv] 기존 %VENV_DIR% 삭제...
rmdir /s /q "%VENV_DIR%"
if exist "%VENV_DIR%" (
  echo [reinstall_venv] ERROR: %VENV_DIR% 삭제 실패 (파일 잠금?). 백업은 %LIB_BACKUP%
  exit /b 1
)

:create_fresh
echo [reinstall_venv] 새 venv 생성...
"%PY%" -m venv "%VENV_DIR%"
if errorlevel 1 (
  echo [reinstall_venv] ERROR: venv 생성 실패
  if exist "%LIB_BACKUP%" (
    echo [reinstall_venv] Lib 백업은 %LIB_BACKUP% 에 남아 있습니다.
  )
  exit /b 1
)

if exist "%LIB_BACKUP%" (
  echo [reinstall_venv] 새 Lib 제거 후 백업 복원...
  rmdir /s /q "%VENV_DIR%\Lib"
  move "%LIB_BACKUP%" "%VENV_DIR%\Lib" >nul
  if errorlevel 1 (
    echo [reinstall_venv] ERROR: Lib 복원 실패. 백업: %LIB_BACKUP%
    exit /b 1
  )
)

set "VPY=%VENV_DIR%\Scripts\python.exe"
if not exist "%VPY%" (
  echo [reinstall_venv] ERROR: %VPY% 없음
  exit /b 1
)

echo [reinstall_venv] pip 확인/복구...
"%VPY%" -m ensurepip --upgrade >nul 2>&1
"%VPY%" -m pip install --upgrade pip setuptools wheel
if errorlevel 1 (
  echo [reinstall_venv] WARNING: pip 업그레이드 실패 — 수동 확인 필요
)

echo.
echo [reinstall_venv] 스모크 테스트 (import sys)...
"%VPY%" -c "import sys; print('OK', sys.version)"
if errorlevel 1 (
  echo [reinstall_venv] ERROR: 인터프리터 실행 실패
  exit /b 1
)

echo.
echo [reinstall_venv] 완료.
echo   - 인터프리터: %VPY%
echo   - Lib 는 이전 패키지를 유지했습니다.
echo.
echo   major.minor 가 바뀌었거나 torch/numpy 등이 깨지면:
echo     %VPY% -m pip install -r %REQ% --force-reinstall
echo   라벨러 추가 의존성:
echo     %VPY% -m pip install -r tools\labeler\requirements.txt
echo.
exit /b 0
