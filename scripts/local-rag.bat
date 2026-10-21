@echo off
chcp 65001 >nul 2>&1
setlocal EnableDelayedExpansion
cd /d "%~dp0.."
set "PYTHONUNBUFFERED=1"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

echo.
echo [local-rag] 시작  (cwd=%CD%)

rem ============================================================
rem  0. Python 인터프리터 탐색 및 검증
rem  우선순위:
rem    1) 환경 변수 PYTHON_EXE (명시 지정)
rem    2) .venv\Scripts\python.exe (유효성 검사, 필요 시 auto-heal)
rem    3) tools\python (포터블 Python)
rem    4) 전역 시스템 Python (PATH 내 python, py -3)
rem ============================================================
set "PY="
set "PY_LABEL="
set "PORTABLE_PY="

rem 포터블 Python 후보 탐색
for /d %%I in ("tools\python\cpython-*") do (
  if not defined PORTABLE_PY if exist "%%~fI\python.exe" (
    "%%~fI\python.exe" -c "import sys" >nul 2>&1
    if not errorlevel 1 set "PORTABLE_PY=%%~fI\python.exe"
  )
)
if not defined PORTABLE_PY if exist "tools\python\python.exe" (
  "tools\python\python.exe" -c "import sys" >nul 2>&1
  if not errorlevel 1 set "PORTABLE_PY=%CD%\tools\python\python.exe"
)

rem 1. 사용자 지정 환경변수 PYTHON_EXE
if defined PYTHON_EXE (
  "%PYTHON_EXE%" -c "import sys" >nul 2>&1
  if not errorlevel 1 (
    set "PY=%PYTHON_EXE%"
    set "PY_LABEL=PYTHON_EXE: %PYTHON_EXE%"
  ) else (
    echo [local-rag] 경고: 지정된 PYTHON_EXE 를 실행할 수 없습니다: !PYTHON_EXE!
  )
)

rem 2. .venv 가상환경 검사 및 필요 시 복구
if not defined PY if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -c "import sys" >nul 2>&1
  if not errorlevel 1 (
    set "PY=%CD%\.venv\Scripts\python.exe"
    set "PY_LABEL=.venv"
  ) else (
    rem .venv 가 깨져있는 경우 (예: base python 경로 이동)
    if defined PORTABLE_PY (
      echo [local-rag] 경고: .venv 인터프리터 바인딩이 깨져 있어 tools\python 으로 자동 복구합니다...
      call :heal_venv_cfg "!PORTABLE_PY!"
      ".venv\Scripts\python.exe" -c "import sys" >nul 2>&1
      if not errorlevel 1 (
        set "PY=%CD%\.venv\Scripts\python.exe"
        set "PY_LABEL=.venv [tools\python 복구됨]"
      )
    )
  )
)

rem 3. tools\python 포터블 Python 직접 사용
if not defined PY if defined PORTABLE_PY (
  set "PY=!PORTABLE_PY!"
  set "PY_LABEL=tools\python [포터블 Python]"
)

rem 4. 전역 시스템 Python (PATH) 탐색
if not defined PY (
  where python >nul 2>&1
  if not errorlevel 1 (
    for /f "delims=" %%i in ('where python 2^>nul') do (
      if not defined PY (
        "%%i" -c "import sys" >nul 2>&1
        if not errorlevel 1 (
          set "PY=%%i"
          set "PY_LABEL=시스템 python: %%i"
        )
      )
    )
  )
)
if not defined PY (
  where py >nul 2>&1
  if not errorlevel 1 (
    for /f "delims=" %%i in ('py -3 -c "import sys; print(sys.executable)" 2^>nul') do (
      if not defined PY (
        "%%i" -c "import sys" >nul 2>&1
        if not errorlevel 1 (
          set "PY=%%i"
          set "PY_LABEL=시스템 py -3: %%i"
        )
      )
    )
  )
)

rem 5. 실행 가능한 Python이 전혀 없는 경우 에러 발생
if not defined PY (
  echo.
  echo [local-rag] [오류] 실행 가능한 Python 인터프리터를 찾을 수 없습니다.
  echo   - 전역 PATH 에 Python이 설치되어 있지 않거나 실행할 수 없습니다.
  echo   - tools\python 에 유효한 포터블 Python 이 존재하지 않습니다.
  echo.
  echo  해결 방법:
  echo   [1] 시스템에 Python 3.11~3.13을 설치하고 PATH에 추가하세요.
  echo   [2] 또는 scripts\setup_portable_python.bat 을 실행해 tools\python 에 포터블 Python을 준비하세요.
  echo.
  pause
  exit /b 1
)

echo [local-rag] Python: !PY_LABEL!

rem .env 파일 자동 로드 (정의되지 않은 환경변수에 한함)
if exist ".env" (
  echo [local-rag] .env 로드
  for /f "usebackq eol=# tokens=1* delims==" %%a in (".env") do (
    if not "%%a"=="" if not defined %%a set "%%a=%%b"
  )
) else (
  echo [local-rag] .env 없음 — 기본값 사용
)

rem ============================================================
rem  1. 모델 및 리랭커 설정
rem ============================================================
if not defined EMBED_MODEL      set "EMBED_MODEL=Qwen/Qwen3-Embedding-0.6B"
rem 리랭커: origin=Qwen 로컬 | laya=laya 패키지 | clm=외부 clm-serve
if not defined RERANK_STRATEGY  set "RERANK_STRATEGY=laya"
if not defined RERANK_MODEL     set "RERANK_MODEL=Qwen/Qwen3-Reranker-0.6B"
if not defined RERANK_LAYA_MODEL set "RERANK_LAYA_MODEL=convaiinnovations/laya-multilingual"
if not defined RERANK_CLM_BASE_URL set "RERANK_CLM_BASE_URL=http://127.0.0.1:8700"
if /i "!RERANK_STRATEGY!"=="laya" (
  set "RERANK_LABEL=laya (!RERANK_LAYA_MODEL!)"
) else if /i "!RERANK_STRATEGY!"=="clm" (
  set "RERANK_LABEL=clm (!RERANK_CLM_BASE_URL!)"
) else (
  set "RERANK_LABEL=origin (!RERANK_MODEL!)"
)
rem 최종 선별 청크 수 (0으로 설정 시 리랭킹 스킵):
if not defined RERANK_TOP_K     set "RERANK_TOP_K=4"
if not defined FAISS_TOP_K      set "FAISS_TOP_K=8"
rem 문서 파서: opendataloader (고정밀 표/레이아웃 추출) | pypdf (초고속 경량 순수파이썬)
if not defined DOC_PARSER        set "DOC_PARSER=opendataloader"

rem ============================================================
rem  2. LLM 서빙 엔진 설정
rem ============================================================
rem 엔진 선택: llama.cpp (기본) | ollama | vllm | sglang | exllamav3 | mlc_llm | llamafile
if not defined SERVING_ENGINE     set "SERVING_ENGINE=llama.cpp"
if not defined LLAMA_GGUF_REPO   set "LLAMA_GGUF_REPO=bartowski/Qwen_Qwen3.5-9B-GGUF"
if not defined LLAMA_GGUF_FILE   set "LLAMA_GGUF_FILE=Qwen_Qwen3.5-9B-Q4_K_M.gguf"
rem 가속 엔진 선택: auto (ACCELERATOR 추종) | cuda (NVIDIA) | vulkan (AMD/Intel iGPU 범용) | cpu (CPU 전용)
if not defined LLAMA_CPP_VARIANT set "LLAMA_CPP_VARIANT=auto"
if not defined LLAMA_CPP_BASE_URL set "LLAMA_CPP_BASE_URL=http://localhost:8080/v1"
if not defined LLAMA_CTX_SIZE    set "LLAMA_CTX_SIZE=4096"
rem GPU 오프로드 레이어 수: 999 (GPU 전량 오프로드) | 0 (CPU 전용)
if not defined LLAMA_NGL         set "LLAMA_NGL=999"
if not defined WEB_PORT          set "WEB_PORT=8787"

rem ============================================================
rem  3. 명령 라우팅
rem ============================================================
set "ACTION=%~1"
if "!ACTION!"=="" set "ACTION=serve"
echo [local-rag] 명령: !ACTION!  ENGINE=!SERVING_ENGINE!  PROFILE=!PROFILE!
echo.

if /i "!ACTION!"=="serve"     goto :serve
if /i "!ACTION!"=="ingest"    goto :ingest
if /i "!ACTION!"=="ask"       goto :ask
if /i "!ACTION!"=="llama"     goto :llama_only
if /i "!ACTION!"=="models"    goto :models_only
if /i "!ACTION!"=="benchmark" goto :benchmark
if /i "!ACTION!"=="test"      goto :run_tests
if /i "!ACTION!"=="help"      goto :usage
if /i "!ACTION!"=="/?"        goto :usage

echo [오류] 알 수 없는 명령: !ACTION!
goto :usage

rem ------------------------------------------------------------
:serve
echo [local-rag] (1/3) 임베딩 확인 (Qwen 리랭커 다운로드는 origin일 때만)...
call :ensure_ml_ready
if errorlevel 1 echo [경고] 임베딩·리랭커 준비 실패 - 질의가 실패할 수 있습니다.

echo [local-rag] (2/3) 서빙 엔진 확인 (llama.cpp 는 data/serving_engines 에 받아 기동, 그 외는 외부 서버)...
call :ensure_llama_ready
if errorlevel 1 echo [경고] llama-server 준비 실패 - 웹 UI만 기동합니다.

echo [local-rag] (3/3) 웹 UI 기동
if /i "!SERVING_ENGINE!"=="llama.cpp" (
  echo [local-rag] 웹 UI: port=!WEB_PORT!  SERVING=llama.cpp ^(!LLAMA_CPP_BASE_URL!^) · data/serving_engines
) else (
  echo [local-rag] 웹 UI: port=!WEB_PORT!  SERVING=!SERVING_ENGINE! · 외부 서버
)
echo [local-rag] 모델: CHAT=!LLAMA_GGUF_FILE!  EMBED=!EMBED_MODEL!  RERANK=!RERANK_LABEL!
echo [local-rag] 브라우저: http://127.0.0.1:!WEB_PORT!  ^(이 창을 닫으면 서버도 종료^)

"!PY!" app\main.py serve
echo [local-rag] 서버 종료됨. 오류가 있으면 위를 확인하세요.
pause
exit /b 0

rem ------------------------------------------------------------
:ingest
call :ensure_ml_embed
if errorlevel 1 (
  echo [오류] 임베딩 모델 !EMBED_MODEL! 준비 실패
  pause
  exit /b 1
)
echo [local-rag] PDF 인제스트 시작 (EMBED=!EMBED_MODEL!)
set "EXTRA_ARGS=%*"
set "EXTRA_ARGS=!EXTRA_ARGS:*ingest=!"
"!PY!" app\main.py ingest --reset --parser "!DOC_PARSER!" !EXTRA_ARGS!
pause
exit /b 0

rem ------------------------------------------------------------
:ask
call :ensure_ml_ready
if errorlevel 1 ( echo [오류] 모델 준비 실패 & pause & exit /b 1 )
call :ensure_llama_ready
if errorlevel 1 ( echo [오류] llama-server 필요 & pause & exit /b 1 )

set "LR_Q=%~2"
if "!LR_Q!"=="" set /p LR_Q="질문: "

"!PY!" app\main.py ask --question "!LR_Q!"
pause
exit /b 0

rem ------------------------------------------------------------
:benchmark
echo [local-rag] 리랭커 백엔드(origin, laya, clm) 벤치마크 실행...
"!PY!" scripts\tests\benchmark\benchmark_reranker.py
pause
exit /b 0

rem ------------------------------------------------------------
:llama_only
call :ensure_llama_ready
pause
exit /b 0

:models_only
call :ensure_ml_ready
pause
exit /b 0

rem ------------------------------------------------------------
:ensure_ml_embed
echo [local-rag]   ^> python app\main.py setup --embed-only
"!PY!" app\main.py setup --embed-only
exit /b !ERRORLEVEL!

:ensure_ml_ready
echo [local-rag]   ^> python app\main.py setup --models
"!PY!" app\main.py setup --models
exit /b !ERRORLEVEL!

:ensure_llama_ready
echo [local-rag]   ^> python app\main.py setup --engine
"!PY!" app\main.py setup --engine
exit /b !ERRORLEVEL!

rem ------------------------------------------------------------
:run_tests
echo [local-rag] 단위 테스트 실행 (scripts/tests)...
"!PY!" -m unittest discover -s scripts\tests
pause
exit /b !ERRORLEVEL!

rem ------------------------------------------------------------
:heal_venv_cfg
rem 인자: %1 = 유효한 base python.exe 경로
for %%P in (%1) do set "H_HOME=%%~dpP"
if "!H_HOME:~-1!"=="\" set "H_HOME=!H_HOME:~0,-1!"
for /f "delims=" %%v in ('%1 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')"') do set "H_VER=%%v"
(
  echo home = !H_HOME!
  echo include-system-site-packages = false
  echo version = !H_VER!
  echo executable = %~1
  echo command = %~1 -m venv %CD%\.venv
) > ".venv\pyvenv.cfg"
exit /b 0

rem ------------------------------------------------------------
:usage
echo.
echo  local-rag.bat [serve^|ingest^|ask^|llama^|models^|benchmark^|test]
echo    serve      웹 UI 실행 (기본값)
echo    ingest     PDF 문서 인제스트 및 벡터 색인 (옵션: --fresh 캐시 초기화)
echo    ask        터미널에서 직접 질문 질의
echo    llama      llama-server 설치 및 기동만 실행
echo    models     임베딩 다운로드 (origin이면 Qwen 리랭커도)
echo    benchmark  리랭커 3종(origin/laya/clm) 성능 및 가용성 측정
echo    test       단위 테스트 실행 (scripts/tests)
echo.
echo  [주요 옵션 환경 변수]
echo    RERANK_STRATEGY   : origin (Qwen 로컬) ^| laya (파이썬 패키지, 프로세스 안) ^| clm (외부 clm-serve)
echo    RERANK_TOP_K      : 4 (최종 선별 청크 수, 0=리랭킹 끄기)
echo    LLAMA_CPP_VARIANT : vulkan (iGPU 범용) ^| cuda (NVIDIA) ^| cpu (CPU 전용)
echo    LLAMA_NGL         : 999 (GPU 전량 오프로드) ^| 0 (CPU 전용)
echo.
echo  현재 설정: EMBED=!EMBED_MODEL!  RERANK=!RERANK_LABEL!  ENGINE=!SERVING_ENGINE!  VARIANT=!LLAMA_CPP_VARIANT!
echo.
exit /b 0
