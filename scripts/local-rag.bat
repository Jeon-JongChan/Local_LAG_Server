@echo off
chcp 65001 >nul 2>&1
setlocal EnableDelayedExpansion
cd /d "%~dp0.."

rem .env 파일 자동 로드 (정의되지 않은 환경변수에 한함)
if exist ".env" (
  for /f "usebackq eol=# tokens=1* delims==" %%a in (".env") do (
    if not "%%a"=="" if not defined %%a set "%%a=%%b"
  )
)

rem ============================================================
rem  1. 모델 및 리랭커 설정
rem ============================================================
if not defined EMBED_MODEL      set "EMBED_MODEL=Qwen/Qwen3-Embedding-0.6B"
rem 리랭커 전략: origin (기본, Qwen 크로스인코더) | laya (Laya Multilingual) | clm (Contrastive-LM)
if not defined RERANK_STRATEGY  set "RERANK_STRATEGY=laya"
if not defined RERANK_MODEL     set "RERANK_MODEL=Qwen/Qwen3-Reranker-0.6B"
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
rem 가속 엔진 선택: vulkan (AMD/Intel iGPU 범용) | cuda (NVIDIA) | cpu (CPU 전용)
if not defined LLAMA_CPP_VARIANT set "LLAMA_CPP_VARIANT=vulkan"
if not defined LLAMA_CTX_SIZE    set "LLAMA_CTX_SIZE=4096"
rem GPU 오프로드 레이어 수: 999 (GPU 전량 오프로드) | 0 (CPU 전용)
if not defined LLAMA_NGL         set "LLAMA_NGL=999"
if not defined WEB_PORT          set "WEB_PORT=8787"

rem ============================================================
rem  3. 명령 라우팅
rem ============================================================
set "ACTION=%~1"
if "!ACTION!"=="" set "ACTION=serve"

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
call :ensure_ml_ready
if errorlevel 1 echo [경고] 임베딩·리랭커 준비 실패 - 질의가 실패할 수 있습니다.
call :ensure_llama_ready
if errorlevel 1 echo [경고] llama-server 준비 실패 - 웹 UI만 기동합니다.

echo [local-rag] 웹 UI: port=!WEB_PORT!  llama=!LLAMA_CPP_BASE_URL!
echo [local-rag] 모델: CHAT=!LLAMA_GGUF_FILE!  EMBED=!EMBED_MODEL!  RERANK=!RERANK_MODEL! [!RERANK_STRATEGY!]

if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" app\main.py serve
) else (
  python app\main.py serve
)
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
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" app\main.py ingest --reset --parser "!DOC_PARSER!"
) else (
  python app\main.py ingest --reset --parser "!DOC_PARSER!"
)
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

if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" app\main.py ask --question "!LR_Q!"
) else (
  python app\main.py ask --question "!LR_Q!"
)
pause
exit /b 0

rem ------------------------------------------------------------
:benchmark
echo [local-rag] 리랭커 백엔드(origin, laya, clm) 벤치마크 실행...
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" scripts\tests\benchmark\benchmark_reranker.py
) else (
  python scripts\tests\benchmark\benchmark_reranker.py
)
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
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" app\main.py setup --embed-only
) else (
  python app\main.py setup --embed-only
)
exit /b !ERRORLEVEL!

:ensure_ml_ready
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" app\main.py setup --models
) else (
  python app\main.py setup --models
)
exit /b !ERRORLEVEL!

:ensure_llama_ready
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" app\main.py setup --engine
) else (
  python app\main.py setup --engine
)
exit /b !ERRORLEVEL!

rem ------------------------------------------------------------
:run_tests
echo [local-rag] 단위 테스트 실행 (scripts/tests)...
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -m unittest discover -s scripts\tests
) else (
  python -m unittest discover -s scripts\tests
)
pause
exit /b !ERRORLEVEL!

rem ------------------------------------------------------------
:usage
echo.
echo  local-rag.bat [serve^|ingest^|ask^|llama^|models^|benchmark^|test]
echo    serve      웹 UI 실행 (기본값)
echo    ingest     PDF 문서 인제스트 및 벡터 색인
echo    ask        터미널에서 직접 질문 질의
echo    llama      llama-server 설치 및 기동만 실행
echo    models     임베딩 및 리랭커 모델 다운로드만 실행
echo    benchmark  리랭커 3종(origin/laya/clm) 성능 및 가용성 측정
echo    test       단위 테스트 실행 (scripts/tests)
echo.
echo  [주요 옵션 환경 변수]
echo    RERANK_STRATEGY   : origin (기본, Qwen) ^| laya (Multilingual) ^| clm (Contrastive-LM)
echo    RERANK_TOP_K      : 4 (최종 선별 청크 수, 0=리랭킹 끄기)
echo    LLAMA_CPP_VARIANT : vulkan (iGPU 범용) ^| cuda (NVIDIA) ^| cpu (CPU 전용)
echo    LLAMA_NGL         : 999 (GPU 전량 오프로드) ^| 0 (CPU 전용)
echo.
echo  현재 설정: EMBED=!EMBED_MODEL!  RERANK=!RERANK_MODEL! [!RERANK_STRATEGY!]  ENGINE=!LLAMA_CPP_VARIANT!
echo.
exit /b 0
