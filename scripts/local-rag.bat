@echo off

chcp 65001 >nul 2>&1

setlocal EnableDelayedExpansion

cd /d "%~dp0.."



rem ============================================================

rem  사용자 설정

rem ============================================================

set "EMBED_MODEL=Qwen/Qwen3-Embedding-0.6B"

set "RERANK_MODEL=Qwen/Qwen3-Reranker-0.6B"

set "FAISS_TOP_K=8"

set "RERANK_TOP_K=4"

set "HYBRID_BM25=1"

set "BM25_TOP_K=20"

rem  리랭크 끄기: set "RERANK_TOP_K=0"  (RRF/dense 상위 FAISS_TOP_K 개를 LLM에)

set "CHUNK_MIN_CHARS=200"

set "CHUNK_MAX_CHARS=800"

set "CHAT_TEMPERATURE=0.2"

set "LLAMA_CPP_BASE_URL=http://localhost:8080/v1"

set "LLAMA_NUM_PREDICT=4096"
set "LLAMA_NUM_PREDICT_THINK=16384"
set "THINK_AUTO_MIN_CHARS=200"

set "WEB_PORT=8787"



rem --- llama.cpp 자동 설치·기동 (scripts\ensure_llama_server.py) ---

rem  variant: vulkan ^| cuda ^| cpu  (비우면 NVIDIA 있으면 cuda, 없으면 vulkan)

set "LLAMA_CPP_VARIANT=vulkan"

rem  고정 릴리스 태그 (비우면 latest). 예: b9851

rem set "LLAMA_CPP_RELEASE_TAG="

set "LLAMA_CTX_SIZE=4096"

rem --- CPU+GPU 자원 최대 활용 (4800U: 8C/16T + Vega iGPU, Vulkan) ---

rem  GPU 레이어: 999 = 가능한 전부 GPU 오프로딩, 안 올라간 레이어는 CPU 처리

set "LLAMA_NGL=999"

rem  CPU 스레드: -t 생성(물리코어 8) / -tb 프롬프트 처리(논리 16)

set "LLAMA_THREADS=8"

set "LLAMA_THREADS_BATCH=16"

rem  Flash Attention (문제 시 off 로 변경)

set "LLAMA_FLASH_ATTN=on"

rem  배치 크기 (프롬프트 처리 속도, iGPU 공유메모리 고려 보수값)

set "LLAMA_BATCH_SIZE=1024"

set "LLAMA_UBATCH_SIZE=256"

rem  iGPU가 CPU보다 느리면 중간 오프로딩 비교: set "LLAMA_NGL=20"

rem  Qwen thinking: auto=요청별 enable_thinking 제어 (Instant/Think UI)

set "LLAMA_REASONING=auto"

rem  전역으로 thinking 강제 OFF (Think 모드도 막힘):

rem set "LLAMA_REASONING=off"

rem set "LLAMA_REASONING_BUDGET=0"



rem  채팅 GGUF (Hugging Face 자동 다운로드)

rem  9B (~5.8GB, Qwen3.5 9B Q4):

set "LLAMA_GGUF_REPO=bartowski/Qwen_Qwen3.5-9B-GGUF"

set "LLAMA_GGUF_FILE=Qwen_Qwen3.5-9B-Q4_K_M.gguf"

rem  4B (~2.6GB) — 현재 사용

rem set "LLAMA_GGUF_REPO=jc-builds/Qwen3.5-4B-Q4_K_M-GGUF"

rem set "LLAMA_GGUF_FILE=Qwen3.5-4B-Q4_K_M.gguf"

rem  직접 URL (단일 파일만; Qwen 공식 2.5 q4_k_m 은 분할이라 URL 404):

rem set "LLAMA_GGUF_REPO="

rem set "LLAMA_GGUF_URL=https://huggingface.co/bartowski/Qwen_Qwen3.5-9B-GGUF/resolve/main/Qwen_Qwen3.5-9B-Q4_K_M.gguf"



set "ACTION=%~1"

if "!ACTION!"=="" set "ACTION=serve"



if /i "!ACTION!"=="serve" goto :serve

if /i "!ACTION!"=="ingest" goto :ingest

if /i "!ACTION!"=="ask" goto :ask

if /i "!ACTION!"=="llama" goto :llama_only

if /i "!ACTION!"=="models" goto :models_only

if /i "!ACTION!"=="help" goto :usage

if /i "!ACTION!"=="/?" goto :usage



echo [오류] 알 수 없는 명령: !ACTION!

goto :usage



:serve

call :ensure_ml_ready

if errorlevel 1 (

    echo [경고] 임베딩·리랭커 준비 실패 — 질의가 실패할 수 있습니다.

)

call :ensure_llama_ready

if errorlevel 1 (

    echo [경고] llama-server 준비 실패 — 웹 UI만 띄웁니다. 질의는 실패할 수 있습니다.

)

echo [local-rag] 웹 UI  port=!WEB_PORT!  llama=!LLAMA_CPP_BASE_URL!
echo [local-rag] CHAT=!LLAMA_GGUF_FILE!
echo [local-rag] EMBED=!EMBED_MODEL!  RERANK=!RERANK_MODEL!

if exist ".venv\Scripts\python.exe" (

  ".venv\Scripts\python.exe" app\main.py serve

) else (

  python app\main.py serve

)

exit /b 0



:ingest

call :ensure_ml_embed

if errorlevel 1 (

    echo [오류] 임베딩 모델 !EMBED_MODEL! 준비 실패

    pause

    exit /b 1

)

echo [local-rag] PDF 인제스트  EMBED=!EMBED_MODEL!  Java 11+ 필요

if exist ".venv\Scripts\python.exe" (

  ".venv\Scripts\python.exe" app\main.py ingest --reset

) else (

  python app\main.py ingest --reset

)

pause

exit /b 0



:ask

call :ensure_ml_ready

if errorlevel 1 (

    echo [오류] 임베딩·리랭커 모델 준비 실패

    pause

    exit /b 1

)

call :ensure_llama_ready

if errorlevel 1 (

    echo [오류] llama-server 가 필요합니다.

    pause

    exit /b 1

)

set "LR_Q=%~2"

if "!LR_Q!"=="" (

  set /p LR_Q="질문: "

)

if exist ".venv\Scripts\python.exe" (

  ".venv\Scripts\python.exe" app\main.py ask --question "!LR_Q!"

) else (

  python app\main.py ask --question "!LR_Q!"

)

pause

exit /b 0



:llama_only

call :ensure_llama_ready

pause

exit /b 0



:models_only

call :ensure_ml_ready

pause

exit /b 0



rem --- 임베딩만 (ingest) ---

:ensure_ml_embed

if exist ".venv\Scripts\python.exe" (

  ".venv\Scripts\python.exe" "scripts\ensure_ml_models.py" --embed-only

) else (

  python "scripts\ensure_ml_models.py" --embed-only

)

exit /b !ERRORLEVEL!



rem --- 임베딩 + 리랭커 (ask / serve) ---

:ensure_ml_ready

if exist ".venv\Scripts\python.exe" (

  ".venv\Scripts\python.exe" "scripts\ensure_ml_models.py"

) else (

  python "scripts\ensure_ml_models.py"

)

exit /b !ERRORLEVEL!



:ensure_llama_ready

if exist ".venv\Scripts\python.exe" (

  ".venv\Scripts\python.exe" "scripts\ensure_llama_server.py"

) else (

  python "scripts\ensure_llama_server.py"

)

exit /b !ERRORLEVEL!



:usage

echo.

echo  local-rag.bat [serve^|ingest^|ask^|llama^|models]

echo    serve   웹 UI (기본) — models + llama-server 자동 준비

echo    ingest  PDF 인제스트 — embed 모델만 (Java 11+)

echo    ask     터미널 질문 — models + llama-server

echo    llama   llama-server 설치·GGUF·기동만

echo    models  임베딩+리랭커 다운로드만

echo.

echo  data\models\embeddings  data\models\reranker  data\models\chat

echo  EMBED=!EMBED_MODEL!  RERANK=!RERANK_MODEL!

echo.

exit /b 0

