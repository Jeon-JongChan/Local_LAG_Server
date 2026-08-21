# local-rag

로컬 PDF 문서 RAG — OpenDataLoader PDF → FAISS/SQLite(+BM25) → Qwen3-Embedding + Reranker → llama.cpp(Qwen).

## 파이프라인

```
PDF → OpenDataLoader PDF → JSON (page·bbox)
  → 청크 분할 → Qwen3-Embedding → FAISS + SQLite FTS5(BM25)

질문 → dense(FAISS) + BM25 → RRF 병합 → Qwen3-Reranker Top-4 → llama-server → 답변+출처
```

웹 UI는 세션·히스토리와 think 모드(`instant` / `think` / `auto`)를 지원합니다.

## 준비

1. **Python 3.10+**
2. **Java 11+** (`java -version`) — OpenDataLoader PDF 필수
3. **llama-server** (llama.cpp) — 채팅용 Qwen GGUF 로드 후 실행

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# Linux
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# 또는 다른 PC에서 옮긴 .venv 복구: ./scripts/reinstall_venv.sh
```

### llama-server

`scripts\local-rag.bat` 가 **llama-server 미실행 시** 자동으로:

1. GitHub에서 llama.cpp Windows zip 다운로드 → `tools/llama-cpp/`
2. Hugging Face에서 GGUF 다운로드 → `data/models/`（기본: **Qwen3.5 9B Q4**, Ollama `qwen3.5:9b` 대응）
3. 새 창에서 `llama-server` 기동

배치 맨 위 `LLAMA_GGUF_REPO` / `LLAMA_GGUF_FILE`, `LLAMA_CPP_VARIANT`(vulkan/cuda/cpu) 로 조정합니다.

수동 실행 예:

```bash
llama-server -m path\to\qwen.gguf --port 8080 -c 4096
```

환경 변수 `LLAMA_CPP_BASE_URL=http://localhost:8080/v1`

## 설정 (환경 변수)

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `EMBED_MODEL` | `Qwen/Qwen3-Embedding-0.6B` | 임베딩 |
| `RERANK_MODEL` | `Qwen/Qwen3-Reranker-0.6B` | 리랭커 |
| `FAISS_TOP_K` | `20` | dense 검색 후보 |
| `BM25_TOP_K` | `20` | BM25 후보 |
| `HYBRID_BM25` | `1` | dense+BM25 RRF (`0`이면 dense만) |
| `RERANK_TOP_K` | `4` | 리랭크 후 LLM 컨텍스트 |
| `CHUNK_MIN_CHARS` / `CHUNK_MAX_CHARS` | `200` / `800` | 청크 크기 |
| `LLAMA_CPP_BASE_URL` | `http://localhost:8080/v1` | llama-server |
| `LLAMA_CPP_MODEL` | (비움) | 모델 alias (선택) |
| `LLAMA_NUM_PREDICT` | `4096` | instant 모드 최대 토큰 |
| `LLAMA_NUM_PREDICT_THINK` | `16384` | think 모드 최대 토큰 |
| `REQUEST_TIMEOUT` | `600` | LLM HTTP 타임아웃(초) |
| `CHAT_HISTORY_MAX_MESSAGES` | `20` | LLM에 넣을 이전 메시지 수 |
| `WEB_PORT` | `8787` | 웹 UI 포트 |

## 사용

`data/docs/` 에 **PDF** 를 넣습니다.

```bash
python app/main.py ingest --reset
python app/main.py ask --question "문서에서 ○○은?"
python app/main.py serve
```

Windows: `scripts\local-rag.bat` (인자: `ingest`, `ask`, `serve`, `llama`, `models`)

Linux:

```bash
chmod +x scripts/*.sh
./scripts/local-rag.sh serve          # 웹 UI (models + llama-server 자동)
./scripts/local-rag.sh ingest         # PDF 인제스트
./scripts/local-rag.sh ask "질문"
./scripts/labeler.sh                  # 라벨링 도구 :8788
./scripts/reinstall_venv.sh           # .venv lib 보존 후 재구성
```

Linux llama 기본 빌드는 `cpu`(NVIDIA면 `vulkan`). 공식 Ubuntu 릴리스에 CUDA 패키지는 없어 `LLAMA_CPP_VARIANT=cuda` 는 vulkan/cpu로 대체됩니다.

**모델 변경 후**에는 반드시 `ingest --reset` 으로 재색인하세요 (bge-m3 → Qwen3 등).

인덱스: `data/index/` (`chunks.sqlite3`, `faiss.index`)

모델 로컬 저장 (`data/models/`):

| 경로 | 용도 |
|------|------|
| `chat/` | Qwen GGUF (llama-server) |
| `embeddings/Qwen3-Embedding-0.6B/` | 임베딩 |
| `reranker/Qwen3-Reranker-0.6B/` | 리랭커 |

없으면 `local-rag.bat` / `ensure_ml_models.py` / `app/models.py` 가 Hugging Face에서 자동 다운로드합니다.

### 웹 API (요약)

- 주력: `GET/POST /api/sessions`, `GET/DELETE /api/sessions/{id}`, `POST /api/sessions/{id}/messages` (`content`, `think_mode`)
- 레거시: `POST /api/ask` — `{"question"}` → `{"answer"}` (세션·히스토리 없음)

상세 스키마·환경 변수는 `overview_detail.txt` 참고.

## 주의

- **PDF만** 지원합니다 (HWP/HWPX는 미지원).
- OpenDataLoader는 PDF당 JVM 기동 비용이 있어, 배치 인제스트를 권장합니다.
- 파싱 실패 PDF는 `data/rejected/` 로 격리됩니다.
