"""
local-rag 전역 설정 모듈.

모든 경로·모델명·하이퍼파라미터를 한곳에서 정의합니다.
환경 변수가 있으면 기본값을 덮어씁니다 (os.getenv).
배치 파일(local-rag.bat)의 set 값도 동일한 이름으로 Python에 전달됩니다.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# 디렉터리·파일 경로 (프로젝트 루트 기준, 환경 변수로 변경 불가)
# ---------------------------------------------------------------------------
# app/settings.py → parent.parent = 프로젝트 루트 (local-rag/)
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DOCS_DIR = DATA_DIR / "docs"          # 색인할 PDF를 넣는 폴더
REJECTED_DIR = DATA_DIR / "rejected"  # 파싱 실패·손상 PDF 격리
INDEX_DIR = DATA_DIR / "index"        # FAISS·SQLite·meta.json 저장
PARSED_DIR = DATA_DIR / "parsed"      # OpenDataLoader 임시 JSON 출력

SQLITE_PATH = INDEX_DIR / "chunks.sqlite3"   # 청크 본문·메타 DB
FAISS_PATH = INDEX_DIR / "faiss.index"       # 벡터 인덱스 파일
INDEX_META_PATH = INDEX_DIR / "meta.json"    # 차원·모델명·청크 수 등

# ML 모델 로컬 저장 (Hugging Face snapshot / GGUF)
MODELS_DIR = DATA_DIR / "models"
CHAT_MODELS_DIR = MODELS_DIR / "chat"           # llama-server GGUF
EMBED_MODELS_DIR = MODELS_DIR / "embeddings"    # Qwen3-Embedding 등
RERANK_MODELS_DIR = MODELS_DIR / "reranker"     # Qwen3-Reranker 등


def hf_repo_local_name(repo_id: str) -> str:
    """Qwen/Qwen3-Embedding-0.6B → Qwen3-Embedding-0.6B (하위 폴더명)."""
    return repo_id.strip().split("/")[-1] or "model"


def hf_model_dir(repo_id: str, category: Path) -> Path:
    return category / hf_repo_local_name(repo_id)

# ---------------------------------------------------------------------------
# OpenDataLoader PDF
# ---------------------------------------------------------------------------
# Java 11+ 가 PATH에 있어야 함. "json" 출력 시 page/bbox 메타 추출 가능
OPENDATALOADER_FORMAT = os.getenv("OPENDATALOADER_FORMAT", "json")

# ---------------------------------------------------------------------------
# 청크 분할 (문자 수) — chunker.py 에서 사용
# ---------------------------------------------------------------------------
# OpenDataLoader가 뽑은 작은 요소(문단 등)를 이 길이 이상이 될 때까지 병합
CHUNK_MIN_CHARS = int(os.getenv("CHUNK_MIN_CHARS", "200"))
# 한 청크 최대 길이. 초과 시 overlap 을 두고 여러 청크로 분할
CHUNK_MAX_CHARS = int(os.getenv("CHUNK_MAX_CHARS", "800"))
# 긴 텍스트 분할 시 앞뒤 청크가 겹치는 문자 수 (문맥 유지)
CHUNK_OVERLAP_CHARS = int(os.getenv("CHUNK_OVERLAP_CHARS", "80"))

# ---------------------------------------------------------------------------
# 임베딩 · 1차 검색 · 리랭킹
# ---------------------------------------------------------------------------
EMBED_MODEL = os.getenv("EMBED_MODEL", "Qwen/Qwen3-Embedding-0.6B")
RERANK_MODEL = os.getenv("RERANK_MODEL", "Qwen/Qwen3-Reranker-0.6B")
FAISS_TOP_K = int(os.getenv("FAISS_TOP_K", "8"))      # dense 후보 (=리랭크 입력 상한)
BM25_TOP_K = int(os.getenv("BM25_TOP_K", "16"))        # BM25(FTS5) 후보 개수
# Reciprocal Rank Fusion 상수 (클수록 순위 차이 완화)
HYBRID_RRF_K = int(os.getenv("HYBRID_RRF_K", "60"))
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "4"))     # 리랭크 후 LLM 개수 (0=리랭크 끄기)
# True/1/yes: dense+BM25 RRF 후 리랭크. False면 dense만
HYBRID_BM25 = os.getenv("HYBRID_BM25", "1").strip().lower() not in (
    "0",
    "false",
    "no",
    "off",
)

# ---------------------------------------------------------------------------
# llama.cpp — llama-server OpenAI 호환 REST API
# ---------------------------------------------------------------------------
LLAMA_CPP_BASE_URL = os.getenv("LLAMA_CPP_BASE_URL", "http://localhost:8080/v1").rstrip("/")
# 비어 있으면 요청 payload 에 model 키를 넣지 않음 (서버 기본 모델 사용)
LLAMA_CPP_MODEL = os.getenv("LLAMA_CPP_MODEL", "").strip() or None
LLAMA_CPP_API_KEY = os.getenv("LLAMA_CPP_API_KEY", "no-key")
CHAT_TEMPERATURE = float(os.getenv("CHAT_TEMPERATURE", "0.2"))
# instant(기본) / auto→instant 최대 생성 토큰
LLAMA_NUM_PREDICT = int(os.getenv("LLAMA_NUM_PREDICT", "4096"))
# thinking 모드·auto→think 최대 생성 토큰 (추론+답변 합산)
LLAMA_NUM_PREDICT_THINK = int(os.getenv("LLAMA_NUM_PREDICT_THINK", "16384"))
# auto 모드: 질문이 이 글자 수 이상이면 think 로 전환
THINK_AUTO_MIN_CHARS = int(os.getenv("THINK_AUTO_MIN_CHARS", "200"))
# LLM에 넣을 이전 대화 메시지 수 (user+assistant 합산, 0=체이닝 끔)
CHAT_HISTORY_MAX_MESSAGES = int(os.getenv("CHAT_HISTORY_MAX_MESSAGES", "20"))
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "600"))    # HTTP 타임아웃(초)

# ---------------------------------------------------------------------------
# 웹 UI (web_server.py)
# ---------------------------------------------------------------------------
WEB_HOST = os.getenv("WEB_HOST", "0.0.0.0")
WEB_PORT = int(os.getenv("WEB_PORT", "8787"))

# ---------------------------------------------------------------------------
# 임베딩·리랭커 연산 디바이스
# ---------------------------------------------------------------------------
# "auto" = CUDA 가능하면 gpu, 아니면 cpu / "cuda" / "cpu" 로 고정 가능
EMBED_DEVICE = os.getenv("EMBED_DEVICE", "auto")
